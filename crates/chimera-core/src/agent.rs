use crate::config::Config;
use crate::session::Session;
use chimera_providers::types::*;
use chimera_providers::Provider;
use chimera_tools::{ToolRegistry, execute_tool};
use tokio::sync::mpsc;
use tracing::{debug, info, warn};

/// Callback for streaming events to the UI layer.
pub type EventCallback = Box<dyn Fn(AgentEvent) + Send>;

/// Events emitted by the agent during a turn.
#[derive(Debug, Clone)]
pub enum AgentEvent {
    /// Streaming text content from the model.
    TextDelta(String),
    /// A tool is about to be called.
    ToolStart { name: String, arguments: String },
    /// A tool finished executing.
    ToolResult { name: String, result: String, is_error: bool },
    /// The turn is complete.
    TurnComplete { text: Option<String>, iterations: usize },
    /// Token usage for this turn.
    Usage { input_tokens: u32, output_tokens: u32 },
    /// An error occurred.
    Error(String),
}

/// The core agent that drives the model-tool interaction loop.
pub struct Agent {
    provider: Box<dyn Provider>,
    model: String,
    config: Config,
    session: Session,
    tool_registry: ToolRegistry,
}

impl Agent {
    /// Create a new agent with the given provider and config.
    pub fn new(
        provider: Box<dyn Provider>,
        model: String,
        config: Config,
    ) -> Self {
        let mut session = Session::new();
        session.set_system_prompt(&config.system_prompt());

        Self {
            provider,
            model,
            config,
            session,
            tool_registry: ToolRegistry::with_builtins(),
        }
    }

    /// Get a reference to the session.
    pub fn session(&self) -> &Session {
        &self.session
    }

    /// Run a single turn: take user input, call the model, execute tools in a
    /// loop until the model produces a final text response.
    ///
    /// This is the core agent loop, inspired by the patterns in Codex and
    /// claw-code-parity: symmetric turn execution where all tool calls from a
    /// response are executed before the next API call.
    pub async fn run_turn(
        &mut self,
        user_input: &str,
        on_event: &EventCallback,
    ) -> anyhow::Result<Option<String>> {
        self.session.push_user(user_input);

        let tool_definitions = self.tool_registry.definitions();

        let mut iterations = 0;
        let mut final_text: Option<String> = None;

        loop {
            iterations += 1;
            if iterations > self.config.max_iterations {
                warn!("Hit max iterations ({}) — forcing stop", self.config.max_iterations);
                on_event(AgentEvent::Error(format!(
                    "Reached maximum tool iterations ({}). Stopping.",
                    self.config.max_iterations
                )));
                break;
            }

            // Build the request
            let request = ChatRequest {
                model: self.model.clone(),
                messages: self.session.messages.clone(),
                tools: if tool_definitions.is_empty() {
                    None
                } else {
                    Some(
                        tool_definitions
                            .iter()
                            .cloned()
                            .map(|v| serde_json::from_value(v).unwrap())
                            .collect(),
                    )
                },
                temperature: self.config.temperature,
                max_tokens: self.config.max_tokens,
                stream: true,
            };

            // Stream the response
            let (tx, mut rx) = mpsc::unbounded_channel::<chimera_providers::StreamEvent>();

            let provider = &self.provider;
            let stream_result = provider.chat_stream(request, tx).await;

            if let Err(e) = stream_result {
                on_event(AgentEvent::Error(format!("Provider error: {e}")));
                return Err(e);
            }

            // Collect streaming events into a full response
            let mut response_text = String::new();
            let mut response: Option<ChatResponse> = None;

            while let Some(event) = rx.recv().await {
                match event {
                    chimera_providers::StreamEvent::ContentDelta(text) => {
                        response_text.push_str(&text);
                        on_event(AgentEvent::TextDelta(text));
                    }
                    chimera_providers::StreamEvent::ToolCallDelta { .. } => {
                        // Tool call deltas are assembled internally
                    }
                    chimera_providers::StreamEvent::Done(resp) => {
                        response = Some(resp);
                    }
                    chimera_providers::StreamEvent::Error(e) => {
                        on_event(AgentEvent::Error(e));
                    }
                }
            }

            let response = match response {
                Some(r) => r,
                None => {
                    on_event(AgentEvent::Error("No response received from model".into()));
                    break;
                }
            };

            // Track usage
            if let Some(usage) = &response.usage {
                self.session
                    .record_usage(usage.prompt_tokens, usage.completion_tokens);
                on_event(AgentEvent::Usage {
                    input_tokens: usage.prompt_tokens,
                    output_tokens: usage.completion_tokens,
                });
            }

            // If no tool calls, this is the final response
            if response.tool_calls.is_empty() {
                let text = response.content.clone();
                if let Some(t) = &text {
                    self.session.push_assistant_text(t);
                }
                final_text = text;
                on_event(AgentEvent::TurnComplete {
                    text: final_text.clone(),
                    iterations,
                });
                break;
            }

            // Record the assistant message with tool calls
            self.session
                .push_assistant_tool_calls(response.content.clone(), response.tool_calls.clone());

            // Execute each tool call
            for tool_call in &response.tool_calls {
                let tool_name = &tool_call.function.name;
                let tool_args = &tool_call.function.arguments;

                info!("Tool call: {tool_name}");
                debug!("Tool args: {tool_args}");

                on_event(AgentEvent::ToolStart {
                    name: tool_name.clone(),
                    arguments: tool_args.clone(),
                });

                let result = execute_tool(tool_name, tool_args).await;

                let (output, is_error) = match result {
                    Ok(output) => (output, false),
                    Err(e) => (format!("Error: {e}"), true),
                };

                on_event(AgentEvent::ToolResult {
                    name: tool_name.clone(),
                    result: output.clone(),
                    is_error,
                });

                self.session.push_tool_result(&tool_call.id, &output);
            }

            // Loop back to get the model's next response
            debug!("Tool execution complete, continuing agent loop (iteration {iterations})");
        }

        Ok(final_text)
    }
}
