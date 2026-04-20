use crate::config::{ApprovalMode, Config};
use crate::session::Session;
use chimera_sigil_providers::Provider;
use chimera_sigil_providers::types::*;
use chimera_sigil_tools::{PermissionLevel, ToolRegistry, execute_tool, resolve_alias};
use serde::Serialize;
use serde_json::Value;
use tokio::sync::mpsc;
use tracing::{debug, info, warn};

/// Decision returned by the approval callback.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApprovalDecision {
    /// Allow this specific tool call.
    Allow,
    /// Allow all tool calls for the rest of the session.
    AllowAll,
    /// Deny this tool call.
    Deny,
}

/// Callback for interactive tool approval prompts.
/// Receives tool name, arguments JSON, and permission level.
pub type ApprovalCallback =
    Box<dyn Fn(&str, &str, PermissionLevel) -> ApprovalDecision + Send + Sync>;

/// Callback for streaming events to the UI layer.
pub type EventCallback = Box<dyn Fn(AgentEvent) + Send>;

/// Events emitted by the agent during a turn.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AgentEvent {
    /// The current session identifier.
    SessionReady { session_id: String },
    /// Streaming text content from the model.
    TextDelta { text: String },
    /// A tool is about to be called.
    ToolStart {
        tool_call_id: String,
        name: String,
        arguments: String,
    },
    /// A tool finished executing.
    ToolResult {
        tool_call_id: String,
        name: String,
        result: String,
        is_error: bool,
    },
    /// A tool was denied by the user.
    ToolDenied {
        tool_call_id: String,
        name: String,
        reason: String,
    },
    /// The turn is complete.
    TurnComplete {
        text: Option<String>,
        iterations: usize,
        session_id: String,
    },
    /// The session was saved to disk.
    SessionSaved { session_id: String, path: String },
    /// Token usage for this turn.
    Usage {
        input_tokens: u32,
        output_tokens: u32,
    },
    /// Provider request failed and will be retried.
    ProviderRetry {
        attempt: usize,
        max_attempts: usize,
        delay_ms: u64,
        error: String,
    },
    /// The session was compacted to keep the loop within context limits.
    ContextCompacted {
        removed_messages: usize,
        estimated_tokens: usize,
        context_window: usize,
    },
    /// An error occurred.
    Error { message: String },
}

/// Outcome of a completed agent turn.
#[derive(Debug, Clone, Serialize)]
pub struct TurnOutcome {
    /// Final text response from the model, if any.
    pub text: Option<String>,
    /// How the turn ended.
    pub exit_reason: ExitReason,
}

/// Why an agent turn ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ExitReason {
    /// Model produced a final text response.
    Complete,
    /// Hit the maximum iteration limit.
    MaxIterations,
    /// Stream ended unexpectedly.
    StreamError,
    /// A tool execution failed.
    ToolError,
}

/// The core agent that drives the model-tool interaction loop.
pub struct Agent {
    provider: Box<dyn Provider>,
    model: String,
    config: Config,
    session: Session,
    tool_registry: ToolRegistry,
    approval_callback: Option<ApprovalCallback>,
    session_approved: bool,
}

fn approval_denial_reason(mode: ApprovalMode) -> &'static str {
    match mode {
        ApprovalMode::Prompt => "Tool execution denied by user.",
        ApprovalMode::Approve => "Tool execution denied by approval mode 'approve'.",
        ApprovalMode::Full => "Tool execution denied.",
    }
}

impl Agent {
    /// Create a new agent with the given provider and config.
    pub fn new(provider: Box<dyn Provider>, model: String, config: Config) -> Self {
        let mut session = Session::new();
        session.set_system_prompt(&config.system_prompt());

        Self {
            provider,
            model,
            config,
            session,
            tool_registry: ToolRegistry::with_builtins(),
            approval_callback: None,
            session_approved: false,
        }
    }

    /// Set the interactive approval callback for tool permission checks.
    pub fn set_approval_callback(&mut self, callback: ApprovalCallback) {
        self.approval_callback = Some(callback);
    }

    /// Get a reference to the session.
    pub fn session(&self) -> &Session {
        &self.session
    }

    /// Get a mutable reference to the session.
    pub fn session_mut(&mut self) -> &mut Session {
        &mut self.session
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
    ) -> anyhow::Result<TurnOutcome> {
        self.session.push_user(user_input);

        let tool_definitions = self.tool_registry.definitions();

        let mut iterations = 0;
        let mut final_text: Option<String> = None;
        let mut exit_reason = ExitReason::Complete;
        let mut had_tool_error = false;

        loop {
            iterations += 1;
            if iterations > self.config.max_iterations {
                warn!(
                    "Hit max iterations ({}) — forcing stop",
                    self.config.max_iterations
                );
                on_event(AgentEvent::Error {
                    message: format!(
                        "Reached maximum tool iterations ({}). Stopping.",
                        self.config.max_iterations
                    ),
                });
                exit_reason = ExitReason::MaxIterations;
                break;
            }

            // Build the request
            let request = ChatRequest {
                model: self.model.clone(),
                messages: self.session.messages.clone(),
                tools: if tool_definitions.is_empty() {
                    None
                } else {
                    let defs: Result<Vec<_>, _> = tool_definitions
                        .iter()
                        .cloned()
                        .map(serde_json::from_value)
                        .collect();
                    Some(defs.map_err(|e| {
                        anyhow::anyhow!("Failed to serialize tool definitions: {e}")
                    })?)
                },
                temperature: self.config.temperature,
                max_tokens: self.config.max_tokens,
                stream: true,
            };

            // Stream the response, retrying provider setup/connect failures.
            let max_attempts = self.config.provider_retries.saturating_add(1).max(1);
            let mut rx = None;
            let mut last_error = None;

            for attempt in 1..=max_attempts {
                let (tx, attempt_rx) =
                    mpsc::unbounded_channel::<chimera_sigil_providers::StreamEvent>();
                match self.provider.chat_stream(request.clone(), tx).await {
                    Ok(()) => {
                        rx = Some(attempt_rx);
                        last_error = None;
                        break;
                    }
                    Err(e) => {
                        let message = e.to_string();
                        last_error = Some(e);
                        if attempt < max_attempts {
                            let delay_ms =
                                retry_delay_ms(self.config.provider_retry_backoff_ms, attempt);
                            on_event(AgentEvent::ProviderRetry {
                                attempt,
                                max_attempts,
                                delay_ms,
                                error: message,
                            });
                            tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;
                        }
                    }
                }
            }

            let Some(mut rx) = rx else {
                let error = last_error
                    .map(|e| e.to_string())
                    .unwrap_or_else(|| "provider failed without an error".into());
                on_event(AgentEvent::Error {
                    message: format!("Provider error after {max_attempts} attempt(s): {error}"),
                });
                anyhow::bail!("Provider error after {max_attempts} attempt(s): {error}");
            };

            // Collect streaming events into a full response
            let mut response_text = String::new();
            let mut response: Option<ChatResponse> = None;

            while let Some(event) = rx.recv().await {
                match event {
                    chimera_sigil_providers::StreamEvent::ContentDelta(text) => {
                        response_text.push_str(&text);
                        on_event(AgentEvent::TextDelta { text });
                    }
                    chimera_sigil_providers::StreamEvent::ToolCallDelta { .. } => {
                        // Tool call deltas are assembled internally
                    }
                    chimera_sigil_providers::StreamEvent::Done(resp) => {
                        response = Some(resp);
                    }
                    chimera_sigil_providers::StreamEvent::Error(e) => {
                        on_event(AgentEvent::Error { message: e });
                    }
                }
            }

            let response = match response {
                Some(r) => r,
                None => {
                    let err_msg = if response_text.is_empty() {
                        "No response received from model (stream closed unexpectedly)".to_string()
                    } else {
                        format!(
                            "Stream ended without completion. Partial text received: {}...",
                            &response_text[..response_text.len().min(100)]
                        )
                    };
                    on_event(AgentEvent::Error { message: err_msg });
                    exit_reason = ExitReason::StreamError;
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

            let mut tool_calls = response.tool_calls.clone();
            let mut tool_call_content = response.content.clone();
            if tool_calls.is_empty()
                && let Some(content) = response.content.as_deref()
                && let Some((assistant_content, parsed_tool_calls)) =
                    extract_textual_tool_calls(content)
            {
                tool_calls = parsed_tool_calls;
                tool_call_content = assistant_content;
            }

            // If no tool calls, this is the final response
            if tool_calls.is_empty() {
                let text = response.content.clone();
                if let Some(t) = &text {
                    self.session.push_assistant_text(t);
                }
                final_text = text;
                on_event(AgentEvent::TurnComplete {
                    text: final_text.clone(),
                    iterations,
                    session_id: self.session.id.clone(),
                });
                break;
            }

            // Record the assistant message with tool calls
            self.session
                .push_assistant_tool_calls(tool_call_content, tool_calls.clone());

            // Auto-compact if approaching context window limit (85% threshold)
            let ctx_window = self.config.context_window();
            let estimated = self.session.estimate_context_tokens();
            let threshold = ctx_window * 85 / 100;
            if estimated > threshold {
                let keep = 20; // keep last 20 messages
                let removed = self.session.compact(keep);
                if removed > 0 {
                    info!(
                        "Context compaction: removed {removed} messages (est. {estimated}/{ctx_window} tokens)"
                    );
                    on_event(AgentEvent::ContextCompacted {
                        removed_messages: removed,
                        estimated_tokens: estimated,
                        context_window: ctx_window,
                    });
                }
            }

            // Execute each tool call
            for tool_call in &tool_calls {
                let tool_name = &tool_call.function.name;
                let tool_args = &tool_call.function.arguments;

                info!("Tool call: {tool_name}");
                debug!("Tool args: {tool_args}");

                // Permission check before execution
                let canonical = resolve_alias(tool_name);
                let permission = self
                    .tool_registry
                    .get(canonical)
                    .map(|spec| spec.permission)
                    .unwrap_or(PermissionLevel::Execute);

                let approved = if permission <= PermissionLevel::ReadOnly {
                    true
                } else if self.session_approved {
                    true
                } else {
                    match self.config.approval_mode {
                        ApprovalMode::Full => true,
                        ApprovalMode::Approve => permission <= PermissionLevel::WorkspaceWrite,
                        ApprovalMode::Prompt => {
                            if let Some(cb) = &self.approval_callback {
                                match cb(tool_name, tool_args, permission) {
                                    ApprovalDecision::Allow => true,
                                    ApprovalDecision::AllowAll => {
                                        self.session_approved = true;
                                        true
                                    }
                                    ApprovalDecision::Deny => false,
                                }
                            } else {
                                false
                            }
                        }
                    }
                };

                if !approved {
                    let reason = approval_denial_reason(self.config.approval_mode).to_string();
                    on_event(AgentEvent::ToolDenied {
                        tool_call_id: tool_call.id.clone(),
                        name: tool_name.clone(),
                        reason: reason.clone(),
                    });
                    self.session.push_tool_result(&tool_call.id, &reason);
                    continue;
                }

                on_event(AgentEvent::ToolStart {
                    tool_call_id: tool_call.id.clone(),
                    name: tool_name.clone(),
                    arguments: tool_args.clone(),
                });

                let result = execute_tool(tool_name, tool_args).await;

                let (output, is_error) = match result {
                    Ok(output) => (output, false),
                    Err(e) => {
                        had_tool_error = true;
                        (format!("Error: {e}"), true)
                    }
                };

                if is_error {
                    debug!("Tool '{tool_name}' failed");
                }

                on_event(AgentEvent::ToolResult {
                    tool_call_id: tool_call.id.clone(),
                    name: tool_name.clone(),
                    result: output.clone(),
                    is_error,
                });

                self.session.push_tool_result(&tool_call.id, &output);
            }

            // Loop back to get the model's next response
            debug!("Tool execution complete, continuing agent loop (iteration {iterations})");
        }

        if exit_reason == ExitReason::Complete && had_tool_error {
            exit_reason = ExitReason::ToolError;
        }

        Ok(TurnOutcome {
            text: final_text,
            exit_reason,
        })
    }
}

fn retry_delay_ms(initial_backoff_ms: u64, attempt: usize) -> u64 {
    let shift = attempt.saturating_sub(1).min(6) as u32;
    initial_backoff_ms.saturating_mul(2u64.saturating_pow(shift))
}

fn extract_textual_tool_calls(content: &str) -> Option<(Option<String>, Vec<ToolCall>)> {
    let trimmed = content.trim();
    if trimmed.is_empty() {
        return None;
    }

    if contains_structured_report_json(trimmed) {
        return None;
    }

    if let Ok(value) = serde_json::from_str::<Value>(trimmed) {
        match value {
            Value::Array(items) => {
                let calls = parse_textual_tool_call_values(items)?;
                return Some((None, calls));
            }
            other => {
                if let Some(call) = parse_textual_tool_call_value(other) {
                    return Some((None, vec![call.with_index(0)]));
                }
            }
        }
    }

    let mut calls = Vec::new();
    let mut text_lines = Vec::new();
    for line in content.lines() {
        let trimmed_line = line.trim();
        if trimmed_line.is_empty() {
            if !text_lines.is_empty() {
                text_lines.push(String::new());
            }
            continue;
        }

        match serde_json::from_str::<Value>(trimmed_line)
            .ok()
            .and_then(parse_textual_tool_call_value)
            .or_else(|| parse_textual_tool_call_line(trimmed_line))
        {
            Some(call) => calls.push(call.with_index(calls.len())),
            None => text_lines.push(line.to_string()),
        }
    }

    if calls.is_empty() {
        return extract_textual_tool_calls_from_fragments(content);
    }

    let assistant_content = text_lines.join("\n").trim().to_string();
    Some((
        if assistant_content.is_empty() {
            None
        } else {
            Some(assistant_content)
        },
        calls,
    ))
}

fn extract_textual_tool_calls_from_fragments(
    content: &str,
) -> Option<(Option<String>, Vec<ToolCall>)> {
    let fragments = outer_json_fragments(content);
    if fragments.is_empty() {
        return None;
    }

    let mut calls = Vec::new();
    let mut assistant_content = String::new();
    let mut cursor = 0;

    for (start, end) in fragments {
        if start > cursor {
            append_assistant_fragment(&mut assistant_content, &content[cursor..start]);
        }

        let fragment = &content[start..end];
        match serde_json::from_str::<Value>(fragment)
            .ok()
            .and_then(parse_textual_tool_call_value)
        {
            Some(call) => calls.push(call.with_index(calls.len())),
            None => append_assistant_fragment(&mut assistant_content, fragment),
        }

        cursor = end;
    }

    if cursor < content.len() {
        append_assistant_fragment(&mut assistant_content, &content[cursor..]);
    }

    if calls.is_empty() {
        return None;
    }

    Some((
        if assistant_content.is_empty() {
            None
        } else {
            Some(assistant_content)
        },
        calls,
    ))
}

fn append_assistant_fragment(buffer: &mut String, fragment: &str) {
    let trimmed = fragment.trim();
    if trimmed.is_empty() {
        return;
    }

    if !buffer.is_empty() {
        buffer.push('\n');
    }
    buffer.push_str(trimmed);
}

fn contains_structured_report_json(content: &str) -> bool {
    if parses_as_structured_report_json(content) {
        return true;
    }

    if content
        .lines()
        .any(|line| parses_as_structured_report_json(line.trim()))
    {
        return true;
    }

    contains_structured_report_json_fragment(content)
}

fn parses_as_structured_report_json(content: &str) -> bool {
    if content.is_empty() {
        return false;
    }

    serde_json::from_str::<Value>(content)
        .ok()
        .is_some_and(|value| value_has_report_shape(&value))
}

fn value_has_report_shape(value: &Value) -> bool {
    match value {
        Value::Object(object) => object.contains_key("task") && object.contains_key("target"),
        Value::Array(items) => items.iter().any(value_has_report_shape),
        _ => false,
    }
}

fn contains_structured_report_json_fragment(content: &str) -> bool {
    outer_json_fragments(content)
        .into_iter()
        .any(|(start, end)| parses_as_structured_report_json(&content[start..end]))
}

fn outer_json_fragments(content: &str) -> Vec<(usize, usize)> {
    let mut fragments = Vec::new();
    let mut start = None;
    let mut expected_closers = Vec::new();
    let mut in_string = false;
    let mut escaped = false;

    for (index, ch) in content.char_indices() {
        if start.is_none() {
            match ch {
                '{' => {
                    start = Some(index);
                    expected_closers.push('}');
                }
                '[' => {
                    start = Some(index);
                    expected_closers.push(']');
                }
                _ => {}
            }
            continue;
        }

        if in_string {
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                in_string = false;
            }
            continue;
        }

        match ch {
            '"' => in_string = true,
            '{' => expected_closers.push('}'),
            '[' => expected_closers.push(']'),
            '}' | ']' => {
                if expected_closers.last().copied() != Some(ch) {
                    start = None;
                    expected_closers.clear();
                    continue;
                }

                expected_closers.pop();
                if expected_closers.is_empty() {
                    fragments.push((start.unwrap(), index + ch.len_utf8()));
                    start = None;
                }
            }
            _ => {}
        }
    }

    fragments
}

fn parse_textual_tool_call_values(values: Vec<Value>) -> Option<Vec<ToolCall>> {
    let mut calls = Vec::new();
    for (index, value) in values.into_iter().enumerate() {
        calls.push(parse_textual_tool_call_value(value)?.with_index(index));
    }

    if calls.is_empty() { None } else { Some(calls) }
}

fn parse_textual_tool_call_value(value: Value) -> Option<TextualToolCall> {
    let Value::Object(mut object) = value else {
        return None;
    };

    let Some(Value::String(name)) = object.remove("name") else {
        return None;
    };
    let Some(arguments) = object.remove("arguments") else {
        return None;
    };

    let arguments = match arguments {
        Value::String(value) => value,
        other => serde_json::to_string(&other).ok()?,
    };

    Some(TextualToolCall { name, arguments })
}

fn parse_textual_tool_call_line(line: &str) -> Option<TextualToolCall> {
    let line = line.trim().trim_start_matches("-").trim();
    let rest = line.strip_prefix("name:")?.trim();
    let (name, arguments) = rest.split_once(", arguments:")?;
    let name = name.trim().trim_matches(['`', '"', '\'']).to_string();
    if name.is_empty() {
        return None;
    }

    let arguments = arguments.trim();
    let arguments = serde_json::from_str::<Value>(arguments)
        .ok()
        .and_then(|value| serde_json::to_string(&value).ok())
        .unwrap_or_else(|| arguments.to_string());

    Some(TextualToolCall { name, arguments })
}

struct TextualToolCall {
    name: String,
    arguments: String,
}

impl TextualToolCall {
    fn with_index(self, index: usize) -> ToolCall {
        ToolCall {
            id: format!("text_tool_call_{index}"),
            call_type: "function".into(),
            function: FunctionCall {
                name: self.name,
                arguments: self.arguments,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use chimera_sigil_providers::types::{FunctionCall, ToolCall};
    use chimera_sigil_providers::{Provider, ProviderKind};
    use std::collections::VecDeque;
    use std::sync::{Arc, Mutex};

    #[derive(Clone)]
    struct MockProvider {
        responses: Arc<Mutex<VecDeque<ChatResponse>>>,
        requests: Arc<Mutex<Vec<ChatRequest>>>,
    }

    impl MockProvider {
        fn new(responses: Vec<ChatResponse>) -> Self {
            Self {
                responses: Arc::new(Mutex::new(responses.into())),
                requests: Arc::new(Mutex::new(Vec::new())),
            }
        }

        fn requests(&self) -> Vec<ChatRequest> {
            self.requests.lock().unwrap().clone()
        }
    }

    #[async_trait]
    impl Provider for MockProvider {
        fn kind(&self) -> ProviderKind {
            ProviderKind::OpenAi
        }

        async fn chat_stream(
            &self,
            request: ChatRequest,
            tx: mpsc::UnboundedSender<chimera_sigil_providers::StreamEvent>,
        ) -> anyhow::Result<()> {
            self.requests.lock().unwrap().push(request);
            let response = self
                .responses
                .lock()
                .unwrap()
                .pop_front()
                .expect("mock response");
            tx.send(chimera_sigil_providers::StreamEvent::Done(response))
                .unwrap();
            Ok(())
        }

        async fn chat(&self, _request: ChatRequest) -> anyhow::Result<ChatResponse> {
            anyhow::bail!("chat() not used in agent tests")
        }
    }

    #[derive(Clone)]
    struct FlakyProvider {
        failures_remaining: Arc<Mutex<usize>>,
        requests: Arc<Mutex<Vec<ChatRequest>>>,
        response: ChatResponse,
    }

    impl FlakyProvider {
        fn new(failures: usize, response: ChatResponse) -> Self {
            Self {
                failures_remaining: Arc::new(Mutex::new(failures)),
                requests: Arc::new(Mutex::new(Vec::new())),
                response,
            }
        }

        fn requests(&self) -> Vec<ChatRequest> {
            self.requests.lock().unwrap().clone()
        }
    }

    #[async_trait]
    impl Provider for FlakyProvider {
        fn kind(&self) -> ProviderKind {
            ProviderKind::Ollama
        }

        async fn chat_stream(
            &self,
            request: ChatRequest,
            tx: mpsc::UnboundedSender<chimera_sigil_providers::StreamEvent>,
        ) -> anyhow::Result<()> {
            self.requests.lock().unwrap().push(request);
            let mut failures = self.failures_remaining.lock().unwrap();
            if *failures > 0 {
                *failures -= 1;
                anyhow::bail!("temporary provider outage");
            }
            drop(failures);

            tx.send(chimera_sigil_providers::StreamEvent::Done(
                self.response.clone(),
            ))
            .unwrap();
            Ok(())
        }

        async fn chat(&self, _request: ChatRequest) -> anyhow::Result<ChatResponse> {
            anyhow::bail!("chat() not used in agent tests")
        }
    }

    fn tool_call(id: &str, name: &str, arguments: &str) -> ToolCall {
        ToolCall {
            id: id.to_string(),
            call_type: "function".into(),
            function: FunctionCall {
                name: name.to_string(),
                arguments: arguments.to_string(),
            },
        }
    }

    #[test]
    fn structured_reports_block_later_textual_tool_calls() {
        let content = "{\n  \"task\": \"repo-scout-chimera\",\n  \"target\": \"../chimera\",\n  \"summary\": \"done\"\n}\n{\"name\":\"glob_search\",\"arguments\":{\"pattern\":\"*.rs\"}}";

        assert!(contains_structured_report_json(content));
        assert!(extract_textual_tool_calls(content).is_none());
    }

    #[test]
    fn tool_call_arguments_do_not_look_like_structured_reports() {
        let (_, calls) = extract_textual_tool_calls(
            r#"{"name":"structured_output","arguments":{"task":"repo-scout","target":"../chimera"}}"#,
        )
        .expect("textual tool call");

        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "structured_output");
        assert_eq!(
            serde_json::from_str::<Value>(&calls[0].function.arguments).unwrap(),
            serde_json::json!({"task": "repo-scout", "target": "../chimera"})
        );
    }

    #[test]
    fn parses_pretty_printed_textual_tool_call_objects() {
        let (_, calls) = extract_textual_tool_calls(
            "{\n  \"name\": \"grep_search\",\n  \"arguments\": {\n    \"include\": \"**/*.rs\",\n    \"pattern\": \"provider\"\n  }\n}",
        )
        .expect("textual tool call");

        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "grep_search");
        assert_eq!(
            serde_json::from_str::<Value>(&calls[0].function.arguments).unwrap(),
            serde_json::json!({"include": "**/*.rs", "pattern": "provider"})
        );
    }

    #[test]
    fn parses_multiple_pretty_printed_textual_tool_call_objects() {
        let (_, calls) = extract_textual_tool_calls(
            "{\n  \"name\": \"glob_search\",\n  \"arguments\": {\"pattern\": \"**/*provider*\"}\n}\n\n{\n  \"name\": \"glob_search\",\n  \"arguments\": {\"pattern\": \"**/*session*\"}\n}",
        )
        .expect("textual tool calls");

        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].id, "text_tool_call_0");
        assert_eq!(calls[0].function.name, "glob_search");
        assert_eq!(calls[1].id, "text_tool_call_1");
        assert_eq!(calls[1].function.name, "glob_search");
    }

    #[test]
    fn parses_name_arguments_textual_tool_call_lines() {
        let (_, calls) = extract_textual_tool_calls(
            "name: read_file, arguments: {\"path\": \"../chimera/README.md\"}\n\
             name: glob_search, arguments: {\"pattern\": \"**/*.rs\", \"path\": \"../chimera\"}",
        )
        .expect("textual tool calls");

        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].function.name, "read_file");
        assert_eq!(
            serde_json::from_str::<Value>(&calls[0].function.arguments).unwrap(),
            serde_json::json!({"path": "../chimera/README.md"})
        );
        assert_eq!(calls[1].function.name, "glob_search");
    }

    #[tokio::test]
    async fn approve_mode_denies_execute_tools_without_prompting() {
        let provider = MockProvider::new(vec![
            ChatResponse {
                content: None,
                tool_calls: vec![tool_call("call_exec_1", "bash", r#"{"command":"echo hi"}"#)],
                usage: None,
                finish_reason: Some("tool_calls".into()),
            },
            ChatResponse {
                content: Some("done".into()),
                tool_calls: Vec::new(),
                usage: None,
                finish_reason: Some("stop".into()),
            },
        ]);

        let mut agent = Agent::new(
            Box::new(provider.clone()),
            "gpt-4o".into(),
            Config {
                approval_mode: ApprovalMode::Approve,
                ..Config::default()
            },
        );

        let events = Arc::new(Mutex::new(Vec::new()));
        let events_ref = events.clone();
        let callback: EventCallback = Box::new(move |event| {
            events_ref.lock().unwrap().push(event);
        });

        let outcome = agent.run_turn("run it", &callback).await.unwrap();
        assert_eq!(outcome.exit_reason, ExitReason::Complete);

        let recorded = events.lock().unwrap().clone();
        assert!(recorded.iter().any(|event| matches!(
            event,
            AgentEvent::ToolDenied {
                tool_call_id,
                name,
                reason
            } if tool_call_id == "call_exec_1"
                && name == "bash"
                && reason.contains("approval mode 'approve'")
        )));

        let requests = provider.requests();
        assert_eq!(requests.len(), 2);
        let last_message = requests[1].messages.last().unwrap();
        assert_eq!(last_message.role, Role::Tool);
        assert_eq!(
            last_message.content.as_deref(),
            Some("Tool execution denied by approval mode 'approve'.")
        );
    }

    #[tokio::test]
    async fn emits_tool_call_ids_and_session_metadata() {
        let provider = MockProvider::new(vec![
            ChatResponse {
                content: None,
                tool_calls: vec![tool_call(
                    "call_json_1",
                    "structured_output",
                    r#"{"status":"ok"}"#,
                )],
                usage: None,
                finish_reason: Some("tool_calls".into()),
            },
            ChatResponse {
                content: Some("done".into()),
                tool_calls: Vec::new(),
                usage: None,
                finish_reason: Some("stop".into()),
            },
        ]);

        let mut agent = Agent::new(Box::new(provider), "gpt-4o".into(), Config::default());
        let session_id = agent.session().id.clone();

        let events = Arc::new(Mutex::new(Vec::new()));
        let events_ref = events.clone();
        let callback: EventCallback = Box::new(move |event| {
            events_ref.lock().unwrap().push(event);
        });

        agent.run_turn("return json", &callback).await.unwrap();

        let recorded = events.lock().unwrap().clone();
        assert!(recorded.iter().any(|event| matches!(
            event,
            AgentEvent::ToolStart {
                tool_call_id,
                name,
                ..
            } if tool_call_id == "call_json_1" && name == "structured_output"
        )));
        assert!(recorded.iter().any(|event| matches!(
            event,
            AgentEvent::ToolResult {
                tool_call_id,
                name,
                is_error,
                ..
            } if tool_call_id == "call_json_1" && name == "structured_output" && !is_error
        )));
        assert!(recorded.iter().any(|event| matches!(
            event,
            AgentEvent::TurnComplete {
                session_id: complete_session_id,
                ..
            } if complete_session_id == &session_id
        )));
    }

    #[tokio::test]
    async fn executes_textual_json_tool_calls_from_local_models() {
        let provider = MockProvider::new(vec![
            ChatResponse {
                content: Some(r#"{"name":"glob_search","arguments":{"pattern":"*.md"}}"#.into()),
                tool_calls: Vec::new(),
                usage: None,
                finish_reason: Some("stop".into()),
            },
            ChatResponse {
                content: Some("scout complete".into()),
                tool_calls: Vec::new(),
                usage: None,
                finish_reason: Some("stop".into()),
            },
        ]);

        let mut agent = Agent::new(
            Box::new(provider.clone()),
            "qwen2.5-coder:32b".into(),
            Config::default(),
        );

        let events = Arc::new(Mutex::new(Vec::new()));
        let events_ref = events.clone();
        let callback: EventCallback = Box::new(move |event| {
            events_ref.lock().unwrap().push(event);
        });

        let outcome = agent.run_turn("inspect markdown", &callback).await.unwrap();

        assert_eq!(outcome.text.as_deref(), Some("scout complete"));
        assert_eq!(provider.requests().len(), 2);

        let recorded = events.lock().unwrap().clone();
        assert!(recorded.iter().any(|event| matches!(
            event,
            AgentEvent::ToolStart { name, .. } if name == "glob_search"
        )));
        assert!(recorded.iter().any(|event| matches!(
            event,
            AgentEvent::ToolResult {
                name,
                is_error,
                ..
            } if name == "glob_search" && !is_error
        )));
    }

    #[tokio::test]
    async fn executes_textual_json_tool_calls_mixed_with_planning_text() {
        let provider = MockProvider::new(vec![
            ChatResponse {
                content: Some(
                    "I'll inspect the docs first.\n\n\
                     {\"name\":\"grep_search\",\"arguments\":{\"pattern\":\"Research Delegation\"}}"
                        .into(),
                ),
                tool_calls: Vec::new(),
                usage: None,
                finish_reason: Some("stop".into()),
            },
            ChatResponse {
                content: Some("mixed scout complete".into()),
                tool_calls: Vec::new(),
                usage: None,
                finish_reason: Some("stop".into()),
            },
        ]);

        let mut agent = Agent::new(
            Box::new(provider.clone()),
            "qwen2.5-coder:32b".into(),
            Config::default(),
        );

        let callback: EventCallback = Box::new(|_| {});
        let outcome = agent.run_turn("inspect docs", &callback).await.unwrap();

        assert_eq!(outcome.text.as_deref(), Some("mixed scout complete"));
        assert_eq!(provider.requests().len(), 2);
        let requests = provider.requests();
        let assistant_message = requests[1]
            .messages
            .iter()
            .find(|message| message.role == Role::Assistant && message.tool_calls.is_some())
            .unwrap();
        assert_eq!(
            assistant_message.content.as_deref(),
            Some("I'll inspect the docs first.")
        );
    }

    #[tokio::test]
    async fn does_not_execute_textual_tool_calls_after_structured_report() {
        let report = "{\n  \"task\": \"repo-scout-chimera\",\n  \"target\": \"../chimera\",\n  \"summary\": \"Repository scout complete\"\n}\n{\"name\":\"glob_search\",\"arguments\":{\"pattern\":\"*.rs\"}}";
        let provider = MockProvider::new(vec![ChatResponse {
            content: Some(report.into()),
            tool_calls: Vec::new(),
            usage: None,
            finish_reason: Some("stop".into()),
        }]);

        let mut agent = Agent::new(
            Box::new(provider.clone()),
            "qwen2.5-coder:32b".into(),
            Config::default(),
        );

        let events = Arc::new(Mutex::new(Vec::new()));
        let events_ref = events.clone();
        let callback: EventCallback = Box::new(move |event| {
            events_ref.lock().unwrap().push(event);
        });

        let outcome = agent
            .run_turn("inspect repository", &callback)
            .await
            .unwrap();

        assert_eq!(outcome.exit_reason, ExitReason::Complete);
        assert!(
            outcome
                .text
                .as_deref()
                .is_some_and(|text| text.contains("\"task\": \"repo-scout-chimera\""))
        );
        assert_eq!(provider.requests().len(), 1);

        let recorded = events.lock().unwrap().clone();
        assert!(
            !recorded
                .iter()
                .any(|event| matches!(event, AgentEvent::ToolStart { .. }))
        );
    }

    #[tokio::test]
    async fn retries_provider_failures_before_failing_the_turn() {
        let provider = FlakyProvider::new(
            1,
            ChatResponse {
                content: Some("ok".into()),
                tool_calls: Vec::new(),
                usage: None,
                finish_reason: Some("stop".into()),
            },
        );

        let mut agent = Agent::new(
            Box::new(provider.clone()),
            "qwen3:4b".into(),
            Config {
                provider_retries: 1,
                provider_retry_backoff_ms: 0,
                ..Config::default()
            },
        );

        let events = Arc::new(Mutex::new(Vec::new()));
        let events_ref = events.clone();
        let callback: EventCallback = Box::new(move |event| {
            events_ref.lock().unwrap().push(event);
        });

        let outcome = agent.run_turn("hello", &callback).await.unwrap();
        assert_eq!(outcome.exit_reason, ExitReason::Complete);
        assert_eq!(provider.requests().len(), 2);

        let recorded = events.lock().unwrap().clone();
        assert!(recorded.iter().any(|event| matches!(
            event,
            AgentEvent::ProviderRetry {
                attempt: 1,
                max_attempts: 2,
                error,
                ..
            } if error.contains("temporary provider outage")
        )));
    }
}
