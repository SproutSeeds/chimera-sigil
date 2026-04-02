use chimera_providers::types::{Message, Role};
use uuid::Uuid;

/// Manages the conversation history for a session.
#[derive(Debug)]
pub struct Session {
    pub id: String,
    pub messages: Vec<Message>,
    pub total_input_tokens: u32,
    pub total_output_tokens: u32,
}

impl Session {
    pub fn new() -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            messages: Vec::new(),
            total_input_tokens: 0,
            total_output_tokens: 0,
        }
    }

    /// Add the system prompt as the first message.
    pub fn set_system_prompt(&mut self, prompt: &str) {
        // Remove existing system prompt if any
        self.messages.retain(|m| m.role != Role::System);

        self.messages.insert(
            0,
            Message {
                role: Role::System,
                content: Some(prompt.to_string()),
                tool_calls: None,
                tool_call_id: None,
            },
        );
    }

    /// Add a user message.
    pub fn push_user(&mut self, content: &str) {
        self.messages.push(Message {
            role: Role::User,
            content: Some(content.to_string()),
            tool_calls: None,
            tool_call_id: None,
        });
    }

    /// Add an assistant message (text response).
    pub fn push_assistant_text(&mut self, content: &str) {
        self.messages.push(Message {
            role: Role::Assistant,
            content: Some(content.to_string()),
            tool_calls: None,
            tool_call_id: None,
        });
    }

    /// Add an assistant message with tool calls.
    pub fn push_assistant_tool_calls(
        &mut self,
        content: Option<String>,
        tool_calls: Vec<chimera_providers::types::ToolCall>,
    ) {
        self.messages.push(Message {
            role: Role::Assistant,
            content,
            tool_calls: Some(tool_calls),
            tool_call_id: None,
        });
    }

    /// Add a tool result message.
    pub fn push_tool_result(&mut self, tool_call_id: &str, result: &str) {
        self.messages.push(Message {
            role: Role::Tool,
            content: Some(result.to_string()),
            tool_calls: None,
            tool_call_id: Some(tool_call_id.to_string()),
        });
    }

    /// Track token usage.
    pub fn record_usage(&mut self, input: u32, output: u32) {
        self.total_input_tokens += input;
        self.total_output_tokens += output;
    }

    /// Get the total token count.
    pub fn total_tokens(&self) -> u32 {
        self.total_input_tokens + self.total_output_tokens
    }
}

impl Default for Session {
    fn default() -> Self {
        Self::new()
    }
}
