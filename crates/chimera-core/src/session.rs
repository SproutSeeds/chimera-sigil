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

#[cfg(test)]
mod tests {
    use super::*;
    use chimera_providers::types::{FunctionCall, ToolCall};

    #[test]
    fn test_session_new() {
        let session = Session::new();
        assert!(!session.id.is_empty());
        assert!(session.messages.is_empty());
        assert_eq!(session.total_tokens(), 0);
    }

    #[test]
    fn test_system_prompt_is_first() {
        let mut session = Session::new();
        session.push_user("hello");
        session.set_system_prompt("system");

        assert_eq!(session.messages[0].role, Role::System);
        assert_eq!(session.messages[0].content.as_deref(), Some("system"));
        assert_eq!(session.messages[1].role, Role::User);
    }

    #[test]
    fn test_system_prompt_replaces_existing() {
        let mut session = Session::new();
        session.set_system_prompt("first");
        session.set_system_prompt("second");

        let system_msgs: Vec<_> = session
            .messages
            .iter()
            .filter(|m| m.role == Role::System)
            .collect();
        assert_eq!(system_msgs.len(), 1);
        assert_eq!(system_msgs[0].content.as_deref(), Some("second"));
    }

    #[test]
    fn test_conversation_flow() {
        let mut session = Session::new();
        session.set_system_prompt("system");
        session.push_user("what files?");
        session.push_assistant_tool_calls(
            None,
            vec![ToolCall {
                id: "call_1".into(),
                call_type: "function".into(),
                function: FunctionCall {
                    name: "bash".into(),
                    arguments: r#"{"command":"ls"}"#.into(),
                },
            }],
        );
        session.push_tool_result("call_1", "file1.txt\nfile2.txt");
        session.push_assistant_text("I see two files.");

        assert_eq!(session.messages.len(), 5);
        assert_eq!(session.messages[0].role, Role::System);
        assert_eq!(session.messages[1].role, Role::User);
        assert_eq!(session.messages[2].role, Role::Assistant);
        assert!(session.messages[2].tool_calls.is_some());
        assert_eq!(session.messages[3].role, Role::Tool);
        assert_eq!(session.messages[3].tool_call_id.as_deref(), Some("call_1"));
        assert_eq!(session.messages[4].role, Role::Assistant);
    }

    #[test]
    fn test_usage_tracking() {
        let mut session = Session::new();
        session.record_usage(100, 50);
        session.record_usage(200, 75);

        assert_eq!(session.total_input_tokens, 300);
        assert_eq!(session.total_output_tokens, 125);
        assert_eq!(session.total_tokens(), 425);
    }
}
