use chimera_sigil_providers::types::{Message, Role};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use uuid::Uuid;

const APP_DIR_NAME: &str = ".chimera-sigil";
const LEGACY_APP_DIR_NAMES: &[&str] = &[".chimera-harness", ".chimera"];

/// Manages the conversation history for a session.
#[derive(Debug, Serialize, Deserialize)]
pub struct Session {
    pub id: String,
    pub messages: Vec<Message>,
    pub total_input_tokens: u32,
    pub total_output_tokens: u32,
}

/// Metadata header written as the first line of a session file.
#[derive(Serialize, Deserialize)]
struct SessionHeader {
    id: String,
    total_input_tokens: u32,
    total_output_tokens: u32,
    message_count: usize,
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

    fn app_dir(dir_name: &str) -> PathBuf {
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
        PathBuf::from(home).join(dir_name)
    }

    /// Primary directory for session files: ~/.chimera-sigil/sessions/
    pub fn sessions_dir() -> PathBuf {
        Self::app_dir(APP_DIR_NAME).join("sessions")
    }

    /// Legacy session directories kept for backward-compatible loading.
    fn legacy_sessions_dirs() -> Vec<PathBuf> {
        LEGACY_APP_DIR_NAMES
            .iter()
            .map(|dir_name| Self::app_dir(dir_name).join("sessions"))
            .collect()
    }

    fn session_lookup_dirs() -> Vec<PathBuf> {
        let mut dirs = vec![Self::sessions_dir()];
        for legacy in Self::legacy_sessions_dirs() {
            if !dirs.contains(&legacy) {
                dirs.push(legacy);
            }
        }
        dirs
    }

    fn format_lookup_dirs() -> String {
        let dirs: Vec<_> = Self::session_lookup_dirs()
            .into_iter()
            .map(|dir| dir.display().to_string())
            .collect();
        if dirs.is_empty() {
            ".".into()
        } else {
            dirs.join(", ")
        }
    }

    /// Save the session to a JSONL file. First line is metadata, rest are messages.
    pub fn save(&self) -> anyhow::Result<PathBuf> {
        let dir = Self::sessions_dir();
        std::fs::create_dir_all(&dir)?;

        let path = dir.join(format!("{}.jsonl", self.id));
        self.save_to(&path)?;
        Ok(path)
    }

    /// Save to a specific path.
    pub fn save_to(&self, path: &Path) -> anyhow::Result<()> {
        let mut lines = Vec::with_capacity(self.messages.len() + 1);

        let header = SessionHeader {
            id: self.id.clone(),
            total_input_tokens: self.total_input_tokens,
            total_output_tokens: self.total_output_tokens,
            message_count: self.messages.len(),
        };
        lines.push(serde_json::to_string(&header)?);

        for msg in &self.messages {
            lines.push(serde_json::to_string(msg)?);
        }

        std::fs::write(path, lines.join("\n") + "\n")?;
        Ok(())
    }

    /// Load a session from a JSONL file.
    pub fn load_from(path: &Path) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let mut lines = content.lines();

        let header_line = lines
            .next()
            .ok_or_else(|| anyhow::anyhow!("Empty session file"))?;
        let header: SessionHeader = serde_json::from_str(header_line)?;

        let mut messages = Vec::with_capacity(header.message_count);
        for line in lines {
            if line.trim().is_empty() {
                continue;
            }
            let msg: Message = serde_json::from_str(line)?;
            messages.push(msg);
        }

        Ok(Self {
            id: header.id,
            messages,
            total_input_tokens: header.total_input_tokens,
            total_output_tokens: header.total_output_tokens,
        })
    }

    /// Load a session by ID from the default sessions directory.
    pub fn load(id: &str) -> anyhow::Result<Self> {
        for dir in Self::session_lookup_dirs() {
            let path = dir.join(format!("{id}.jsonl"));
            if path.exists() {
                return Self::load_from(&path);
            }
        }

        anyhow::bail!(
            "Session file not found in any known session directory: {}",
            Self::format_lookup_dirs()
        );
    }

    /// List all saved session IDs from both the current and legacy session
    /// directories (sorted by modification time, newest first).
    pub fn list_saved() -> anyhow::Result<Vec<(String, std::time::SystemTime)>> {
        let mut sessions_by_id = HashMap::new();

        for dir in Self::session_lookup_dirs() {
            if !dir.exists() {
                continue;
            }

            for entry in std::fs::read_dir(&dir)? {
                let entry = entry?;
                let path = entry.path();
                if path.extension().is_some_and(|e| e == "jsonl")
                    && let Some(stem) = path.file_stem().and_then(|s| s.to_str())
                {
                    let modified = entry.metadata()?.modified()?;
                    sessions_by_id
                        .entry(stem.to_string())
                        .and_modify(|existing| {
                            if modified > *existing {
                                *existing = modified;
                            }
                        })
                        .or_insert(modified);
                }
            }
        }

        let mut sessions: Vec<_> = sessions_by_id.into_iter().collect();
        sessions.sort_by(|a, b| b.1.cmp(&a.1));
        Ok(sessions)
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
        tool_calls: Vec<chimera_sigil_providers::types::ToolCall>,
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

    /// Estimate current context size in tokens (rough: chars / 4).
    pub fn estimate_context_tokens(&self) -> usize {
        self.messages
            .iter()
            .map(|m| {
                let content_len = m.content.as_deref().map_or(0, |c| c.len());
                let tool_len = m.tool_calls.as_ref().map_or(0, |tcs| {
                    tcs.iter()
                        .map(|tc| tc.function.name.len() + tc.function.arguments.len())
                        .sum()
                });
                (content_len + tool_len) / 4 + 4 // +4 for message overhead
            })
            .sum()
    }

    /// Compact the conversation by dropping the oldest non-system messages,
    /// keeping the system prompt, the first user message (for context),
    /// and the most recent `keep_recent` messages.
    ///
    /// Returns the number of messages removed.
    pub fn compact(&mut self, keep_recent: usize) -> usize {
        let total = self.messages.len();
        if total <= keep_recent + 2 {
            return 0; // Nothing to compact
        }

        // Find boundaries: system prompt (index 0 typically), then keep
        // the first user message and the last `keep_recent` messages.
        let system_end = self
            .messages
            .iter()
            .position(|m| m.role != Role::System)
            .unwrap_or(0);

        // Keep: [0..system_end] + first user msg + last `keep_recent`
        let first_user_idx = self
            .messages
            .iter()
            .position(|m| m.role == Role::User)
            .unwrap_or(system_end);

        let keep_from = self.adjust_keep_from_for_tool_context(total.saturating_sub(keep_recent));

        // If the keep_from is already before the first user, nothing to drop
        if keep_from <= first_user_idx + 1 {
            return 0;
        }

        // Build new message list
        let mut new_messages = Vec::new();

        // System messages
        for msg in &self.messages[..=system_end.max(0)] {
            if msg.role == Role::System {
                new_messages.push(msg.clone());
            }
        }

        // First user message (if not already in the recent window)
        if first_user_idx < keep_from {
            new_messages.push(self.messages[first_user_idx].clone());

            // Add a compaction marker so the model knows context was dropped
            new_messages.push(Message {
                role: Role::User,
                content: Some(
                    "[Earlier conversation messages were removed to stay within context limits. \
                     The conversation continues below.]"
                        .into(),
                ),
                tool_calls: None,
                tool_call_id: None,
            });
        }

        // Recent messages
        for msg in &self.messages[keep_from..] {
            new_messages.push(msg.clone());
        }

        let removed = total - new_messages.len();
        self.messages = new_messages;
        removed
    }

    /// Move the compaction boundary back to the assistant tool-call message if
    /// it would otherwise start in the middle of a tool exchange.
    fn adjust_keep_from_for_tool_context(&self, keep_from: usize) -> usize {
        let mut adjusted = keep_from.min(self.messages.len());

        if adjusted < self.messages.len() && self.messages[adjusted].role == Role::Tool {
            if let Some(tool_call_idx) = (0..adjusted).rev().find(|&idx| {
                let msg = &self.messages[idx];
                msg.role == Role::Assistant
                    && msg
                        .tool_calls
                        .as_ref()
                        .is_some_and(|tool_calls| !tool_calls.is_empty())
            }) {
                adjusted = tool_call_idx;
            }
        }

        adjusted
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
    use chimera_sigil_providers::types::{FunctionCall, ToolCall};

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

    #[test]
    fn test_save_and_load_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test_session.jsonl");

        let mut session = Session::new();
        session.set_system_prompt("system prompt");
        session.push_user("hello");
        session.push_assistant_text("hi there");
        session.record_usage(100, 50);

        session.save_to(&path).unwrap();

        let loaded = Session::load_from(&path).unwrap();
        assert_eq!(loaded.id, session.id);
        assert_eq!(loaded.messages.len(), 3);
        assert_eq!(loaded.messages[0].role, Role::System);
        assert_eq!(loaded.messages[1].role, Role::User);
        assert_eq!(loaded.messages[2].role, Role::Assistant);
        assert_eq!(loaded.total_input_tokens, 100);
        assert_eq!(loaded.total_output_tokens, 50);
    }

    #[test]
    fn test_save_and_load_with_tool_calls() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tool_session.jsonl");

        let mut session = Session::new();
        session.push_user("list files");
        session.push_assistant_tool_calls(
            Some("I'll check.".into()),
            vec![ToolCall {
                id: "call_1".into(),
                call_type: "function".into(),
                function: FunctionCall {
                    name: "bash".into(),
                    arguments: r#"{"command":"ls"}"#.into(),
                },
            }],
        );
        session.push_tool_result("call_1", "file1.txt");

        session.save_to(&path).unwrap();

        let loaded = Session::load_from(&path).unwrap();
        assert_eq!(loaded.messages.len(), 3);
        assert!(loaded.messages[1].tool_calls.is_some());
        let tc = &loaded.messages[1].tool_calls.as_ref().unwrap()[0];
        assert_eq!(tc.function.name, "bash");
        assert_eq!(loaded.messages[2].tool_call_id.as_deref(), Some("call_1"));
    }

    #[test]
    fn test_list_saved_sessions() {
        let dir = tempfile::tempdir().unwrap();
        // Override sessions dir by saving directly
        let path1 = dir.path().join("session1.jsonl");
        let path2 = dir.path().join("session2.jsonl");

        let mut s1 = Session::new();
        s1.push_user("hello");
        s1.save_to(&path1).unwrap();

        let mut s2 = Session::new();
        s2.push_user("world");
        s2.save_to(&path2).unwrap();

        // Both files should exist
        assert!(path1.exists());
        assert!(path2.exists());
    }

    #[test]
    fn test_estimate_context_tokens() {
        let mut session = Session::new();
        session.set_system_prompt("You are helpful.");
        session.push_user("Hello world");
        session.push_assistant_text("Hi there!");

        let tokens = session.estimate_context_tokens();
        // Rough estimate: 3 messages, each with some text + overhead
        assert!(tokens > 0);
        assert!(tokens < 100); // Small conversation
    }

    #[test]
    fn test_compact_removes_middle_messages() {
        let mut session = Session::new();
        session.set_system_prompt("system");
        session.push_user("first question");
        session.push_assistant_text("answer 1");
        session.push_user("second question");
        session.push_assistant_text("answer 2");
        session.push_user("third question");
        session.push_assistant_text("answer 3");
        session.push_user("fourth question");
        session.push_assistant_text("answer 4");
        session.push_user("fifth question");
        session.push_assistant_text("answer 5");

        assert_eq!(session.messages.len(), 11); // 1 system + 10

        let removed = session.compact(4);
        assert!(removed > 0);

        // Should keep: system, first user, compaction marker, last 4
        assert!(session.messages.len() <= 7);
        assert_eq!(session.messages[0].role, Role::System);
        assert_eq!(session.messages[1].role, Role::User);
        assert_eq!(
            session.messages[1].content.as_deref(),
            Some("first question")
        );
        // Compaction marker
        assert!(
            session.messages[2]
                .content
                .as_deref()
                .unwrap()
                .contains("removed")
        );
    }

    #[test]
    fn test_compact_noop_when_small() {
        let mut session = Session::new();
        session.set_system_prompt("system");
        session.push_user("hello");
        session.push_assistant_text("hi");

        let removed = session.compact(10);
        assert_eq!(removed, 0);
        assert_eq!(session.messages.len(), 3);
    }

    #[test]
    fn test_compact_keeps_tool_exchange_intact() {
        let mut session = Session::new();
        session.set_system_prompt("system");
        session.push_user("opening question");
        session.push_assistant_text("opening answer");
        session.push_user("first question");
        session.push_assistant_tool_calls(
            Some("let me check".into()),
            vec![ToolCall {
                id: "call_1".into(),
                call_type: "function".into(),
                function: FunctionCall {
                    name: "bash".into(),
                    arguments: r#"{"command":"ls"}"#.into(),
                },
            }],
        );
        session.push_tool_result("call_1", "file1.txt");
        session.push_assistant_text("done");
        session.push_user("second question");
        session.push_assistant_text("answer 2");

        let removed = session.compact(4);
        assert!(removed > 0);

        let tool_call_idx = session
            .messages
            .iter()
            .position(|msg| msg.tool_calls.is_some())
            .unwrap();
        let tool_result_idx = session
            .messages
            .iter()
            .position(|msg| msg.role == Role::Tool)
            .unwrap();

        assert!(tool_call_idx < tool_result_idx);
        assert_eq!(
            session.messages[tool_result_idx].tool_call_id.as_deref(),
            Some("call_1")
        );
    }
}
