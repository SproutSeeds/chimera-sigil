/// Agent configuration.
#[derive(Debug, Clone)]
pub struct Config {
    /// Model to use (e.g., "grok-3", "gpt-4o", "llama3.2:latest").
    pub model: String,
    /// Maximum tool-calling iterations per turn before forcing a response.
    pub max_iterations: usize,
    /// Temperature for model responses.
    pub temperature: Option<f32>,
    /// Maximum tokens for model responses.
    pub max_tokens: Option<u32>,
    /// System prompt override. If None, uses the default.
    pub system_prompt: Option<String>,
    /// Whether to auto-approve tool executions (dangerous mode).
    pub auto_approve: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            model: "grok-3".into(),
            max_iterations: 25,
            temperature: None,
            max_tokens: None,
            system_prompt: None,
            auto_approve: false,
        }
    }
}

impl Config {
    pub fn system_prompt(&self) -> String {
        if let Some(custom) = &self.system_prompt {
            return custom.clone();
        }

        let cwd = std::env::current_dir()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|_| ".".into());

        let date = chrono::Local::now().format("%Y-%m-%d").to_string();

        format!(
            r#"You are Chimera, a multi-model AI agent orchestrator running in the user's terminal.
You help with software engineering tasks by reading, writing, and editing code, running commands, and searching codebases.

# Environment
- Working directory: {cwd}
- Date: {date}
- Platform: {os}
- Model: {model}

# Tools
You have access to tools for interacting with the filesystem and running commands.
Use them to accomplish the user's requests. Prefer using tools over asking the user to do things manually.

# Guidelines
- Read files before editing them.
- Use glob_search and grep_search to find files and code patterns.
- Use bash for running commands, git operations, builds, and tests.
- Use edit_file for targeted changes (exact string replacement).
- Use write_file only for new files or complete rewrites.
- Keep responses concise and action-oriented.
- If a task is ambiguous, ask clarifying questions."#,
            os = std::env::consts::OS,
            model = self.model,
        )
    }
}
