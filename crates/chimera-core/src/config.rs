use std::collections::HashSet;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};

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
    /// Build the system prompt as sections. Includes discovered instruction
    /// files from the directory tree (CLAUDE.md, AGENTS.md, CHIMERA.md).
    ///
    /// Inspired by claw-code's recursive instruction discovery pattern:
    /// walk from cwd to filesystem root, collect instruction files,
    /// deduplicate by content hash, enforce per-file budget.
    pub fn system_prompt(&self) -> String {
        if let Some(custom) = &self.system_prompt {
            return custom.clone();
        }

        let cwd = std::env::current_dir()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|_| ".".into());

        let date = chrono::Local::now().format("%Y-%m-%d").to_string();

        let mut sections = vec![format!(
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
- Use structured_output to return data in a specific JSON format when requested.
- Keep responses concise and action-oriented.
- If a task is ambiguous, ask clarifying questions."#,
            os = std::env::consts::OS,
            model = self.model,
        )];

        // Discover and append project instruction files
        let instructions = discover_instruction_files(Path::new(&cwd));
        if !instructions.is_empty() {
            sections.push("\n# Project Instructions\nThe following instruction files were found in the project hierarchy:\n".into());
            for file in &instructions {
                sections.push(format!(
                    "## {}\n{}\n",
                    file.path.display(),
                    file.content
                ));
            }
        }

        sections.join("\n")
    }
}

/// An instruction file discovered in the directory tree.
#[derive(Debug)]
struct InstructionFile {
    path: PathBuf,
    content: String,
}

/// Maximum characters per instruction file before truncation.
const MAX_INSTRUCTION_FILE_CHARS: usize = 4000;

/// Total budget for all instruction files combined.
const MAX_TOTAL_INSTRUCTION_CHARS: usize = 12000;

/// File names to look for at each directory level.
const INSTRUCTION_FILE_NAMES: &[&str] = &[
    "CLAUDE.md",
    "AGENTS.md",
    "CHIMERA.md",
    "CLAUDE.local.md",
];

/// Walk from `cwd` up to the filesystem root, collecting instruction files.
/// Deduplicates by content hash (same content at different paths = included once).
/// Enforces per-file and total character budgets.
fn discover_instruction_files(cwd: &Path) -> Vec<InstructionFile> {
    let mut files = Vec::new();
    let mut seen_hashes = HashSet::new();
    let mut total_chars = 0usize;

    // Collect ancestor directories (cwd first, then parents)
    let mut directories = Vec::new();
    let mut cursor = Some(cwd.to_path_buf());
    while let Some(dir) = cursor {
        directories.push(dir.clone());
        cursor = dir.parent().map(|p| p.to_path_buf());
    }

    for dir in &directories {
        for name in INSTRUCTION_FILE_NAMES {
            let path = dir.join(name);
            if !path.is_file() {
                continue;
            }

            let content = match std::fs::read_to_string(&path) {
                Ok(c) if !c.trim().is_empty() => c,
                _ => continue,
            };

            // Deduplicate by content hash
            let hash = content_hash(&content);
            if !seen_hashes.insert(hash) {
                continue;
            }

            // Enforce per-file budget
            let content = if content.len() > MAX_INSTRUCTION_FILE_CHARS {
                let mut truncated = content[..MAX_INSTRUCTION_FILE_CHARS].to_string();
                truncated.push_str("\n... (truncated)");
                truncated
            } else {
                content
            };

            // Enforce total budget
            if total_chars + content.len() > MAX_TOTAL_INSTRUCTION_CHARS {
                break;
            }
            total_chars += content.len();

            files.push(InstructionFile { path, content });
        }

        if total_chars >= MAX_TOTAL_INSTRUCTION_CHARS {
            break;
        }
    }

    files
}

/// Stable content hash for deduplication.
fn content_hash(content: &str) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    content.hash(&mut hasher);
    hasher.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_system_prompt_contains_key_sections() {
        let config = Config::default();
        let prompt = config.system_prompt();
        assert!(prompt.contains("Chimera"));
        assert!(prompt.contains("# Environment"));
        assert!(prompt.contains("# Tools"));
        assert!(prompt.contains("# Guidelines"));
        assert!(prompt.contains("grok-3"));
    }

    #[test]
    fn test_custom_system_prompt_overrides() {
        let config = Config {
            system_prompt: Some("Custom prompt".into()),
            ..Config::default()
        };
        assert_eq!(config.system_prompt(), "Custom prompt");
    }

    #[test]
    fn test_discover_instruction_files_from_temp_dir() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("CLAUDE.md"), "# Project\nDo things right.\n").unwrap();
        std::fs::write(dir.path().join("AGENTS.md"), "# Agents\nBe helpful.\n").unwrap();

        let files = discover_instruction_files(dir.path());
        assert_eq!(files.len(), 2);
        assert!(files[0].content.contains("Project"));
        assert!(files[1].content.contains("Agents"));
    }

    #[test]
    fn test_discover_deduplicates_by_content() {
        let parent = tempfile::tempdir().unwrap();
        let child = parent.path().join("child");
        std::fs::create_dir(&child).unwrap();

        // Same content at both levels
        std::fs::write(parent.path().join("CLAUDE.md"), "identical\n").unwrap();
        std::fs::write(child.join("CLAUDE.md"), "identical\n").unwrap();

        let files = discover_instruction_files(&child);
        // Should only include one copy despite being at two paths
        assert_eq!(files.len(), 1);
    }

    #[test]
    fn test_discover_truncates_large_files() {
        let dir = tempfile::tempdir().unwrap();
        let big_content = "x".repeat(MAX_INSTRUCTION_FILE_CHARS + 1000);
        std::fs::write(dir.path().join("CLAUDE.md"), &big_content).unwrap();

        let files = discover_instruction_files(dir.path());
        assert_eq!(files.len(), 1);
        assert!(files[0].content.len() <= MAX_INSTRUCTION_FILE_CHARS + 20); // +truncation msg
        assert!(files[0].content.ends_with("... (truncated)"));
    }

    #[test]
    fn test_discover_skips_empty_files() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("CLAUDE.md"), "").unwrap();
        std::fs::write(dir.path().join("AGENTS.md"), "   \n  \n").unwrap();

        let files = discover_instruction_files(dir.path());
        assert!(files.is_empty());
    }
}
