use serde_json::{json, Value};

/// Permission level required to execute a tool.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum PermissionLevel {
    /// Safe read-only operations.
    ReadOnly,
    /// Can modify files within the workspace.
    WorkspaceWrite,
    /// Can execute arbitrary commands. Requires explicit approval.
    Execute,
}

/// A tool specification sent to the model as a function definition.
#[derive(Debug, Clone)]
pub struct ToolSpec {
    pub name: &'static str,
    pub description: &'static str,
    pub parameters: Value,
    pub permission: PermissionLevel,
}

impl ToolSpec {
    /// Convert to the OpenAI-compatible tool definition format.
    pub fn to_definition(&self) -> Value {
        json!({
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            }
        })
    }
}

/// Central registry of all available tools.
pub struct ToolRegistry {
    specs: Vec<ToolSpec>,
}

impl ToolRegistry {
    /// Create a registry with the default built-in tools.
    pub fn with_builtins() -> Self {
        Self {
            specs: builtin_tool_specs(),
        }
    }

    /// Get all tool specifications.
    pub fn specs(&self) -> &[ToolSpec] {
        &self.specs
    }

    /// Get tool definitions in the format expected by chat completions APIs.
    pub fn definitions(&self) -> Vec<Value> {
        self.specs.iter().map(|s| s.to_definition()).collect()
    }

    /// Look up a tool spec by name.
    pub fn get(&self, name: &str) -> Option<&ToolSpec> {
        self.specs.iter().find(|s| s.name == name)
    }
}

/// All built-in tool specifications.
fn builtin_tool_specs() -> Vec<ToolSpec> {
    vec![
        ToolSpec {
            name: "bash",
            description: "Execute a shell command and return its output. Use for running \
                          programs, git operations, build commands, and system tasks. \
                          The command runs in the current working directory.",
            parameters: json!({
                "type": "object",
                "required": ["command"],
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute"
                    },
                    "timeout_ms": {
                        "type": "integer",
                        "description": "Timeout in milliseconds (default: 120000)"
                    }
                },
                "additionalProperties": false
            }),
            permission: PermissionLevel::Execute,
        },
        ToolSpec {
            name: "read_file",
            description: "Read the contents of a file. Returns the file content with line numbers. \
                          For large files, use offset and limit to read specific sections.",
            parameters: json!({
                "type": "object",
                "required": ["file_path"],
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Absolute path to the file to read"
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Line number to start reading from (1-based)"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of lines to read"
                    }
                },
                "additionalProperties": false
            }),
            permission: PermissionLevel::ReadOnly,
        },
        ToolSpec {
            name: "write_file",
            description: "Write content to a file, creating it if it doesn't exist or \
                          overwriting if it does. Creates parent directories as needed.",
            parameters: json!({
                "type": "object",
                "required": ["file_path", "content"],
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Absolute path to the file to write"
                    },
                    "content": {
                        "type": "string",
                        "description": "The content to write to the file"
                    }
                },
                "additionalProperties": false
            }),
            permission: PermissionLevel::WorkspaceWrite,
        },
        ToolSpec {
            name: "edit_file",
            description: "Make a targeted edit to a file by replacing an exact string match. \
                          The old_string must match exactly one location in the file.",
            parameters: json!({
                "type": "object",
                "required": ["file_path", "old_string", "new_string"],
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Absolute path to the file to edit"
                    },
                    "old_string": {
                        "type": "string",
                        "description": "The exact string to find and replace"
                    },
                    "new_string": {
                        "type": "string",
                        "description": "The replacement string"
                    }
                },
                "additionalProperties": false
            }),
            permission: PermissionLevel::WorkspaceWrite,
        },
        ToolSpec {
            name: "glob_search",
            description: "Find files matching a glob pattern. Returns matching file paths \
                          sorted by modification time.",
            parameters: json!({
                "type": "object",
                "required": ["pattern"],
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern to match (e.g., '**/*.rs', 'src/**/*.ts')"
                    },
                    "path": {
                        "type": "string",
                        "description": "Base directory to search from (default: cwd)"
                    }
                },
                "additionalProperties": false
            }),
            permission: PermissionLevel::ReadOnly,
        },
        ToolSpec {
            name: "grep_search",
            description: "Search file contents using a regular expression. Returns matching \
                          file paths, or matching lines with context.",
            parameters: json!({
                "type": "object",
                "required": ["pattern"],
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Regular expression pattern to search for"
                    },
                    "path": {
                        "type": "string",
                        "description": "File or directory to search in (default: cwd)"
                    },
                    "include": {
                        "type": "string",
                        "description": "Glob pattern to filter files (e.g., '*.rs')"
                    },
                    "context_lines": {
                        "type": "integer",
                        "description": "Number of context lines before and after each match"
                    }
                },
                "additionalProperties": false
            }),
            permission: PermissionLevel::ReadOnly,
        },
        ToolSpec {
            name: "list_dir",
            description: "List the contents of a directory. Returns file and directory names \
                          with type indicators.",
            parameters: json!({
                "type": "object",
                "required": ["path"],
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute path to the directory to list"
                    }
                },
                "additionalProperties": false
            }),
            permission: PermissionLevel::ReadOnly,
        },
    ]
}
