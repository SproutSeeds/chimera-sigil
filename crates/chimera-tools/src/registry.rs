use serde_json::{Value, json};

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
        let canonical = resolve_alias(name);
        self.specs.iter().find(|s| s.name == canonical)
    }
}

/// Resolve short tool aliases to canonical names.
/// Inspired by claw-code's alias normalization pattern.
pub fn resolve_alias(name: &str) -> &str {
    match name {
        "read" => "read_file",
        "write" => "write_file",
        "edit" => "edit_file",
        "glob" => "glob_search",
        "grep" => "grep_search",
        "ls" | "dir" => "list_dir",
        "sh" | "shell" | "exec" => "bash",
        "structured" | "json_output" => "structured_output",
        other => other,
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
        ToolSpec {
            name: "structured_output",
            description: "Return structured data as JSON. Use this when the user requests \
                          data in a specific format (tables, lists, configs, summaries). \
                          Pass any JSON object as the payload.",
            parameters: json!({
                "type": "object",
                "additionalProperties": true
            }),
            permission: PermissionLevel::ReadOnly,
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_has_all_builtins() {
        let registry = ToolRegistry::with_builtins();
        let names: Vec<&str> = registry.specs().iter().map(|s| s.name).collect();

        assert!(names.contains(&"bash"));
        assert!(names.contains(&"read_file"));
        assert!(names.contains(&"write_file"));
        assert!(names.contains(&"edit_file"));
        assert!(names.contains(&"glob_search"));
        assert!(names.contains(&"grep_search"));
        assert!(names.contains(&"list_dir"));
        assert!(names.contains(&"structured_output"));
        assert_eq!(names.len(), 8);
    }

    #[test]
    fn test_registry_get() {
        let registry = ToolRegistry::with_builtins();

        let bash = registry.get("bash").unwrap();
        assert_eq!(bash.name, "bash");
        assert_eq!(bash.permission, PermissionLevel::Execute);

        let read = registry.get("read_file").unwrap();
        assert_eq!(read.permission, PermissionLevel::ReadOnly);

        assert!(registry.get("nonexistent").is_none());
    }

    #[test]
    fn test_registry_get_resolves_aliases() {
        let registry = ToolRegistry::with_builtins();

        // Short aliases should resolve to canonical names
        assert_eq!(registry.get("read").unwrap().name, "read_file");
        assert_eq!(registry.get("write").unwrap().name, "write_file");
        assert_eq!(registry.get("edit").unwrap().name, "edit_file");
        assert_eq!(registry.get("glob").unwrap().name, "glob_search");
        assert_eq!(registry.get("grep").unwrap().name, "grep_search");
        assert_eq!(registry.get("ls").unwrap().name, "list_dir");
        assert_eq!(registry.get("sh").unwrap().name, "bash");
        assert_eq!(
            registry.get("structured").unwrap().name,
            "structured_output"
        );
    }

    #[test]
    fn test_resolve_alias_passthrough() {
        // Non-aliased names pass through unchanged
        assert_eq!(resolve_alias("bash"), "bash");
        assert_eq!(resolve_alias("read_file"), "read_file");
        assert_eq!(resolve_alias("unknown_tool"), "unknown_tool");
    }

    #[test]
    fn test_tool_definitions_are_valid_json() {
        let registry = ToolRegistry::with_builtins();
        let defs = registry.definitions();

        for def in &defs {
            assert_eq!(def["type"], "function");
            assert!(def["function"]["name"].is_string());
            assert!(def["function"]["description"].is_string());
            assert!(def["function"]["parameters"].is_object());

            // Each should have "type": "object" in parameters
            assert_eq!(def["function"]["parameters"]["type"], "object");
        }
    }

    #[test]
    fn test_tool_definitions_deserialize_to_tool_definition() {
        // Verify the definitions can round-trip through ToolDefinition
        use chimera_sigil_providers::types::ToolDefinition;

        let registry = ToolRegistry::with_builtins();
        for def in registry.definitions() {
            let result: Result<ToolDefinition, _> = serde_json::from_value(def.clone());
            assert!(
                result.is_ok(),
                "Failed to deserialize tool definition: {:?}",
                def
            );
        }
    }

    #[test]
    fn test_permission_ordering() {
        assert!(PermissionLevel::ReadOnly < PermissionLevel::WorkspaceWrite);
        assert!(PermissionLevel::WorkspaceWrite < PermissionLevel::Execute);
    }
}
