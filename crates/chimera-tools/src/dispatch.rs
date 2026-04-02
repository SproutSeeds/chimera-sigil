use crate::builtins;
use serde_json::Value;
use tracing::{debug, warn};

/// Execute a tool by name with the given JSON arguments.
/// Returns the tool output as a string.
pub async fn execute_tool(name: &str, arguments: &str) -> anyhow::Result<String> {
    let input: Value = serde_json::from_str(arguments).map_err(|e| {
        anyhow::anyhow!("Invalid tool arguments for '{name}': {e}")
    })?;

    debug!("Executing tool '{name}'");

    match name {
        "bash" => {
            let params: builtins::bash::BashInput = serde_json::from_value(input)?;
            builtins::bash::run(params).await
        }
        "read_file" => {
            let params: builtins::read_file::ReadFileInput = serde_json::from_value(input)?;
            builtins::read_file::run(params)
        }
        "write_file" => {
            let params: builtins::write_file::WriteFileInput = serde_json::from_value(input)?;
            builtins::write_file::run(params)
        }
        "edit_file" => {
            let params: builtins::edit_file::EditFileInput = serde_json::from_value(input)?;
            builtins::edit_file::run(params)
        }
        "glob_search" => {
            let params: builtins::glob_search::GlobSearchInput = serde_json::from_value(input)?;
            builtins::glob_search::run(params)
        }
        "grep_search" => {
            let params: builtins::grep_search::GrepSearchInput = serde_json::from_value(input)?;
            builtins::grep_search::run(params)
        }
        "list_dir" => {
            let params: builtins::list_dir::ListDirInput = serde_json::from_value(input)?;
            builtins::list_dir::run(params)
        }
        _ => {
            warn!("Unknown tool: {name}");
            Ok(format!("Error: unknown tool '{name}'"))
        }
    }
}
