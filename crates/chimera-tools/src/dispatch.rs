use crate::builtins;
use crate::registry::resolve_alias;
use serde_json::Value;
use tracing::{debug, warn};

/// Execute a tool by name with the given JSON arguments.
/// Resolves aliases (e.g., "read" → "read_file") before dispatch.
pub async fn execute_tool(name: &str, arguments: &str) -> anyhow::Result<String> {
    let canonical = resolve_alias(name);
    let input: Value = serde_json::from_str(arguments)
        .map_err(|e| anyhow::anyhow!("Invalid tool arguments for '{canonical}': {e}"))?;

    debug!("Executing tool '{canonical}' (requested as '{name}')");

    match canonical {
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
        "structured_output" => builtins::structured_output::run(input),
        _ => {
            warn!("Unknown tool: {canonical}");
            anyhow::bail!("unknown tool '{canonical}'");
        }
    }
}
