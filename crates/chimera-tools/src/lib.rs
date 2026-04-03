pub mod registry;
pub mod dispatch;
pub mod builtins;

pub use registry::{ToolRegistry, ToolSpec, PermissionLevel, resolve_alias};
pub use dispatch::execute_tool;
