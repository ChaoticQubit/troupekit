use crate::error::WorkflowError;
use toml::Table;

/// Compiles handoff pattern `text` read from `file` into a flow table.
/// Not built yet: every call returns a `WorkflowError` naming the
/// `handoff` field.
pub fn compile(file: &str, text: &str) -> Result<Table, WorkflowError> {
    let _ = text;
    Err(WorkflowError::new(
        file,
        "handoff",
        "handoff patterns are not built yet",
    ))
}
