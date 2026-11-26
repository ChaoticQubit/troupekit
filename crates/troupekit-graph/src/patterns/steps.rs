use crate::error::WorkflowError;
use toml::Table;

/// Compiles steps pattern `text` read from `file` into a flow table.
/// Not built yet: every call returns a `WorkflowError` naming the
/// `steps` field.
pub fn compile(file: &str, text: &str) -> Result<Table, WorkflowError> {
    let _ = text;
    Err(WorkflowError::new(
        file,
        "steps",
        "steps patterns are not built yet",
    ))
}
