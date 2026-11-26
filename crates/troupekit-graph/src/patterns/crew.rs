use crate::error::WorkflowError;
use toml::Table;

/// Compiles crew pattern `text` read from `file` into a flow table.
/// Not built yet: every call returns a `WorkflowError` naming the
/// `crew` field.
pub fn compile(file: &str, text: &str) -> Result<Table, WorkflowError> {
    let _ = text;
    Err(WorkflowError::new(
        file,
        "crew",
        "crew patterns are not built yet",
    ))
}
