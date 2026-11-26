use crate::error::WorkflowError;
use crate::flow::Flow;

pub mod crew;
pub mod handoff;
pub mod steps;
pub mod team;

/// Loads pattern TOML `text` read from `file` (a team, crew, handoff or
/// steps pattern) and compiles it into a validated `Flow`.
/// Returns the first problem found.
/// Not built yet: every call returns a `WorkflowError`.
pub fn load(file: &str, text: &str) -> Result<Flow, WorkflowError> {
    let _ = text;
    Err(WorkflowError::new(file, "", "patterns are not built yet"))
}

/// Compiles pattern TOML `text` read from `file` into the flow TOML table
/// that `Flow::parse` reads, so builders can produce the same table.
/// Returns the table or the first problem found.
/// Not built yet: every call returns a `WorkflowError`.
pub fn compile(file: &str, text: &str) -> Result<toml::Table, WorkflowError> {
    let _ = text;
    Err(WorkflowError::new(file, "", "patterns are not built yet"))
}
