use troupekit_core::error::CoreError;
use troupekit_core::ids::RunId;
use troupekit_graph::GraphError;

/// Why a host call failed: a graph error (bad checkpoint, resume that does
/// not fit), a store or queue failure, or a run with no checkpoint.
/// A run that fails during a step is not an error: its job fails instead.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum HostError {
    #[error(transparent)]
    Graph(#[from] GraphError),
    #[error(transparent)]
    Store(#[from] CoreError),
    #[error("run {0} has no checkpoint")]
    NoCheckpoint(RunId),
}
