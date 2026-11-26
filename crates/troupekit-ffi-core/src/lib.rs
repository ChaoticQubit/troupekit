mod agent;
mod error;
mod graph;
mod host;
mod records;
mod runtime;
mod stream;

pub use agent::{AgentBuilder, AgentHandle};
pub use error::{ErrorCode, FfiError};
pub use graph::{GraphBuilder, GraphHandle};
#[cfg(feature = "knowledge")]
pub use host::HostEmbedder;
pub use host::{EventListener, HostModel, HostTool, HostTransport, HttpSink, TokenSink};
pub use records::{
    FfiEvent, FfiEventKind, FfiPause, FfiUsage, GraphResult, HostError, HostHeader,
    HostHttpRequest, HostMessage, HostModelCaps, HostReply, HostRequest, Progress, RunResult,
};
#[cfg(feature = "knowledge")]
pub use records::{KnowledgeConfig, MemoryConfig};
pub use runtime::{init, RuntimeSettings};
pub use stream::RunStream;

/// Returns the troupekit version this library was built from.
/// Input: none. Output: the workspace package version, for example `0.1.0`,
/// which every troupekit crate shares, so it matches `troupekit-core`.
pub fn version() -> String {
    let value = env!("CARGO_PKG_VERSION");
    assert!(!value.is_empty());
    assert!(value.split('.').count() == 3);
    value.to_owned()
}
