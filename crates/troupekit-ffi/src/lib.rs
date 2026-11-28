uniffi::setup_scaffolding!();

mod agent;
mod error;
mod graph;
mod host;
mod records;
mod stream;

pub use agent::{AgentBuilder, AgentHandle};
pub use error::{ErrorCode, HostError, TroupekitError};
pub use graph::{GraphBuilder, GraphHandle};
#[cfg(feature = "knowledge")]
pub use host::HostEmbedder;
pub use host::{EventListener, HostModel, HostTool, HostTransport, HttpSink, TokenSink};
pub use records::{
    Event, EventKind, GraphResult, HostHeader, HostHttpRequest, HostMessage, HostModelCaps,
    HostReply, HostRequest, Pause, Progress, RunResult, RuntimeSettings, Usage,
};
#[cfg(feature = "knowledge")]
pub use records::{KnowledgeConfig, MemoryConfig};
pub use stream::RunStream;

/// Returns the troupekit version, exported to Swift and Kotlin.
/// Input: none. Output: the value of `troupekit_ffi_core::version`,
/// passed through unchanged.
#[uniffi::export]
pub fn version() -> String {
    troupekit_ffi_core::version()
}

/// Sets the process-wide runtime once, before the first agent is built.
/// Exported as `initialize` (freeze open question Q2). Input: the settings.
/// Output: the neutral layer's result, its error converted.
#[uniffi::export(name = "initialize")]
pub fn init(settings: RuntimeSettings) -> Result<(), TroupekitError> {
    Ok(troupekit_ffi_core::init(settings.into())?)
}
