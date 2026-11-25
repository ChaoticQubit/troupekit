mod agent;
mod built;
mod error;
mod graph;
mod laws;
#[cfg(feature = "knowledge")]
mod memory;
mod model;
mod output;
mod storage;
mod tool;

#[doc(hidden)]
pub mod __private;

pub use agent::Agent;
pub use built::{BuiltAgent, Input, RunOutcome};
pub use error::KitError;
pub use graph::{BuiltGraph, Graph, GraphReport};
pub use laws::Laws;
#[cfg(feature = "knowledge")]
pub use memory::AgentMemory;
pub use output::{IntoToolOutput, Json};
pub use storage::Storage;
pub use tool::{tool, AsTool};
pub use troupekit_macros::{agent, tool};
pub use troupekit_tools::{Tool, ToolContext, ToolOutput};

/// The names most programs need, for `use troupekit::prelude::*;`.
/// Builder, built agent, outcome, storage, laws, errors and tools.
/// `AsTool`, `IntoToolOutput` and `Input` stay at the crate root.
pub mod prelude {
    #[cfg(feature = "knowledge")]
    pub use crate::AgentMemory;
    pub use crate::{
        agent, tool, Agent, BuiltAgent, BuiltGraph, Graph, Json, KitError, Laws, RunOutcome,
        Storage,
    };
}
