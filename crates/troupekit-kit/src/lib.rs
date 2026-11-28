mod agent;
mod agent_history;
mod built;
mod error;
mod flow;
mod graph;
mod history;
#[cfg(feature = "knowledge")]
pub mod knowledge;
mod laws;
#[cfg(feature = "knowledge")]
mod memory;
mod model;
mod output;
mod patterns;
mod storage;
mod team_agent;
mod tool;

#[doc(hidden)]
pub mod __private;

pub use agent::Agent;
pub use built::{BuiltAgent, Input, RunOutcome};
pub use error::KitError;
pub use flow::Trigger;
pub use graph::{BuiltGraph, Graph, GraphReport};
#[cfg(feature = "knowledge")]
pub use knowledge::Knowledge;
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
    pub use crate::{
        agent, tool, Agent, BuiltAgent, BuiltGraph, Graph, Json, KitError, Laws, RunOutcome,
        Storage,
    };
    #[cfg(feature = "knowledge")]
    pub use crate::{AgentMemory, Knowledge};
}
