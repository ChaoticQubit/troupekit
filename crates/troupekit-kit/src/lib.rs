mod agent;
mod built;
mod error;
mod laws;
mod model;
mod output;
mod storage;
mod tool;

#[doc(hidden)]
pub mod __private;

pub use agent::Agent;
pub use built::{BuiltAgent, Input, RunOutcome};
pub use error::KitError;
pub use laws::Laws;
pub use output::{IntoToolOutput, Json};
pub use storage::Storage;
pub use tool::{tool, AsTool};

/// The names most programs need, for `use troupekit::prelude::*;`.
/// Builder, built agent, outcome, storage, laws, errors and tools.
/// `AsTool`, `IntoToolOutput` and `Input` stay at the crate root.
pub mod prelude {
    pub use crate::{tool, Agent, BuiltAgent, Json, KitError, Laws, RunOutcome, Storage};
}

use troupekit_graph as _;
use troupekit_macros as _;
