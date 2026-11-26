//! Durable workflows for Troupekit: a TOML graph of agent, tool, human,
//! router, join and end nodes over a typed shared state, run by `GraphRun`,
//! a serializable state machine the host steps and checkpoints, with
//! parallel branches, human pauses, graph budgets and retries.

pub mod condition;
pub mod error;
pub mod io;
pub mod run;
pub mod testing;
pub mod workflow;

pub use condition::{ConditionFn, Conditions};
pub use error::{GraphError, WorkflowError};
pub use io::{GraphIo, SimpleGraphIo};
pub use run::{GraphOutcome, GraphRun, GraphStatus, HumanAsk, Stop, GRAPH_FORMAT};
pub use workflow::{Ask, Edge, Node, NodeKind, Workflow};

