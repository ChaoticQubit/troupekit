//! Persona TOML agents for Troupekit: a layered persona loader with strict
//! validation, and `AgentRun`, a serializable state machine the host steps
//! and checkpoints, with concurrent allowed tools, typed output, evals with
//! one repair round, layered budgets and core events.

pub mod budget;
pub mod delegate;
pub mod dsl;
pub mod error;
pub mod eval;
pub mod io;
pub mod loader;
pub mod persona;
pub mod run;
pub mod schema;

pub use io::{AgentIo, EventSink, ModelHandle, SimpleIo, VecSink};
pub use run::{drive, AgentRun, PendingCall, RunState, StepOutcome, RUN_FORMAT};

pub use delegate::{AgentTool, DelegateSettings, DelegationTools};
pub use budget::{InMemoryLedger, Limits, PeriodLimit, UsageLedger};
pub use dsl::{parse_call, Call};
pub use error::{AgentError, PersonaError};
pub use eval::{EvalFn, EvalRegistry};
pub use loader::{builtin, DirSource, MemorySource, PersonaLoader, PersonaSource};
pub use persona::{Budget, Persona, DEFAULT_MAX_TOKENS, DEFAULT_MAX_TURNS};
pub use schema::{compile_output, validate};
