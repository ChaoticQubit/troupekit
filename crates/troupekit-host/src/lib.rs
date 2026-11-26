//! Hosting for Troupekit graph runs: a job queue with one leased job per
//! run, an event log that dedupes on `(run_id, seq)`, and a `Runner` that
//! claims a job, steps the run from its latest checkpoint and saves it, in
//! worker mode (a loop) or step mode (one `tick` per serverless request).

pub mod error;
pub mod events;
pub mod queue;
pub mod runner;

pub use error::HostError;
pub use events::EventLog;
pub use queue::{InMemQueue, JobQueue, JobState, Lease};
pub use runner::{Processed, Runner, TickOptions, TickReport, WorkOptions};

