use crate::agent::AgentBuilder;
use crate::error::FfiError;
use crate::host::HostTool;
use crate::records::{GraphResult, Progress};
use crate::stream::RunStream;
use std::sync::Arc;

/// Builds a graph of agent and step nodes in steps from the host
/// language. Every setter returns the same builder; nothing is checked
/// until `build`.
#[derive(Debug)]
pub struct GraphBuilder {
    _private: (),
}

impl GraphBuilder {
    /// Starts a builder for a graph named `name`.
    /// Input: the name, shown in events and errors.
    /// Output: a new builder.
    pub fn new(_name: String) -> Arc<Self> {
        Arc::new(GraphBuilder { _private: () })
    }

    /// Adds node `id` that runs `agent` on the text its feeder wrote.
    /// The agent builder is read at build.
    /// Output: the same builder, for chaining.
    pub fn agent(self: &Arc<Self>, _id: String, _agent: Arc<AgentBuilder>) -> Arc<Self> {
        self.clone()
    }

    /// Adds node `id` that calls host tool `tool`; its one argument is
    /// named after the node that feeds it.
    /// Output: the same builder, for chaining.
    pub fn step(self: &Arc<Self>, _id: String, _tool: Arc<dyn HostTool>) -> Arc<Self> {
        self.clone()
    }

    /// Connects node `from` to node `to`. Without edges the nodes run in
    /// the order they were added.
    /// Output: the same builder, for chaining.
    pub fn edge(self: &Arc<Self>, _from: String, _to: String) -> Arc<Self> {
        self.clone()
    }

    /// Keeps graph checkpoints in the SQLite file at `path`, so runs
    /// survive a restart. Needed by `start` and `resume_for`.
    /// Output: the same builder, for chaining.
    pub fn sqlite(self: &Arc<Self>, _path: String) -> Arc<Self> {
        self.clone()
    }

    /// Caps the steps of one graph run.
    /// Zero is refused at build.
    /// Output: the same builder, for chaining.
    pub fn max_steps(self: &Arc<Self>, _steps: u32) -> Arc<Self> {
        self.clone()
    }

    /// Sets how often a failed node is retried.
    /// Bounded by the graph engine; checked at build.
    /// Output: the same builder, for chaining.
    pub fn retries(self: &Arc<Self>, _retries: u32) -> Arc<Self> {
        self.clone()
    }

    /// Adds the law file at `path`, a `.bend` file checked at build.
    /// The path must not contain `..`.
    /// Output: the same builder, for chaining.
    pub fn laws(self: &Arc<Self>, _path: String) -> Arc<Self> {
        self.clone()
    }

    /// Adds one law as text, checked at build.
    /// Laws guard every node.
    /// Output: the same builder, for chaining.
    pub fn law(self: &Arc<Self>, _text: String) -> Arc<Self> {
        self.clone()
    }

    /// Turns the built-in laws on or off.
    /// On by default.
    /// Output: the same builder, for chaining.
    pub fn auto_laws(self: &Arc<Self>, _on: bool) -> Arc<Self> {
        self.clone()
    }

    /// Checks the nodes, edges, agents, tools and storage and returns a
    /// ready graph. Makes no model call. Output: the graph handle, or
    /// the first problem found with its code.
    pub async fn build(self: &Arc<Self>) -> Result<Arc<GraphHandle>, FfiError> {
        Err(FfiError::unsupported("GraphBuilder::build"))
    }
}

/// A built graph, ready to run many times.
/// Outputs cross as JSON text; run ids as text.
/// Agent nodes report events under their own run ids.
#[derive(Debug)]
pub struct GraphHandle {
    _private: (),
}

impl GraphHandle {
    /// Runs the graph on `input` to the end.
    /// Output: the last node's output as JSON text, or the error with
    /// its code and run id.
    pub async fn run(&self, _input: String) -> Result<String, FfiError> {
        Err(FfiError::unsupported("GraphHandle::run"))
    }

    /// Runs the graph on `input` to the end.
    /// Output: the whole result: output and state as JSON text, usage,
    /// steps and run id.
    pub async fn run_result(&self, _input: String) -> Result<GraphResult, FfiError> {
        Err(FfiError::unsupported("GraphHandle::run_result"))
    }

    /// Starts a graph run on `input` with no node run yet and saves its
    /// first checkpoint. Needs `sqlite`.
    /// Output: the run id text, to continue with `resume` or `resume_for`.
    pub async fn start(&self, _input: String) -> Result<String, FfiError> {
        Err(FfiError::unsupported("GraphHandle::start"))
    }

    /// Continues graph run `run_id` from its last checkpoint to the end.
    /// Output: the last node's output as JSON text, or the error.
    /// The run must have been started by the same workflow.
    pub async fn resume(&self, _run_id: String) -> Result<String, FfiError> {
        Err(FfiError::unsupported("GraphHandle::resume"))
    }

    /// Continues graph run `run_id` for about `budget_ms` milliseconds,
    /// saving after every step. Output: `Done` with the last output, or
    /// `Pending` when time ran out.
    pub async fn resume_for(&self, _run_id: String, _budget_ms: u64) -> Result<Progress, FfiError> {
        Err(FfiError::unsupported("GraphHandle::resume_for"))
    }

    /// Starts a graph run on `input` and returns its event stream once the
    /// run id is known. Output: the stream, or the setup error (input,
    /// tools, models) with its code, as `start` reports it.
    pub async fn stream(&self, _input: String) -> Result<Arc<RunStream>, FfiError> {
        Err(FfiError::unsupported("GraphHandle::stream"))
    }
}
