use crate::agent::{parse_run_id, AgentBuilder};
use crate::error::{ErrorCode, FfiError};
use crate::host::{guarded, host_tool, HostTool};
use crate::records::{GraphResult, Progress};
use crate::runtime::{check_input, spawn};
use crate::stream::RunStream;
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;
use troupekit_kit::{BuiltGraph, Graph, GraphReport, StreamSettings};

/// Builds a graph of agent and step nodes in steps from the host
/// language. Every setter returns the same builder; nothing is checked
/// until `build`, which consumes the settings.
#[derive(Debug)]
pub struct GraphBuilder {
    state: Mutex<State>,
}

#[derive(Debug)]
enum State {
    Open(Box<Graph>),
    Broken(FfiError),
    Built,
}

impl GraphBuilder {
    /// Starts a builder for a graph named `name`.
    /// Input: the name, shown in events and errors.
    /// Output: a new builder.
    pub fn new(name: String) -> Arc<Self> {
        let state = match check_input("name", &name) {
            Ok(()) => State::Open(Box::new(Graph::new(name))),
            Err(err) => State::Broken(err),
        };
        Arc::new(GraphBuilder {
            state: Mutex::new(state),
        })
    }

    fn update(
        self: &Arc<Self>,
        change: impl FnOnce(Graph) -> Result<Graph, FfiError>,
    ) -> Arc<Self> {
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        let next = match std::mem::replace(&mut *state, State::Built) {
            State::Open(graph) => match change(*graph) {
                Ok(graph) => State::Open(Box::new(graph)),
                Err(err) => State::Broken(err),
            },
            other => other,
        };
        *state = next;
        self.clone()
    }

    /// Adds node `id` that runs `agent` on the text its feeder wrote.
    /// The agent builder's settings are taken now, so it cannot be built
    /// on its own afterwards. Output: the same builder, for chaining.
    pub fn agent(self: &Arc<Self>, id: String, agent: Arc<AgentBuilder>) -> Arc<Self> {
        let taken = agent.take();
        self.update(|graph| {
            check_input("node id", &id)?;
            Ok(graph.agent(id, taken?))
        })
    }

    /// Adds node `id` that calls host tool `tool`; its argument names
    /// must be the ids of the nodes that feed it (`input` for the first).
    /// Output: the same builder, for chaining.
    pub fn step(self: &Arc<Self>, id: String, tool: Arc<dyn HostTool>) -> Arc<Self> {
        let wrapped = guarded("tool", || host_tool(tool));
        self.update(|graph| {
            check_input("node id", &id)?;
            Ok(graph.step(id, wrapped?))
        })
    }

    /// Connects node `from` to node `to`. Without edges the nodes run in
    /// the order they were added.
    /// Output: the same builder, for chaining.
    pub fn edge(self: &Arc<Self>, from: String, to: String) -> Arc<Self> {
        self.update(|graph| {
            check_input("edge from", &from)?;
            check_input("edge to", &to)?;
            Ok(graph.edge(from, to))
        })
    }

    /// Keeps graph checkpoints in the SQLite file at `path`, so runs
    /// survive a restart. Needed by `start` and `resume_for`.
    /// Output: the same builder, for chaining.
    pub fn sqlite(self: &Arc<Self>, path: String) -> Arc<Self> {
        self.update(|graph| {
            check_input("sqlite", &path)?;
            sqlite_storage(graph, path)
        })
    }

    /// Caps the steps of one graph run.
    /// Zero is refused at build.
    /// Output: the same builder, for chaining.
    pub fn max_steps(self: &Arc<Self>, steps: u32) -> Arc<Self> {
        self.update(|graph| Ok(graph.max_steps(steps)))
    }

    /// Sets how often a failed node is retried.
    /// Bounded by the graph engine; checked at build.
    /// Output: the same builder, for chaining.
    pub fn retries(self: &Arc<Self>, retries: u32) -> Arc<Self> {
        self.update(|graph| Ok(graph.retries(retries)))
    }

    /// Adds the law file at `path`, a `.bend` file checked at build.
    /// The path must not contain `..`.
    /// Output: the same builder, for chaining.
    pub fn laws(self: &Arc<Self>, path: String) -> Arc<Self> {
        self.update(|graph| {
            check_input("laws", &path)?;
            Ok(graph.laws(path))
        })
    }

    /// Adds one law as text, checked at build.
    /// Laws guard every node.
    /// Output: the same builder, for chaining.
    pub fn law(self: &Arc<Self>, text: String) -> Arc<Self> {
        self.update(|graph| {
            check_input("law", &text)?;
            Ok(graph.law(text))
        })
    }

    /// Turns the built-in laws on or off.
    /// On by default.
    /// Output: the same builder, for chaining.
    pub fn auto_laws(self: &Arc<Self>, on: bool) -> Arc<Self> {
        self.update(|graph| Ok(graph.auto_laws(on)))
    }

    /// Checks the nodes, edges, agents, tools and storage and returns a
    /// ready graph. Makes no model call. Output: the graph handle, or
    /// the first problem found with its code.
    pub async fn build(self: &Arc<Self>) -> Result<Arc<GraphHandle>, FfiError> {
        let graph = {
            let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
            match std::mem::replace(&mut *state, State::Built) {
                State::Open(graph) => *graph,
                State::Broken(err) => return Err(err),
                State::Built => {
                    return Err(FfiError::new(
                        ErrorCode::InvalidArgument,
                        "builder already built",
                    ))
                }
            }
        };
        let built = spawn(async move { graph.build().await.map_err(FfiError::from) }).await?;
        Ok(Arc::new(GraphHandle { graph: built }))
    }
}

#[cfg(feature = "sqlite")]
fn sqlite_storage(graph: Graph, path: String) -> Result<Graph, FfiError> {
    Ok(graph.checkpoints(troupekit_kit::Storage::sqlite(path)))
}

#[cfg(not(feature = "sqlite"))]
fn sqlite_storage(_graph: Graph, _path: String) -> Result<Graph, FfiError> {
    Err(FfiError::new(
        ErrorCode::Unsupported,
        "sqlite needs troupekit built with the sqlite feature",
    ))
}

/// A built graph, ready to run many times.
/// Outputs cross as JSON text; run ids as text.
/// Agent nodes report events under their own run ids.
#[derive(Debug)]
pub struct GraphHandle {
    graph: BuiltGraph,
}

impl GraphHandle {
    /// Runs the graph on `input` to the end.
    /// Output: the last node's output as JSON text, or the error with
    /// its code and run id.
    pub async fn run(&self, input: String) -> Result<String, FfiError> {
        check_input("input", &input)?;
        let graph = self.graph.clone();
        let output = spawn(async move { graph.run(input).await.map_err(FfiError::from) }).await?;
        Ok(output.to_string())
    }

    /// Runs the graph on `input` to the end.
    /// Output: the whole result: output and state as JSON text, usage,
    /// steps and run id.
    pub async fn run_result(&self, input: String) -> Result<GraphResult, FfiError> {
        check_input("input", &input)?;
        let graph = self.graph.clone();
        let report =
            spawn(async move { graph.run_with(input).await.map_err(FfiError::from) }).await?;
        Ok(GraphResult::from_report(&report))
    }

    /// Starts a graph run on `input` with no node run yet and saves its
    /// first checkpoint. Needs `sqlite`.
    /// Output: the run id text, to continue with `resume` or `resume_for`.
    pub async fn start(&self, input: String) -> Result<String, FfiError> {
        check_input("input", &input)?;
        let graph = self.graph.clone();
        let run_id = spawn(async move { graph.start(input).await.map_err(FfiError::from) }).await?;
        Ok(run_id.to_string())
    }

    /// Continues graph run `run_id` from its last checkpoint to the end.
    /// Output: the last node's output as JSON text, or the error.
    /// The run must have been started by the same workflow.
    pub async fn resume(&self, run_id: String) -> Result<String, FfiError> {
        let run_id = parse_run_id(&run_id)?;
        let graph = self.graph.clone();
        let output =
            spawn(async move { graph.resume(run_id).await.map_err(FfiError::from) }).await?;
        Ok(output.to_string())
    }

    /// Continues graph run `run_id` for about `budget_ms` milliseconds,
    /// saving after every step. Output: `Done` with the last output as
    /// text, or `Pending` when time ran out.
    pub async fn resume_for(&self, run_id: String, budget_ms: u64) -> Result<Progress, FfiError> {
        let run_id = parse_run_id(&run_id)?;
        let graph = self.graph.clone();
        let budget = Duration::from_millis(budget_ms);
        let progress = spawn(async move {
            graph
                .resume_for(run_id, budget)
                .await
                .map_err(FfiError::from)
        })
        .await?;
        Progress::from_kit(progress)
    }

    /// Starts a graph run on `input` and returns its event stream once the
    /// run id is known. Its `result` holds the last output as text (a
    /// string as is, else JSON) and as JSON; `model_calls` is 0.
    pub async fn stream(&self, input: String) -> Result<Arc<RunStream>, FfiError> {
        check_input("input", &input)?;
        let graph = self.graph.clone();
        let mut settings = StreamSettings::default();
        settings.capacity = crate::runtime::settings().event_buffer;
        let stream = spawn(async move {
            graph
                .stream_with(input, settings)
                .await
                .map_err(FfiError::from)
        })
        .await?;
        Ok(RunStream::graph(stream))
    }
}

impl GraphResult {
    pub(crate) fn from_report(report: &GraphReport) -> GraphResult {
        let state = serde_json::Value::Object(report.state().clone());
        GraphResult {
            output_json: report.output().to_string(),
            state_json: state.to_string(),
            usage: crate::stream::usage_of(report.usage()),
            steps: report.steps(),
            run_id: report.run_id().to_string(),
        }
    }
}
