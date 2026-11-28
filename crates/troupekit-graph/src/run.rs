use crate::error::GraphError;
use crate::io::{Emitter, GraphIo, NodeIo};
use crate::workflow::{Ask, Node, NodeKind, Workflow};
use futures_util::future::join_all;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::sync::atomic::{AtomicU64, Ordering};
use troupekit_agent::{AgentRun, Budget, Limits, RunState, StepOutcome};
use troupekit_core::event::{EventBody, PauseReason, Usage};
use troupekit_core::ids::RunId;
use troupekit_core::message::{Message, Role};
use troupekit_core::store::Checkpoint;
use troupekit_tools::{fence_untrusted, ToolContext, ToolError};
use troupekit_agent::persona::{open_trace, SavedTrace};
use troupekit_trace::{content, trace, trace_span, Span, Trace, TraceSpec};

/// Version of the serialized graph run.
/// `from_checkpoint` refuses any other, so a deploy never resumes
/// a run it cannot read.
pub const GRAPH_FORMAT: u32 = 1;

const MAX_RETRY_DELAY_MS: u64 = 300_000;
const BASE_RETRY_DELAY_MS: u64 = 1_000;

/// Where a graph run stands, as in the run lifecycle: `queued` until its
/// first step or after a resume, `running` only inside `step`,
/// `checkpointed` between steps, waiting for a human or budget, or final.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case", deny_unknown_fields)]
pub enum GraphStatus {
    Queued,
    Running,
    Checkpointed,
    WaitingHuman { reason: PauseReason },
    WaitingBudget { reason: PauseReason },
    Done,
    Failed { error: String },
}

/// Why a run was stopped from outside a step: it used up its active-work
/// time limit, or its owner cancelled it. A stopped run is `failed` and
/// takes no step until `GraphRun::retry`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Stop {
    TimeLimit { limit_ms: u64 },
    Cancelled,
}

impl Stop {
    fn error(&self) -> String {
        match self {
            Stop::TimeLimit { limit_ms } => format!("time limit of {limit_ms} ms of active work reached"),
            Stop::Cancelled => "cancelled".into(),
        }
    }
}

/// Why a run failed inside a step. `Tool`: a tool node's call failed on a
/// store or network error; `Transient`: transient retries ran out. Both
/// can be retried. `Final`: anything else, which a retry would repeat.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Failure {
    Final,
    Tool,
    Transient,
}

impl Failure {
    fn of_error(error: &str) -> Failure {
        let after_node = error
            .strip_prefix("node `")
            .and_then(|rest| rest.split_once("`: tool `"))
            .and_then(|(_, rest)| rest.split_once("`: "))
            .map(|(_, rest)| rest);
        let retries = error
            .strip_prefix("error after ")
            .is_some_and(|rest| rest.contains(" retries: "));
        match after_node {
            Some(rest) if rest.starts_with("failed: ") || rest.starts_with("http: ") => Failure::Tool,
            _ if retries => Failure::Transient,
            _ => Failure::Final,
        }
    }
}

/// The step and branch index a tool call first ran with, kept on a branch
/// a retry re-runs so the call keeps its idempotency key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CallKey {
    step: u64,
    index: usize,
}

/// A tool node call that failed: the message, and whether the error kind
/// is one a later call may not repeat (`failed`, `http`).
struct ToolFailure {
    message: String,
    transient: bool,
}

/// What one `step` did. `Continue`: step again now. `RetryAt`: a transient
/// failure; step again at `at_ms`. `Waiting`, `Done` and `Failed` repeat on
/// every further step until a resume or budget raise.
#[derive(Debug, Clone, PartialEq)]
pub enum GraphOutcome {
    Continue,
    RetryAt { at_ms: i64, error: String },
    Waiting(PauseReason),
    Done(Map<String, Value>),
    Failed(String),
}

/// A human node waiting for its answer: the node, what it asks, its
/// prompt and the state keys it reads, as the host shows them.
/// Answer it with `GraphRun::resume`.
#[derive(Debug, Clone, PartialEq)]
pub struct HumanAsk {
    pub node: String,
    pub ask: Ask,
    pub prompt: Option<String>,
    pub data: Map<String, Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "at", rename_all = "snake_case", deny_unknown_fields)]
enum At {
    Enter,
    Agent {
        run: Box<AgentRun>,
        call_cap: Option<u64>,
        run_cap: Option<u64>,
    },
    Human {
        data: Map<String, Value>,
    },
    Joined,
    Left,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Branch {
    node: String,
    at: At,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    call: Option<CallKey>,
}

impl Branch {
    fn enter(node: &str) -> Branch {
        Branch {
            node: node.to_string(),
            at: At::Enter,
            call: None,
        }
    }

    fn budget_paused(&self) -> Option<&PauseReason> {
        match &self.at {
            At::Agent { run, .. } => match run.state() {
                RunState::Paused { reason, .. } => Some(reason),
                _ => None,
            },
            _ => None,
        }
    }
}

enum Work {
    Idle,
    Agent(StepOutcome),
    Tool(Result<Value, ToolFailure>),
}

/// One graph run as plain serializable data: the workflow snapshot, the
/// limits, the typed state, the active branches, status, usage, retry
/// count, event `seq` and step number. The host calls `step` and saves
/// `checkpoint()` after every step whose number moved.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GraphRun {
    format: u32,
    run_id: RunId,
    workflow: Workflow,
    limits: Limits,
    state: Map<String, Value>,
    branches: Vec<Branch>,
    status: GraphStatus,
    usage: Usage,
    retries: u32,
    seq: u64,
    steps: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    time_limit_ms: Option<u64>,
    #[serde(default)]
    active_ms: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    stopped: Option<Stop>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    failure: Option<Failure>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    trace: Option<SavedTrace>,
}

impl GraphRun {
    /// Starts a run of `workflow` on `input` under `limits`. Checks before
    /// any step that every persona loads and can run, every tool and
    /// condition is registered and every input key is declared and typed.
    pub async fn new(
        workflow: Workflow,
        input: Map<String, Value>,
        limits: Limits,
        io: &dyn GraphIo,
    ) -> Result<GraphRun, GraphError> {
        let setup = |id: &str, m: String| GraphError::Setup(format!("node `{id}`: {m}"));
        for n in &workflow.nodes {
            match &n.kind {
                NodeKind::Agent { persona, .. } => {
                    let p = io
                        .persona(persona)
                        .await
                        .map_err(|e| setup(&n.id, e.to_string()))?;
                    let probe = vec![Message::text(Role::User, "setup check")];
                    AgentRun::new(&p, probe, Limits::default(), io.agent())
                        .await
                        .map_err(|e| setup(&n.id, e.to_string()))?;
                }
                NodeKind::Tool { tool } if io.agent().tools().get(tool).is_none() => {
                    return Err(setup(&n.id, format!("tool `{tool}` is not registered")));
                }
                NodeKind::Router { condition }
                    if io.conditions().get(&condition.name).is_none() =>
                {
                    return Err(setup(
                        &n.id,
                        format!("condition `{}` is not registered", condition.name),
                    ));
                }
                _ => {}
            }
        }
        for (k, v) in &input {
            if !workflow.state.contains_key(k) {
                return Err(GraphError::Setup(format!(
                    "input `{k}` is not declared in [state]"
                )));
            }
            let errs = workflow.check_value(k, v);
            if !errs.is_empty() {
                return Err(GraphError::Setup(format!(
                    "input `{k}`: {}",
                    errs.join("; ")
                )));
            }
        }
        let start = Branch::enter(&workflow.start);
        Ok(GraphRun {
            format: GRAPH_FORMAT,
            run_id: RunId::new(),
            workflow,
            limits,
            state: input,
            branches: vec![start],
            status: GraphStatus::Queued,
            usage: Usage::default(),
            retries: 0,
            seq: 0,
            steps: 0,
            time_limit_ms: None,
            active_ms: 0,
            stopped: None,
            failure: None,
            trace: None,
        })
    }

    /// Traces this run with `spec` from its next step on; the spec is kept
    /// in the checkpoint, so a resumed run keeps it. Input: the spec.
    /// Output: none.
    pub fn set_trace(&mut self, spec: TraceSpec) {
        self.trace = Some(SavedTrace::from_spec(&spec));
        assert_eq!(self.trace(), Some(spec));
    }

    /// Returns the trace spec set with `set_trace`, if any.
    /// Input: none. Output: the spec.
    pub fn trace(&self) -> Option<TraceSpec> {
        self.trace
            .as_ref()
            .map(|saved| saved.spec().expect("trace spec checked when set or loaded"))
    }

    /// Sets the active-work time limit: the run stops once its steps have
    /// taken `limit_ms` in total; time spent waiting never counts.
    /// Returns the run. Call it before the first step.
    pub fn with_time_limit(mut self, limit_ms: u64) -> GraphRun {
        assert!(limit_ms > 0, "a time limit allows some work");
        self.time_limit_ms = Some(limit_ms);
        self
    }

    /// Adds `ms` of active work, as the host measured one step.
    /// The host saves it with that step's checkpoint.
    /// Saturates instead of overflowing.
    pub fn add_active_ms(&mut self, ms: u64) {
        self.active_ms = self.active_ms.saturating_add(ms);
    }

    /// Returns the active work so far, in milliseconds.
    /// Only time spent inside steps counts.
    /// Zero for runs saved before time was tracked.
    pub fn active_ms(&self) -> u64 {
        self.active_ms
    }

    /// Returns the active-work time limit, if the run has one.
    /// Set at start or on retry.
    /// None means no limit.
    pub fn time_limit_ms(&self) -> Option<u64> {
        self.time_limit_ms
    }

    /// Whether the run has used up its active-work time limit.
    /// Always false without a limit.
    /// The host checks it before every step.
    pub fn over_time(&self) -> bool {
        self.time_limit_ms.is_some_and(|limit| self.active_ms >= limit)
    }

    /// Returns why the run was stopped, while it is stopped.
    /// None for runs that run, wait, finished or failed on their own.
    /// Cleared by `retry`.
    pub fn stopped(&self) -> Option<&Stop> {
        self.stopped.as_ref()
    }

    /// Stops the run for `why`: it becomes `failed` and takes no step until
    /// `retry`, so it calls no model. A durable change for the next save.
    /// Returns `Resume` when the run is already finished.
    pub fn stop(&mut self, why: Stop) -> Result<(), GraphError> {
        if matches!(self.status, GraphStatus::Done | GraphStatus::Failed { .. }) {
            return Err(GraphError::Resume("the run is finished".into()));
        }
        self.status = GraphStatus::Failed { error: why.error() };
        self.stopped = Some(why);
        self.steps += 1;
        Ok(())
    }

    /// Whether `retry` would continue the run: it was stopped, or it failed
    /// on a tool's store or network error or ran out of transient retries.
    /// False for runs that run, wait, finished, or failed for good.
    pub fn retryable(&self) -> bool {
        self.stopped.is_some() || matches!(self.failure, Some(Failure::Tool | Failure::Transient))
    }

    /// Continues a stopped or retryably failed run where it stopped, with
    /// active time back at zero and `time_limit_ms` as its new limit, and
    /// queues it. A failed tool call re-runs with its first idempotency key.
    /// Returns `Resume` when the run is not `retryable`.
    pub fn retry(&mut self, time_limit_ms: Option<u64>) -> Result<(), GraphError> {
        if !self.retryable() {
            return Err(GraphError::Resume(
                "only a stopped run or one failed on a transient error can be retried".into(),
            ));
        }
        assert!(matches!(self.status, GraphStatus::Failed { .. }), "a retryable run is failed");
        if self.stopped.is_none() && self.failure == Some(Failure::Tool) {
            let step = self.steps;
            let tools: Vec<bool> = self
                .branches
                .iter()
                .map(|b| b.at == At::Enter && matches!(self.node(&b.node).kind, NodeKind::Tool { .. }))
                .collect();
            assert!(tools.iter().any(|t| *t), "a tool failure leaves its branch at the tool");
            for (index, (b, tool)) in self.branches.iter_mut().zip(tools).enumerate() {
                if tool {
                    b.call.get_or_insert(CallKey { step, index });
                }
            }
        }
        self.stopped = None;
        self.failure = None;
        self.retries = 0;
        self.active_ms = 0;
        self.time_limit_ms = time_limit_ms;
        self.queue();
        Ok(())
    }

    /// Restores a run from `cp`. Returns `Checkpoint` when the state does
    /// not parse, has another `GRAPH_FORMAT`, holds an agent run of another
    /// format, is `running`, or does not match the checkpoint header.
    pub fn from_checkpoint(cp: &Checkpoint) -> Result<GraphRun, GraphError> {
        let bad = |m: String| GraphError::Checkpoint(m);
        let mut run: GraphRun =
            serde_json::from_value(cp.state.clone()).map_err(|e| bad(e.to_string()))?;
        if run.format != GRAPH_FORMAT {
            return Err(bad(format!(
                "graph format {} is not {GRAPH_FORMAT}",
                run.format
            )));
        }
        if run.run_id != cp.run_id || run.steps != cp.step {
            return Err(bad("checkpoint header does not match its state".into()));
        }
        if run.status == GraphStatus::Running {
            return Err(bad("a checkpoint never holds a running step".into()));
        }
        if let Some(saved) = &run.trace {
            saved.spec().map_err(bad)?;
        }
        for b in &run.branches {
            if let At::Agent { run: agent, .. } = &b.at {
                AgentRun::from_checkpoint(&agent.checkpoint())
                    .map_err(|e| bad(format!("node `{}`: {e}", b.node)))?;
            }
        }
        if let (GraphStatus::Failed { error }, None, None) = (&run.status, &run.stopped, run.failure) {
            run.failure = Some(Failure::of_error(error));
        }
        Ok(run)
    }

    /// Snapshots the run for a `CheckpointStore`: step 0 before the first
    /// step, then one more after every durable change.
    /// Returns the checkpoint.
    pub fn checkpoint(&self) -> Checkpoint {
        assert_ne!(
            self.status,
            GraphStatus::Running,
            "checkpoints are taken between steps"
        );
        Checkpoint {
            run_id: self.run_id,
            step: self.steps,
            state: serde_json::to_value(self).expect("graph run serializes"),
        }
    }

    /// Returns the run's id.
    /// Every event and checkpoint of the run carries it.
    /// It never changes.
    pub fn run_id(&self) -> RunId {
        self.run_id
    }

    /// Returns the current status.
    /// It changes only in `step`, `resume` and `raise_budget`.
    /// It is never `running` outside `step`.
    pub fn status(&self) -> &GraphStatus {
        &self.status
    }

    /// Returns the shared state: input, agent outputs, tool outputs and
    /// human answers by key.
    /// Every value matches its key's declared type.
    pub fn state(&self) -> &Map<String, Value> {
        &self.state
    }

    /// Returns tokens and cost of every agent model call so far, in
    /// finished and in-flight nodes.
    /// This is the spend the graph run budget caps.
    pub fn usage(&self) -> Usage {
        self.branches.iter().fold(self.usage, |acc, b| match &b.at {
            At::Agent { run, .. } => acc + run.usage(),
            _ => acc,
        })
    }

    /// Returns the step number: how many durable changes the run has had.
    /// It is the step of the next checkpoint.
    /// Starts at 0.
    pub fn steps(&self) -> u64 {
        self.steps
    }

    /// Returns the workflow snapshot the run started with.
    /// Edits to the workflow file never reach a running run.
    /// It never changes.
    pub fn workflow(&self) -> &Workflow {
        &self.workflow
    }

    /// Lists every human node waiting for an answer, first branch first,
    /// with its ask, prompt and the state keys it reads.
    /// Empty when no branch waits for a human.
    pub fn pending_human(&self) -> Vec<HumanAsk> {
        self.branches
            .iter()
            .filter_map(|b| {
                let At::Human { data } = &b.at else {
                    return None;
                };
                let NodeKind::Human { ask, prompt } = &self.node(&b.node).kind else {
                    unreachable!("human branches sit at human nodes");
                };
                Some(HumanAsk {
                    node: b.node.clone(),
                    ask: *ask,
                    prompt: prompt.clone(),
                    data: data.clone(),
                })
            })
            .collect()
    }

    /// Answers human node `node` with `answer`, checked against the node's
    /// `writes` type, and queues the run; the branch moves on at the next
    /// step. Returns `Resume` when no branch waits there or it does not fit.
    pub fn resume(&mut self, node: &str, answer: Value) -> Result<(), GraphError> {
        let bad = |m: String| GraphError::Resume(m);
        if matches!(self.status, GraphStatus::Done | GraphStatus::Failed { .. }) {
            return Err(bad("the run is finished".into()));
        }
        let i = self
            .branches
            .iter()
            .position(|b| b.node == node && matches!(b.at, At::Human { .. }))
            .ok_or_else(|| bad(format!("no branch waits at human node `{node}`")))?;
        let key = self
            .node(node)
            .writes
            .clone()
            .expect("human nodes write their answer");
        let errs = self.workflow.check_value(&key, &answer);
        if !errs.is_empty() {
            return Err(bad(format!(
                "answer does not match `{key}`: {}",
                errs.join("; ")
            )));
        }
        self.state.insert(key, answer);
        self.branches[i].at = At::Left;
        self.queue();
        Ok(())
    }

    /// Replaces the graph limits with `limits` and resumes every branch
    /// paused on its budget with the new remaining run budget, then queues
    /// the run. Returns `Resume` when the run is finished.
    pub fn raise_budget(&mut self, limits: Limits) -> Result<(), GraphError> {
        if matches!(self.status, GraphStatus::Done | GraphStatus::Failed { .. }) {
            return Err(GraphError::Resume("the run is finished".into()));
        }
        self.limits = limits;
        let spent = self.usage().cost_usd_micros;
        let base = self.limits.clone();
        for b in &mut self.branches {
            if b.budget_paused().is_none() {
                continue;
            }
            let At::Agent {
                run,
                call_cap,
                run_cap,
            } = &mut b.at
            else {
                unreachable!("budget pauses happen in agent branches");
            };
            let own = run.usage().cost_usd_micros;
            run.resume(node_limits(&base, spent - own, *call_cap, *run_cap))?;
        }
        self.queue();
        Ok(())
    }

    /// Runs one step: one unit of work (one agent step or one tool call)
    /// for every branch that can work, concurrently, with control nodes
    /// resolved before and after. Waiting and final runs change nothing.
    /// With a trace spec the step nests in the active scope, or writes its file.
    pub async fn step(&mut self, io: &dyn GraphIo) -> GraphOutcome {
        let working = matches!(self.status, GraphStatus::Queued | GraphStatus::Checkpointed);
        let Some(spec) = self.trace().filter(|_| working) else {
            return self.step_once(io).await;
        };
        if let Some(scope) = troupekit_trace::current() {
            let nested = scope.nest(spec.categories(), None);
            return nested.in_scope(self.step_once(io)).await;
        }
        match open_trace(&spec, &self.workflow.name, self.run_id) {
            Ok(tracer) => tracer.in_scope(self.step_once(io)).await,
            Err(_) => self.step_once(io).await,
        }
    }

    async fn step_once(&mut self, io: &dyn GraphIo) -> GraphOutcome {
        match &self.status {
            GraphStatus::Done => return GraphOutcome::Done(self.state.clone()),
            GraphStatus::Failed { error } => return GraphOutcome::Failed(error.clone()),
            GraphStatus::WaitingHuman { reason } | GraphStatus::WaitingBudget { reason } => {
                return GraphOutcome::Waiting(reason.clone())
            }
            GraphStatus::Running => unreachable!("a step never starts inside another"),
            GraphStatus::Queued | GraphStatus::Checkpointed => {}
        }
        self.steps += 1;
        self.status = GraphStatus::Running;
        let seq = AtomicU64::new(self.seq);
        let em = Emitter {
            run_id: self.run_id,
            seq: &seq,
            io: io.agent(),
        };
        if self.seq == 0 {
            em.emit(EventBody::RunStarted {
                workflow: self.workflow.name.clone(),
            });
        }
        self.failure = None;
        let outcome = self.work(io, &em).await;
        match &outcome {
            GraphOutcome::Continue | GraphOutcome::RetryAt { .. } => {
                self.status = GraphStatus::Checkpointed
            }
            GraphOutcome::Waiting(reason) => {
                self.status = match reason {
                    PauseReason::Budget { .. } => GraphStatus::WaitingBudget {
                        reason: reason.clone(),
                    },
                    _ => GraphStatus::WaitingHuman {
                        reason: reason.clone(),
                    },
                };
                em.emit(EventBody::RunPaused {
                    reason: reason.clone(),
                });
            }
            GraphOutcome::Done(_) => {
                self.status = GraphStatus::Done;
                em.emit(EventBody::RunFinished {
                    usage: self.usage(),
                });
            }
            GraphOutcome::Failed(error) => {
                self.failure.get_or_insert(Failure::Final);
                self.status = GraphStatus::Failed {
                    error: error.clone(),
                };
                em.emit(EventBody::RunFailed {
                    error: error.clone(),
                });
            }
        }
        self.seq = seq.load(Ordering::SeqCst);
        outcome
    }

    fn queue(&mut self) {
        self.steps += 1;
        self.status = GraphStatus::Queued;
    }

    fn node(&self, id: &str) -> &Node {
        self.workflow
            .node(id)
            .expect("branches sit at workflow nodes")
    }

    async fn work(&mut self, io: &dyn GraphIo, em: &Emitter<'_>) -> GraphOutcome {
        let held = self.steps <= u64::from(self.workflow.max_steps);
        trace!(
            Trace::Laws,
            "law.check",
            law = "step_stays_within_max_steps",
            held = held,
            steps = self.steps,
            max_steps = self.workflow.max_steps,
        );
        if !held {
            return GraphOutcome::Failed(format!("max_steps {} reached", self.workflow.max_steps));
        }
        match self.settle(io).await {
            Err(e) => return GraphOutcome::Failed(e),
            Ok(true) => return GraphOutcome::Done(self.state.clone()),
            Ok(false) => {}
        }
        let results = self.run_branches(io, em).await;
        let mut retry: Option<(String, Option<u64>)> = None;
        for (i, w) in results.into_iter().enumerate() {
            let node = self.node(&self.branches[i].node).clone();
            match w {
                Work::Idle | Work::Agent(StepOutcome::Continue | StepOutcome::Paused(_)) => {}
                Work::Agent(StepOutcome::Retry {
                    error,
                    retry_after_ms,
                }) => {
                    retry.get_or_insert((error, retry_after_ms));
                }
                Work::Agent(StepOutcome::Failed(e)) => {
                    return GraphOutcome::Failed(format!("node `{}`: {e}", node.id))
                }
                Work::Tool(Err(f)) => {
                    if f.transient {
                        self.failure = Some(Failure::Tool);
                    }
                    return GraphOutcome::Failed(format!("node `{}`: {}", node.id, f.message));
                }
                Work::Agent(StepOutcome::Done(output)) => {
                    let At::Agent { run, .. } = &self.branches[i].at else {
                        unreachable!("agent work comes from agent branches");
                    };
                    self.usage += run.usage();
                    self.branches[i].at = At::Left;
                    if let Err(e) = self.write(&node, output) {
                        return GraphOutcome::Failed(e);
                    }
                }
                Work::Tool(Ok(output)) => {
                    self.branches[i].at = At::Left;
                    if let Err(e) = self.write(&node, output) {
                        return GraphOutcome::Failed(e);
                    }
                }
            }
        }
        match self.settle(io).await {
            Err(e) => return GraphOutcome::Failed(e),
            Ok(true) => return GraphOutcome::Done(self.state.clone()),
            Ok(false) => {}
        }
        if let Some((error, after)) = retry {
            self.retries += 1;
            if self.retries > self.workflow.max_retries {
                self.failure = Some(Failure::Transient);
                return GraphOutcome::Failed(format!(
                    "error after {} retries: {error}",
                    self.workflow.max_retries
                ));
            }
            let backoff = BASE_RETRY_DELAY_MS.saturating_mul(1 << (self.retries - 1).min(20));
            let delay = after.unwrap_or(backoff.min(MAX_RETRY_DELAY_MS));
            let at_ms = io.agent().clock().now_ms() + i64::try_from(delay).unwrap_or(i64::MAX / 2);
            return GraphOutcome::RetryAt { at_ms, error };
        }
        self.retries = 0;
        self.wait_or_continue()
    }

    fn wait_or_continue(&self) -> GraphOutcome {
        if self.branches.iter().any(|b| self.can_work(b)) {
            return GraphOutcome::Continue;
        }
        if let Some(b) = self
            .branches
            .iter()
            .find(|b| matches!(b.at, At::Human { .. }))
        {
            let NodeKind::Human { ask, .. } = &self.node(&b.node).kind else {
                unreachable!("human branches sit at human nodes");
            };
            let node = b.node.clone();
            return GraphOutcome::Waiting(match ask {
                Ask::Approval => PauseReason::Approval { node },
                Ask::Question => PauseReason::Question { node },
            });
        }
        let reason = self
            .branches
            .iter()
            .find_map(Branch::budget_paused)
            .expect("a run that cannot work waits for a human or a budget");
        GraphOutcome::Waiting(reason.clone())
    }

    fn can_work(&self, b: &Branch) -> bool {
        match &b.at {
            At::Agent { .. } => b.budget_paused().is_none(),
            At::Enter => matches!(self.node(&b.node).kind, NodeKind::Tool { .. }),
            At::Human { .. } | At::Joined | At::Left => false,
        }
    }

    async fn run_branches(&mut self, io: &dyn GraphIo, em: &Emitter<'_>) -> Vec<Work> {
        let workers: Vec<bool> = self.branches.iter().map(|b| self.can_work(b)).collect();
        let (wf, state, run_id, step) = (&self.workflow, &self.state, self.run_id, self.steps);
        let futures =
            self.branches
                .iter_mut()
                .zip(workers)
                .enumerate()
                .map(|(i, (b, works))| async move {
                    if !works {
                        return Work::Idle;
                    }
                    let node = wf.node(&b.node).expect("branches sit at workflow nodes");
                    match (&mut b.at, &node.kind) {
                        (At::Agent { run, .. }, NodeKind::Agent { persona, .. }) => {
                            let nio = NodeIo::new(em, &node.id, persona);
                            let span = graph_step_span(&node.id, "agent", step);
                            Work::Agent(span.instrument(run.step(&nio)).await)
                        }
                        (At::Enter, NodeKind::Tool { tool }) => {
                            let key = b.call.unwrap_or(CallKey { step, index: i });
                            let ctx = ToolContext::for_call(run_id, key.step, key.index);
                            let span = graph_step_span(&node.id, "tool", step);
                            let call = call_tool(io, em, node, tool, state, ctx);
                            Work::Tool(span.instrument(call).await)
                        }
                        _ => unreachable!("only agent and tool branches work"),
                    }
                });
        join_all(futures).await
    }

    fn write(&mut self, node: &Node, value: Value) -> Result<(), String> {
        let Some(key) = &node.writes else {
            return Ok(());
        };
        let errs = self.workflow.check_value(key, &value);
        if !errs.is_empty() {
            return Err(format!(
                "node `{}`: output does not match `{key}`: {}",
                node.id,
                errs.join("; ")
            ));
        }
        self.state.insert(key.clone(), value);
        Ok(())
    }

    async fn settle(&mut self, io: &dyn GraphIo) -> Result<bool, String> {
        let passes = 2 * self.workflow.nodes.len() + 4;
        for _ in 0..passes {
            let mut changed = false;
            let mut next = Vec::with_capacity(self.branches.len());
            for b in std::mem::take(&mut self.branches) {
                if b.at != At::Left {
                    next.push(b);
                    continue;
                }
                changed = true;
                let out = self.workflow.next(&b.node);
                assert!(!out.is_empty(), "only end nodes have no out-edges");
                for e in &out {
                    graph_edge(&b.node, &e.to, e.when.as_deref());
                }
                next.extend(out.into_iter().map(|e| Branch::enter(&e.to)));
            }
            self.branches = next;
            for i in 0..self.branches.len() {
                if self.branches[i].at != At::Enter {
                    continue;
                }
                let node = self.node(&self.branches[i].node).clone();
                let fail = |m: String| format!("node `{}`: {m}", node.id);
                match &node.kind {
                    NodeKind::Tool { .. } => continue,
                    NodeKind::End => {
                        assert_eq!(self.branches.len(), 1, "branches join before an end");
                        return Ok(true);
                    }
                    NodeKind::Join => self.branches[i].at = At::Joined,
                    NodeKind::Router { condition } => {
                        let f = io.conditions().get(&condition.name).ok_or_else(|| {
                            fail(format!("condition `{}` is not registered", condition.name))
                        })?;
                        let label = f(&self.state, &condition.args).map_err(fail)?;
                        let edge = self
                            .workflow
                            .next(&node.id)
                            .into_iter()
                            .find(|e| e.when.as_deref() == Some(label.as_str()))
                            .ok_or_else(|| fail(format!("no edge for label `{label}`")))?;
                        graph_edge(&node.id, &edge.to, Some(label.as_str()));
                        self.branches[i] = Branch::enter(&edge.to);
                    }
                    NodeKind::Human { .. } => {
                        let data = reads(&self.state, &node.reads).map_err(fail)?;
                        self.branches[i].at = At::Human { data };
                    }
                    NodeKind::Agent { persona, prompt } => {
                        let data = reads(&self.state, &node.reads).map_err(fail)?;
                        let p = io.persona(persona).await.map_err(|e| fail(e.to_string()))?;
                        let (call_cap, run_cap) =
                            (p.budget.call_usd_micros, p.budget.run_usd_micros);
                        let limits = node_limits(
                            &self.limits,
                            self.usage().cost_usd_micros,
                            call_cap,
                            run_cap,
                        );
                        let input = vec![agent_input(prompt.as_deref(), data)];
                        let mut run = AgentRun::new(&p, input, limits, io.agent())
                            .await
                            .map_err(|e| fail(e.to_string()))?;
                        if let Some(spec) = io.persona_trace(persona).await {
                            run.set_trace(spec);
                        }
                        self.branches[i].at = At::Agent {
                            run: Box::new(run),
                            call_cap,
                            run_cap,
                        };
                    }
                }
                changed = true;
            }
            changed |= self.close_joins();
            if !changed {
                return Ok(false);
            }
        }
        Err(format!(
            "routing passed {passes} control nodes without work; check router loops"
        ))
    }

    fn close_joins(&mut self) -> bool {
        let ready = self.branches.iter().find_map(|b| {
            let parked = self
                .branches
                .iter()
                .filter(|o| o.node == b.node && o.at == At::Joined)
                .count();
            (b.at == At::Joined && parked == self.workflow.in_degree(&b.node))
                .then(|| b.node.clone())
        });
        let Some(join) = ready else {
            return false;
        };
        let first = self
            .branches
            .iter()
            .position(|b| b.node == join)
            .expect("the join has parked branches");
        self.branches.retain(|b| b.node != join);
        self.branches.insert(
            first.min(self.branches.len()),
            Branch {
                node: join,
                at: At::Left,
                call: None,
            },
        );
        true
    }
}

/// Opens the `graph.step` span for one branch's work at `node`.
/// Input: the node id, its kind name and the run step. Output: the span,
/// disabled when the `graphs` category is off.
fn graph_step_span(node: &str, kind: &'static str, step: u64) -> Span {
    trace_span!(Trace::Graphs, "graph.step", node = node, kind = kind, step = step)
}

/// Writes one `graph.edge` event for the edge taken from `from` to `to`.
/// Input: both node ids and the edge label, if any. Output: none.
fn graph_edge(from: &str, to: &str, label: Option<&str>) {
    trace!(Trace::Graphs, "graph.edge", from = from, to = to, label = label.unwrap_or(""));
}

fn node_limits(
    base: &Limits,
    spent_by_others: u64,
    call_cap: Option<u64>,
    run_cap: Option<u64>,
) -> Limits {
    let mut limits = base.clone();
    limits.run = limits.run.map(|r| r.saturating_sub(spent_by_others));
    limits.with_budget(&Budget {
        call_usd_micros: call_cap,
        run_usd_micros: run_cap,
    })
}

fn reads(state: &Map<String, Value>, keys: &[String]) -> Result<Map<String, Value>, String> {
    keys.iter()
        .map(|k| {
            state
                .get(k)
                .map(|v| (k.clone(), v.clone()))
                .ok_or_else(|| format!("reads `{k}`, which is not set"))
        })
        .collect()
}

fn agent_input(prompt: Option<&str>, data: Map<String, Value>) -> Message {
    let mut parts: Vec<String> = prompt.map(str::to_string).into_iter().collect();
    if !data.is_empty() {
        let json = serde_json::to_string(&Value::Object(data)).expect("state serializes");
        parts.push(fence_untrusted("workflow state", &json, false));
    }
    assert!(!parts.is_empty(), "agent nodes have a prompt or reads");
    Message::text(Role::User, parts.join("\n\n"))
}

async fn call_tool(
    io: &dyn GraphIo,
    em: &Emitter<'_>,
    node: &Node,
    tool: &str,
    state: &Map<String, Value>,
    ctx: ToolContext,
) -> Result<Value, ToolFailure> {
    let args = reads(state, &node.reads).map_err(|message| ToolFailure {
        message,
        transient: false,
    })?;
    let key = ctx.idempotency_key.clone().expect("for_call sets the key");
    let step = format!("{}/tool", node.id);
    em.emit(EventBody::StepStarted { step: step.clone(), persona: None });
    em.emit(EventBody::ToolCallStarted {
        call_id: key.clone(),
        name: tool.to_string(),
    });
    let mut span = trace_span!(
        Trace::Tools,
        "tool.call",
        name = tool,
        call_id = key.as_str(),
        node = node.id.as_str(),
        args = content(serde_json::to_string(&args).expect("tool arguments serialize")),
    );
    let call = io.agent().tools().call(tool, Value::Object(args), &ctx);
    let result = span.instrument(call).await;
    match &result {
        Ok(out) if span.is_enabled() => {
            span.record("is_error", out.is_error);
            span.record("output", content(out.content.as_str()));
        }
        Ok(_) => {}
        Err(e) => {
            span.fail("tool call failed");
            span.record("error", content(e.to_string()));
        }
    }
    drop(span);
    em.emit(EventBody::ToolCallFinished {
        call_id: key,
        is_error: !matches!(&result, Ok(o) if !o.is_error),
    });
    em.emit(EventBody::StepFinished {
        step,
        usage: Usage::default(),
    });
    match result {
        Ok(out) if !out.is_error => {
            Ok(serde_json::from_str(&out.content).unwrap_or(Value::String(out.content)))
        }
        Ok(out) => Err(ToolFailure {
            message: format!("tool `{tool}` failed: {}", out.content),
            transient: false,
        }),
        Err(e) => Err(ToolFailure {
            message: format!("tool `{tool}`: {e}"),
            transient: matches!(e, ToolError::Failed(_) | ToolError::Http(_)),
        }),
    }
}
