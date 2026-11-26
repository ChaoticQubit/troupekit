use crate::error::GraphError;
use crate::flow::{BudgetSplit, JoinMode, Flow, FlowKind, GRAPH_FORMAT_FLOW};
use crate::io::{Emitter, GraphIo, NodeIo, ToolsIo};
use crate::workflow::{Ask, Node, NodeKind, Workflow};
use futures_util::future::join_all;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use troupekit_agent::{compile_output, AgentRun, AgentTool, DelegateSettings, DelegationTools, Budget, Limits, Persona, RunState, StepOutcome};
use troupekit_core::event::{EventBody, PauseReason, Usage};
use troupekit_core::ids::RunId;
use troupekit_core::message::{Message, Role};
use std::collections::{BTreeMap, BTreeSet};
use troupekit_core::error::CoreError;
use troupekit_core::store::{BranchWrite, Checkpoint, CheckpointStore};
use troupekit_tools::{fence_untrusted, ToolContext, ToolError, ToolRegistry};

/// Version of the serialized graph run of a 0.1 workflow.
/// A run of a flow that uses a 0.5 feature writes `GRAPH_FORMAT_FLOW` (2).
/// `from_checkpoint` refuses any other, so a deploy never resumes a run it
/// cannot read.
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
        #[serde(default, skip_serializing_if = "Option::is_none")]
        share: Option<u64>,
    },
    Human {
        data: Map<String, Value>,
    },
    Flow {
        run: Box<GraphRun>,
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    item: Option<Box<Item>>,
}

/// The branch-local part of one map item branch: its map node, item index,
/// the item key and value the body reads, and the body's result once it
/// left the body. Never written to shared state.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Item {
    map: String,
    index: usize,
    key: String,
    value: Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    result: Option<Value>,
}

impl Branch {
    fn enter(node: &str) -> Branch {
        Branch {
            node: node.to_string(),
            at: At::Enter,
            call: None,
            item: None,
        }
    }

    fn of_map(&self, map: &str) -> bool {
        self.item.as_ref().is_some_and(|it| it.map == map)
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
    Flow(GraphOutcome),
    Agent(StepOutcome),
    Tool(Result<Value, ToolFailure>),
}

/// The durable result of one finished branch unit, kept as a branch write:
/// a tool's output, or an agent's stepped run and its outcome. Only
/// successes are kept; a failed or retried unit runs again, as in 0.1.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum Finished {
    Tool {
        output: Value,
    },
    Agent {
        run: Box<AgentRun>,
        outcome: AgentDone,
    },
}

/// The event `seq` range one restored attempt of a step emits from, kept
/// as a branch write of node `SEQ_NODE` before the attempt emits anything.
/// Each attempt owns `[from, from + SEQ_STRIDE)`, so a later restore starts
/// past every number an earlier attempt may have sent.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SeqRange {
    from: u64,
}

const SEQ_NODE: &str = "#seq";
const SEQ_STRIDE: u64 = 1 << 32;
const MAX_SEQ_RANGES: usize = 1 << 16;

/// A seq range `restore` chose for the next attempt of `step`, written by
/// `step_durable` at branch index `branch` before the step emits. `base`
/// is the checkpoint's seq, used again when the store does not keep it.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Reserve {
    step: u64,
    branch: u32,
    range: SeqRange,
    base: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "outcome", rename_all = "snake_case", deny_unknown_fields)]
enum AgentDone {
    Continue,
    Paused { reason: PauseReason },
    Done { output: Value },
}

impl AgentDone {
    fn of(outcome: &StepOutcome) -> Option<AgentDone> {
        match outcome {
            StepOutcome::Continue => Some(AgentDone::Continue),
            StepOutcome::Paused(reason) => Some(AgentDone::Paused {
                reason: reason.clone(),
            }),
            StepOutcome::Done(output) => Some(AgentDone::Done {
                output: output.clone(),
            }),
            StepOutcome::Retry { .. } | StepOutcome::Failed(_) => None,
        }
    }

    fn outcome(self) -> StepOutcome {
        match self {
            AgentDone::Continue => StepOutcome::Continue,
            AgentDone::Paused { reason } => StepOutcome::Paused(reason),
            AgentDone::Done { output } => StepOutcome::Done(output),
        }
    }
}

/// Branch writes loaded by `GraphRun::restore`: the step they belong to
/// and, per branch index, the node and the finished unit. Applied only by
/// the `step_durable` call that runs that same step.
#[derive(Debug, Clone, PartialEq)]
struct Marks {
    step: u64,
    done: BTreeMap<usize, (String, Finished)>,
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
    flow: Option<Box<Flow>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    split: Option<BudgetSplit>,
    #[serde(skip)]
    marks: Option<Marks>,
    #[serde(skip)]
    merged: BTreeMap<String, BTreeMap<String, String>>,
    #[serde(skip)]
    reserve: Option<Reserve>,
    #[serde(skip)]
    registries: NodeTools,
    #[serde(skip)]
    scratch: Map<String, Value>,
}

/// Per-node tool registries of agent nodes with delegates, by node id.
/// A cache rebuilt from the flow and io after restore, never state, so it
/// compares equal and shows only node ids.
#[derive(Clone, Default)]
struct NodeTools(BTreeMap<String, Arc<ToolRegistry>>);

impl PartialEq for NodeTools {
    fn eq(&self, _: &NodeTools) -> bool {
        true
    }
}

impl std::fmt::Debug for NodeTools {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_set().entries(self.0.keys()).finish()
    }
}

struct Reduced {
    key: String,
    branch: usize,
    node: String,
    value: Value,
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
        GraphRun::setup(workflow, None, input, limits, io).await
    }

    async fn setup(
        workflow: Workflow,
        flow: Option<&Flow>,
        input: Map<String, Value>,
        limits: Limits,
        io: &dyn GraphIo,
    ) -> Result<GraphRun, GraphError> {
        let setup = |id: &str, m: String| GraphError::Setup(format!("node `{id}`: {m}"));
        for n in &workflow.nodes {
            if flow.is_some_and(|f| {
                matches!(f.kind(&n.id), Some(FlowKind::Map { .. } | FlowKind::Switch { .. } | FlowKind::Flow { .. }))
            }) {
                continue;
            }
            match &n.kind {
                NodeKind::Agent { persona, .. } => {
                    let p = io
                        .persona(persona)
                        .await
                        .map_err(|e| setup(&n.id, e.to_string()))?;
                    let p = node_persona(flow, io, &n.id, p).map_err(|e| setup(&n.id, e))?;
                    let delegates = node_delegates(flow, &n.id);
                    if !delegates.is_empty() && io.agent_arc().is_none() {
                        return Err(setup(
                            &n.id,
                            "delegates need a GraphIo that shares its AgentIo (agent_arc)".into(),
                        ));
                    }
                    for d in delegates {
                        io.persona(d)
                            .await
                            .map_err(|e| setup(&n.id, format!("delegate `{d}`: {e}")))?;
                    }
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
            flow: None,
            marks: None,
            merged: BTreeMap::new(),
            reserve: None,
            registries: NodeTools::default(),
            scratch: Map::new(),
            split: None,
        })
    }

    /// Starts a run of `flow` on `input` under `limits`, with the checks of
    /// `new`. A flow with no 0.5 feature runs exactly as `new` would; any
    /// other writes `GRAPH_FORMAT_FLOW`. Returns `Setup` for a flow feature
    /// this engine does not run yet.
    pub async fn start(
        flow: Flow,
        input: Map<String, Value>,
        limits: Limits,
        io: &dyn GraphIo,
    ) -> Result<GraphRun, GraphError> {
        if let Some(feature) = unsupported(&flow) {
            return Err(GraphError::Setup(format!(
                "flow `{}`: {feature} is not run by this engine yet",
                flow.name()
            )));
        }
        if let Some(m) = untracked_misuse(&flow) {
            return Err(GraphError::Setup(format!("flow `{}`: {m}", flow.name())));
        }
        let mut path = vec![flow.name().to_string()];
        check_children(&flow, io, &mut path).await?;
        GraphRun::start_child(flow, input, limits, io).await
    }

    async fn start_child(
        flow: Flow,
        input: Map<String, Value>,
        limits: Limits,
        io: &dyn GraphIo,
    ) -> Result<GraphRun, GraphError> {
        let mut run = GraphRun::setup(flow.workflow().clone(), Some(&flow), input, limits, io).await?;
        for key in flow.untracked() {
            if let Some(v) = run.state.remove(key) {
                run.scratch.insert(key.clone(), v);
            }
        }
        if !flow.is_v1() {
            run.format = GRAPH_FORMAT_FLOW;
            run.flow = Some(Box::new(flow));
        }
        Ok(run)
    }

    /// Returns the flow the run started with: the stored flow of a run
    /// started from a flow with 0.5 features, else its workflow as a flow.
    /// It never changes.
    pub fn flow(&self) -> Flow {
        match &self.flow {
            Some(flow) => (**flow).clone(),
            None => Flow::from(self.workflow.clone()),
        }
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
        let expected = if run.flow.is_some() {
            GRAPH_FORMAT_FLOW
        } else {
            GRAPH_FORMAT
        };
        if run.format != expected {
            return Err(bad(format!(
                "graph format {} is not {expected}",
                run.format
            )));
        }
        if run.run_id != cp.run_id || run.steps != cp.step {
            return Err(bad("checkpoint header does not match its state".into()));
        }
        if run.status == GraphStatus::Running {
            return Err(bad("a checkpoint never holds a running step".into()));
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
            At::Flow { run } => acc + run.usage(),
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
        if self.even() {
            self.top_up(true).map_err(GraphError::Resume)?;
            self.queue();
            return Ok(());
        }
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
                ..
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

    /// Sets how concurrent agent branches share the remaining run budget,
    /// over the flow's `[budget] split`. `BudgetSplit::Shared` restores the
    /// 0.1 split, also on a v1 workflow run. Returns the run.
    pub fn with_budget_split(mut self, split: BudgetSplit) -> GraphRun {
        self.split = Some(split);
        self
    }

    fn even(&self) -> bool {
        let split = self.split.or(self.flow.as_ref().map(|f| f.budget_split()));
        self.limits.run.is_some() && split.unwrap_or_default() == BudgetSplit::Even
    }

    fn unspent(&self, skip_paused: bool) -> u64 {
        self.branches
            .iter()
            .map(|b| match &b.at {
                At::Agent { run, share: Some(s), .. } if !(skip_paused && b.budget_paused().is_some()) => {
                    s.saturating_sub(run.usage().cost_usd_micros)
                }
                At::Flow { run } => run.limits.run.map_or(0, |s| s.saturating_sub(run.usage().cost_usd_micros)),
                _ => 0,
            })
            .sum()
    }

    fn free(&self, skip_paused: bool) -> u64 {
        let cap = self.limits.run.expect("an even split has a run cap");
        cap.saturating_sub(self.usage().cost_usd_micros)
            .saturating_sub(self.unspent(skip_paused))
    }

    fn share(&self, entering: &mut usize) -> Option<u64> {
        if !self.even() {
            return None;
        }
        assert!(*entering > 0, "every spending branch that enters was counted");
        let n = u64::try_from(*entering).expect("branch count fits in u64");
        *entering -= 1;
        Some(self.free(false) / n)
    }

    fn spends(&self, node: &str) -> bool {
        matches!(self.node(node).kind, NodeKind::Agent { .. })
            || matches!(self.flow_kind(node), Some(FlowKind::Flow { .. }))
    }

    fn top_up(&mut self, raised: bool) -> Result<(), String> {
        let paused: Vec<usize> = (0..self.branches.len())
            .filter(|&i| match self.branches[i].budget_paused() {
                Some(PauseReason::Budget { limit }) => raised || limit == "run",
                _ => false,
            })
            .collect();
        if paused.is_empty() {
            return Ok(());
        }
        let free = self.free(true);
        let each = free / u64::try_from(paused.len()).expect("branch count fits in u64");
        let base = self.limits.clone();
        let mut granted = 0u64;
        for i in paused {
            let At::Agent { run, call_cap, run_cap, share } = &mut self.branches[i].at else {
                unreachable!("budget pauses happen in agent branches");
            };
            let grant = run.usage().cost_usd_micros.saturating_add(each);
            let grant = run_cap.map_or(grant, |c| c.min(grant));
            let old = share.unwrap_or(0);
            granted = granted.saturating_add(grant.saturating_sub(run.usage().cost_usd_micros));
            *share = Some(grant);
            if raised || grant > old {
                let mut limits = node_limits(&base, 0, *call_cap, *run_cap);
                limits.run = Some(grant);
                run.resume(limits).map_err(|e| e.to_string())?;
            }
        }
        assert!(granted <= free, "paused branches never get more than the free budget");
        Ok(())
    }

    /// Runs one step: one unit of work (one agent step or one tool call)
    /// for every branch that can work, concurrently, with control nodes
    /// resolved before and after. Waiting and final runs change nothing.
    pub async fn step(&mut self, io: &dyn GraphIo) -> GraphOutcome {
        self.marks = None;
        self.step_with(io, None)
            .await
            .expect("a step without a store or branch writes never errors")
    }

    /// Runs one step as `step` does, keeping each finished branch's result
    /// in `store` before its siblings finish; on a run from `restore`, kept
    /// results apply instead of running again. On `Err` (store error or a
    /// write that does not fit) the caller drops this run and reloads with
    /// `restore`, never checkpointing it or using `from_checkpoint`.
    pub async fn step_durable(
        &mut self,
        io: &dyn GraphIo,
        store: &dyn CheckpointStore,
    ) -> Result<GraphOutcome, GraphError> {
        self.step_with(io, Some(store)).await
    }

    /// Restores run `run_id` from the latest checkpoint and the branch writes
    /// of the step after it, which the next `step_durable` applies. With a
    /// store that keeps branch writes, event `seq` jumps past every earlier
    /// attempt's range (gaps, never repeats); with one that keeps whole steps
    /// only, or with plain `step`, numbering is 0.1's and a re-run step may repeat.
    pub async fn restore(store: &dyn CheckpointStore, run_id: RunId) -> Result<GraphRun, GraphError> {
        let cp = store
            .latest(run_id)
            .await?
            .ok_or_else(|| GraphError::Resume(format!("run {run_id} has no checkpoint")))?;
        let mut run = GraphRun::from_checkpoint(&cp)?;
        let step = cp.step + 1;
        let mut done = BTreeMap::new();
        let mut ranges: Vec<u64> = Vec::new();
        for w in store.branch_writes(run_id, step).await? {
            let bad = |m: String| GraphError::Checkpoint(format!("branch write {} of step {step}: {m}", w.branch()));
            let index = usize::try_from(w.branch()).map_err(|e| bad(e.to_string()))?;
            if w.node() == SEQ_NODE {
                let range: SeqRange = serde_json::from_value(w.write().clone()).map_err(|e| bad(e.to_string()))?;
                ranges.push(range.from);
                continue;
            }
            let finished: Finished = serde_json::from_value(w.write().clone()).map_err(|e| bad(e.to_string()))?;
            if done.insert(index, (w.node().to_string(), finished)).is_some() {
                return Err(bad("the branch has two writes".into()));
            }
        }
        if ranges.len() >= MAX_SEQ_RANGES {
            return Err(GraphError::Checkpoint(format!("step {step} was restored {} times", ranges.len())));
        }
        let floor = ranges.iter().copied().fold(run.seq, u64::max);
        let from = floor
            .checked_add(SEQ_STRIDE)
            .ok_or_else(|| GraphError::Checkpoint(format!("event seq {floor} of step {step} is exhausted")))?;
        let branch = u32::MAX - u32::try_from(ranges.len()).expect("bounded by MAX_SEQ_RANGES");
        let base = run.seq;
        run.seq = from;
        run.reserve = Some(Reserve {
            step,
            branch,
            range: SeqRange { from },
            base,
        });
        if !done.is_empty() {
            run.marks = Some(Marks { step, done });
        }
        Ok(run)
    }

    async fn step_with(
        &mut self,
        io: &dyn GraphIo,
        store: Option<&dyn CheckpointStore>,
    ) -> Result<GraphOutcome, GraphError> {
        let marks = self.marks.take();
        let reserve = self.reserve.take();
        match &self.status {
            GraphStatus::Done => return Ok(GraphOutcome::Done(self.state.clone())),
            GraphStatus::Failed { error } => return Ok(GraphOutcome::Failed(error.clone())),
            GraphStatus::WaitingHuman { reason } | GraphStatus::WaitingBudget { reason } => {
                return Ok(GraphOutcome::Waiting(reason.clone()))
            }
            GraphStatus::Running => unreachable!("a step never starts inside another"),
            GraphStatus::Queued | GraphStatus::Checkpointed => {}
        }
        match (reserve, store) {
            (Some(r), Some(store)) if r.step == self.steps + 1 => {
                assert_eq!(self.seq, r.range.from, "a reserved attempt emits from its range");
                let value = serde_json::to_value(r.range).expect("a seq range serializes");
                let kept = store
                    .save_branch(self.run_id, r.step, BranchWrite::new(r.branch, SEQ_NODE, value))
                    .await?;
                if !kept {
                    self.seq = r.base;
                }
            }
            (Some(r), None) if r.step == self.steps + 1 => self.seq = r.base,
            _ => {}
        }
        self.steps += 1;
        self.status = GraphStatus::Running;
        self.merged.clear();
        let first_seq = self.seq;
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
        let marks = marks.filter(|m| m.step == self.steps);
        let outcome = self.work(io, &em, store, marks).await?;
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
        assert!(self.seq - first_seq < SEQ_STRIDE, "one attempt of a step stays inside its seq range");
        Ok(outcome)
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

    async fn work(
        &mut self,
        io: &dyn GraphIo,
        em: &Emitter<'_>,
        store: Option<&dyn CheckpointStore>,
        marks: Option<Marks>,
    ) -> Result<GraphOutcome, GraphError> {
        if self.steps > u64::from(self.workflow.max_steps) {
            return Ok(GraphOutcome::Failed(format!("max_steps {} reached", self.workflow.max_steps)));
        }
        match self.settle(io).await {
            Err(e) => return Ok(GraphOutcome::Failed(e)),
            Ok(true) => return Ok(GraphOutcome::Done(self.state.clone())),
            Ok(false) => {}
        }
        if let Err(e) = self.prepare_registries(io).await {
            return Ok(GraphOutcome::Failed(e));
        }
        let done = match marks {
            Some(m) => self.check_marks(m.done)?,
            None => BTreeMap::new(),
        };
        let (results, failed_save) = self.run_branches(io, em, store, done).await;
        if let Some(e) = failed_save {
            return Err(GraphError::Store(e));
        }
        Ok(self.apply(io, results).await)
    }

    fn check_marks(
        &self,
        done: BTreeMap<usize, (String, Finished)>,
    ) -> Result<BTreeMap<usize, Finished>, GraphError> {
        let mut out = BTreeMap::new();
        let workers = self.workers();
        for (i, (node, finished)) in done {
            let bad = |m: String| GraphError::Checkpoint(format!("branch write {i} of step {}: {m}", self.steps));
            let Some(b) = self.branches.get(i) else {
                return Err(bad(format!("the step has {} branches", self.branches.len())));
            };
            if b.node != node {
                return Err(bad(format!("it is for node `{node}`, but the branch is at `{}`", b.node)));
            }
            if !workers[i] {
                return Err(bad(format!("the branch at `{node}` has no work this step")));
            }
            match (&finished, &b.at) {
                (Finished::Tool { .. }, At::Enter) => {}
                (Finished::Agent { run, .. }, At::Agent { .. }) => {
                    AgentRun::from_checkpoint(&run.checkpoint()).map_err(|e| bad(e.to_string()))?;
                }
                _ => return Err(bad(format!("its kind does not fit the work at `{node}`"))),
            }
            out.insert(i, finished);
        }
        Ok(out)
    }

    async fn apply(&mut self, io: &dyn GraphIo, results: Vec<Work>) -> GraphOutcome {
        let mut retry: Option<(String, Option<u64>)> = None;
        let mut reduced: Vec<Reduced> = Vec::new();
        for (i, w) in results.into_iter().enumerate() {
            let node = self.node(&self.branches[i].node).clone();
            match w {
                Work::Idle
                | Work::Agent(StepOutcome::Continue | StepOutcome::Paused(_))
                | Work::Flow(GraphOutcome::Continue | GraphOutcome::Waiting(_)) => {}
                Work::Flow(GraphOutcome::Failed(e) | GraphOutcome::RetryAt { error: e, .. }) => {
                    return self.fail_after(reduced, format!("node `{}`: subflow: {e}", node.id));
                }
                Work::Flow(GraphOutcome::Done(child)) => {
                    let At::Flow { run } = &self.branches[i].at else {
                        unreachable!("flow work comes from flow branches");
                    };
                    self.usage += run.usage();
                    let Some(FlowKind::Flow { returns, .. }) = self.flow_kind(&node.id) else {
                        unreachable!("flow branches sit at flow nodes");
                    };
                    let value = child.get(returns).cloned().unwrap_or(Value::Null);
                    self.branches[i].at = At::Left;
                    if let Err(e) = self.write(&node, i, value, &mut reduced) {
                        return GraphOutcome::Failed(e);
                    }
                }
                Work::Agent(StepOutcome::Retry {
                    error,
                    retry_after_ms,
                }) => {
                    retry.get_or_insert((error, retry_after_ms));
                }
                Work::Agent(StepOutcome::Failed(e)) => {
                    return self.fail_after(reduced, format!("node `{}`: {e}", node.id));
                }
                Work::Tool(Err(f)) => {
                    if f.transient {
                        self.failure = Some(Failure::Tool);
                    }
                    return self.fail_after(reduced, format!("node `{}`: {}", node.id, f.message));
                }
                Work::Agent(StepOutcome::Done(output)) => {
                    let At::Agent { run, .. } = &self.branches[i].at else {
                        unreachable!("agent work comes from agent branches");
                    };
                    self.usage += run.usage();
                    self.branches[i].at = At::Left;
                    let target = match self.goto(&node.id) {
                        None => None,
                        Some(targets) => match goto_choice(&output, targets) {
                            Ok(t) => Some(t),
                            Err(e) => return self.fail_after(reduced, format!("node `{}`: {e}", node.id)),
                        },
                    };
                    if let Err(e) = self.write(&node, i, output, &mut reduced) {
                        return GraphOutcome::Failed(e);
                    }
                    if let Some(t) = target {
                        self.branches[i] = Branch::enter(&t);
                    }
                }
                Work::Tool(Ok(output)) => {
                    self.branches[i].at = At::Left;
                    if let Err(e) = self.write(&node, i, output, &mut reduced) {
                        return GraphOutcome::Failed(e);
                    }
                }
            }
        }
        if let Err(e) = self.reduce(reduced) {
            return GraphOutcome::Failed(e);
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

    fn fail_after(&mut self, reduced: Vec<Reduced>, error: String) -> GraphOutcome {
        match self.reduce(reduced) {
            Ok(()) => GraphOutcome::Failed(error),
            Err(e) => {
                self.failure = None;
                GraphOutcome::Failed(e)
            }
        }
    }

    fn wait_or_continue(&self) -> GraphOutcome {
        if self.workers().into_iter().any(|w| w) {
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
        let child = self.branches.iter().find_map(|b| match &b.at {
            At::Flow { run } => match &run.status {
                GraphStatus::WaitingHuman { reason } | GraphStatus::WaitingBudget { reason } => Some(reason),
                _ => None,
            },
            _ => None,
        });
        let reason = child
            .or_else(|| self.branches.iter().find_map(Branch::budget_paused))
            .expect("a run that cannot work waits for a human, a budget or a waiting subflow");
        GraphOutcome::Waiting(reason.clone())
    }

    fn has_work(&self, b: &Branch) -> bool {
        match &b.at {
            At::Agent { .. } => b.budget_paused().is_none(),
            At::Enter => matches!(self.node(&b.node).kind, NodeKind::Tool { .. }),
            At::Flow { run } => matches!(run.status, GraphStatus::Queued | GraphStatus::Checkpointed),
            At::Human { .. } | At::Joined | At::Left => false,
        }
    }

    fn workers(&self) -> Vec<bool> {
        let window = self.openings();
        self.branches
            .iter()
            .zip(window)
            .map(|(b, open)| open && self.has_work(b))
            .collect()
    }

    fn flow_kind(&self, id: &str) -> Option<&FlowKind> {
        self.flow.as_ref()?.kind(id)
    }

    fn goto(&self, id: &str) -> Option<&[String]> {
        match self.flow_kind(id) {
            Some(FlowKind::Agent { goto, .. }) if !goto.is_empty() => Some(goto),
            _ => None,
        }
    }

    fn switch_to(&self, id: &str, on: &str) -> Result<String, String> {
        let view = self.view();
        let label = match view.get(on) {
            Some(Value::String(v)) => v.clone(),
            Some(Value::Bool(v)) => v.to_string(),
            Some(Value::Null) | None => return Err(format!("reads `{on}`, which is not set")),
            Some(other) => return Err(format!("`{on}` is not a string, enum or bool: {other}")),
        };
        self.workflow
            .next(id)
            .into_iter()
            .find(|e| e.when.as_deref() == Some(label.as_str()))
            .map(|e| e.to.clone())
            .ok_or_else(|| format!("no edge for value `{label}` of `{on}`"))
    }

    fn deferred(&self, id: &str) -> bool {
        match self.flow_kind(id) {
            Some(FlowKind::Agent { defer, .. } | FlowKind::Tool { defer, .. } | FlowKind::Flow { defer, .. }) => *defer,
            _ => false,
        }
    }

    fn is_any(&self, id: &str) -> bool {
        matches!(self.flow_kind(id), Some(FlowKind::Join { mode: JoinMode::Any }))
    }

    fn openings(&self) -> Vec<bool> {
        let window = self.windows();
        if !self.branches.iter().any(|b| self.deferred(&b.node)) {
            return window;
        }
        let others_parked = self
            .branches
            .iter()
            .all(|b| self.deferred(&b.node) || b.at == At::Joined);
        self.branches
            .iter()
            .zip(window)
            .map(|(b, open)| open && (others_parked || !self.deferred(&b.node)))
            .collect()
    }

    fn fork_outs(&self, id: &str) -> Option<Vec<String>> {
        let outs = self.workflow.next(id);
        (outs.len() >= 2 && outs.iter().all(|e| e.when.is_none()))
            .then(|| outs.iter().map(|e| e.to.clone()).collect())
    }

    fn join_of(&self, fork: &str, depth: usize) -> String {
        assert!(depth <= self.workflow.nodes.len(), "fan-outs nest at most once per node");
        let outs = self.fork_outs(fork).expect("join_of walks from a fork");
        let mut cur = outs[0].clone();
        for _ in 0..=self.workflow.nodes.len() {
            if matches!(self.node(&cur).kind, NodeKind::Join) {
                return cur;
            }
            let from = if self.fork_outs(&cur).is_some() {
                self.join_of(&cur, depth + 1)
            } else {
                cur.clone()
            };
            let next = self.workflow.next(&from);
            assert_eq!(next.len(), 1, "a fan-out branch leads on by one edge");
            cur = next[0].to.clone();
        }
        unreachable!("a validated fan-out branch ends at its join");
    }

    fn region(&self, fork: &str, join: &str, out: &mut BTreeSet<String>, depth: usize) {
        assert!(depth <= self.workflow.nodes.len(), "fan-outs nest at most once per node");
        for first in self.fork_outs(fork).expect("region walks from a fork") {
            let mut cur = first;
            for _ in 0..=self.workflow.nodes.len() {
                if cur == join {
                    break;
                }
                out.insert(cur.clone());
                if let Some(FlowKind::Map { body, .. }) = self.flow_kind(&cur) {
                    out.insert(body.clone());
                }
                let from = if self.fork_outs(&cur).is_some() {
                    let inner = self.join_of(&cur, depth + 1);
                    self.region(&cur, &inner, out, depth + 1);
                    out.insert(inner.clone());
                    inner
                } else {
                    cur.clone()
                };
                let next = self.workflow.next(&from);
                assert_eq!(next.len(), 1, "a fan-out branch leads on by one edge");
                cur = next[0].to.clone();
            }
            assert_eq!(cur, join, "every branch of the fan-out ends at its join");
        }
    }

    fn close_any(&mut self) -> bool {
        let Some(join) = self
            .branches
            .iter()
            .find(|b| b.at == At::Joined && self.is_any(&b.node))
            .map(|b| b.node.clone())
        else {
            return false;
        };
        let fork = self
            .workflow
            .nodes
            .iter()
            .map(|n| n.id.clone())
            .find(|id| self.fork_outs(id).is_some() && self.join_of(id, 0) == join)
            .expect("every join closes one fan-out");
        let mut region = BTreeSet::new();
        self.region(&fork, &join, &mut region, 0);
        region.insert(join.clone());
        let before = self.usage();
        let mut first = None;
        let mut kept = Vec::with_capacity(self.branches.len());
        for b in std::mem::take(&mut self.branches) {
            if !region.contains(&b.node) {
                kept.push(b);
                continue;
            }
            first.get_or_insert(kept.len());
            match &b.at {
                At::Agent { run, .. } => self.usage += run.usage(),
                At::Flow { run } => self.usage += run.usage(),
                _ => {}
            }
        }
        let at = first.expect("the join has a parked branch");
        kept.insert(
            at,
            Branch {
                node: join,
                at: At::Left,
                call: None,
                item: None,
            },
        );
        self.branches = kept;
        let after = self.usage();
        assert!(
            after.cost_usd_micros >= before.cost_usd_micros && after.input_tokens >= before.input_tokens,
            "dropped branches keep their usage"
        );
        true
    }

    fn windows(&self) -> Vec<bool> {
        windows(&self.branches, |map| match self.flow_kind(map) {
            Some(FlowKind::Map { concurrency, .. }) => *concurrency as usize,
            _ => unreachable!("item branches belong to map nodes"),
        })
    }

    fn expand_map(&mut self, i: usize) -> Result<(), String> {
        let id = self.branches[i].node.clone();
        let Some(FlowKind::Map {
            over,
            item,
            body,
            max_items,
            ..
        }) = self.flow_kind(&id).cloned()
        else {
            unreachable!("only map nodes expand");
        };
        let fail = |m: String| format!("node `{id}`: {m}");
        let list = match self.state.get(&over) {
            Some(Value::Array(list)) => list.clone(),
            Some(Value::Null) | None => return Err(fail(format!("reads `{over}`, which is not set"))),
            Some(other) => return Err(fail(format!("`{over}` is not a list: {other}"))),
        };
        if list.len() > max_items as usize {
            return Err(fail(format!(
                "{} items in `{over}` are over the cap of {max_items}",
                list.len()
            )));
        }
        if list.is_empty() {
            let node = self.node(&id).clone();
            self.write_map(&node, Vec::new())?;
            self.branches[i].at = At::Left;
            return Ok(());
        }
        for (index, value) in list.iter().enumerate() {
            let errs = self.workflow.check_value(&item, value);
            if !errs.is_empty() {
                return Err(fail(format!("item {index} does not match `{item}`: {}", errs.join("; "))));
            }
        }
        let items: Vec<Branch> = list
            .into_iter()
            .enumerate()
            .map(|(index, value)| Branch {
                node: body.clone(),
                at: At::Enter,
                call: None,
                item: Some(Box::new(Item {
                    map: id.clone(),
                    index,
                    key: item.clone(),
                    value,
                    result: None,
                })),
            })
            .collect();
        self.branches.splice(i..=i, items);
        Ok(())
    }

    fn write_map(&mut self, node: &Node, results: Vec<(usize, Value)>) -> Result<(), String> {
        let key = node.writes.clone().expect("map nodes write");
        let value = match self.reducer(&key) {
            Some(_) if results.is_empty() => return Ok(()),
            Some(reducer) => {
                let fail = |m: String| format!("node `{}`: reducing `{key}`: {m}", node.id);
                let value = reducer.reduce(self.state.get(&key), &results).map_err(fail)?;
                let writes: Vec<(&str, &Value)> = results.iter().map(|(_, v)| (node.id.as_str(), v)).collect();
                self.note_merge(&key, &writes).map_err(fail)?;
                value
            }
            None => Value::Array(results.into_iter().map(|(_, v)| v).collect()),
        };
        let errs = self.workflow.check_value(&key, &value);
        if !errs.is_empty() {
            return Err(format!(
                "node `{}`: result does not match `{key}`: {}",
                node.id,
                errs.join("; ")
            ));
        }
        self.state.insert(key, value);
        Ok(())
    }

    fn collect_maps(&mut self) -> Result<bool, String> {
        let Some((map, first)) = ready_map(&self.branches) else {
            return Ok(false);
        };
        let mut results: Vec<(usize, Value)> = self
            .branches
            .iter()
            .filter_map(|b| b.item.as_ref().filter(|it| it.map == map))
            .map(|it| (it.index, it.result.clone().expect("a finished item holds its result")))
            .collect();
        results.sort_by_key(|(index, _)| *index);
        assert!(
            results.iter().enumerate().all(|(n, (index, _))| n == *index),
            "a map collects every item exactly once"
        );
        let node = self.node(&map).clone();
        self.write_map(&node, results)?;
        self.branches.retain(|b| !b.of_map(&map));
        let at = first.min(self.branches.len());
        self.branches.insert(
            at,
            Branch {
                node: map,
                at: At::Left,
                call: None,
                item: None,
            },
        );
        Ok(true)
    }

    async fn prepare_registries(&mut self, io: &dyn GraphIo) -> Result<(), String> {
        let missing: Vec<String> = self
            .branches
            .iter()
            .filter(|b| matches!(b.at, At::Agent { .. }))
            .map(|b| b.node.clone())
            .filter(|id| !self.registries.0.contains_key(id))
            .filter(|id| !node_delegates(self.flow.as_deref(), id).is_empty())
            .collect();
        for id in missing {
            let NodeKind::Agent { persona, .. } = &self.node(&id).kind else {
                unreachable!("agent branches sit at agent nodes");
            };
            let fail = |m: String| format!("node `{id}`: {m}");
            let p = io.persona(persona).await.map_err(|e| fail(e.to_string()))?;
            let delegates = node_delegates(self.flow.as_deref(), &id).to_vec();
            let (reg, _) = node_registry(io, &delegates, &p.tools).await.map_err(fail)?;
            self.registries.0.insert(id, Arc::new(reg));
        }
        Ok(())
    }

    async fn run_branches(
        &mut self,
        io: &dyn GraphIo,
        em: &Emitter<'_>,
        store: Option<&dyn CheckpointStore>,
        mut done: BTreeMap<usize, Finished>,
    ) -> (Vec<Work>, Option<CoreError>) {
        let workers = self.workers();
        let merged = (!self.scratch.is_empty()).then(|| {
            let mut m = self.state.clone();
            m.extend(self.scratch.iter().map(|(k, v)| (k.clone(), v.clone())));
            m
        });
        let (wf, run_id, step) = (&self.workflow, self.run_id, self.steps);
        let state = merged.as_ref().unwrap_or(&self.state);
        let regs = &self.registries.0;
        let untracked: &[String] = self.flow.as_ref().map_or(&[], |f| f.untracked());
        let futures = self
            .branches
            .iter_mut()
            .zip(workers)
            .enumerate()
            .map(|(i, (b, works))| {
                let kept = done.remove(&i);
                async move {
                    if let Some(finished) = kept {
                        return (apply_finished(b, finished), None);
                    }
                    if !works {
                        return (Work::Idle, None);
                    }
                    let node = wf.node(&b.node).expect("branches sit at workflow nodes");
                    if let At::Flow { run } = &mut b.at {
                        let out = Box::pin(run.step(io)).await;
                        return (Work::Flow(out), None);
                    }
                    let (work, finished) = match (&mut b.at, &node.kind) {
                        (At::Agent { run, .. }, NodeKind::Agent { persona, .. }) => {
                            let nio = NodeIo::new(em, &node.id, persona).with_tools(regs.get(&node.id).map(|r| r.as_ref()));
                            let outcome = run.step(&nio).await;
                            let finished = AgentDone::of(&outcome).map(|outcome| Finished::Agent {
                                run: run.clone(),
                                outcome,
                            });
                            (Work::Agent(outcome), finished)
                        }
                        (At::Enter, NodeKind::Tool { tool }) => {
                            let key = b.call.unwrap_or(CallKey { step, index: i });
                            let ctx = ToolContext::for_call(run_id, key.step, key.index);
                            let result = call_tool(io, em, wf, node, tool, state, b.item.as_deref(), ctx).await;
                            let finished = result.as_ref().ok().map(|output| Finished::Tool {
                                output: output.clone(),
                            });
                            (Work::Tool(result), finished)
                        }
                        _ => unreachable!("only agent and tool branches work"),
                    };
                    let secret = node.writes.as_ref().is_some_and(|w| untracked.contains(w));
                    let (Some(store), Some(finished), false) = (store, finished, secret) else {
                        return (work, None);
                    };
                    let branch = u32::try_from(i).expect("a step has far fewer than 2^32 branches");
                    let value = serde_json::to_value(&finished).expect("a finished unit serializes");
                    let saved = store.save_branch(run_id, step, BranchWrite::new(branch, &node.id, value)).await;
                    (work, saved.err())
                }
            });
        let results = join_all(futures).await;
        let mut works = Vec::with_capacity(results.len());
        let mut failed = None;
        for (work, error) in results {
            works.push(work);
            failed = failed.or(error);
        }
        assert!(done.is_empty(), "every kept write belongs to a branch of the step");
        (works, failed)
    }

    fn write(
        &mut self,
        node: &Node,
        branch: usize,
        value: Value,
        reduced: &mut Vec<Reduced>,
    ) -> Result<(), String> {
        let Some(key) = &node.writes else {
            return Ok(());
        };
        if self.branches[branch].item.is_some() {
            let errs = self.workflow.check_value(key, &value);
            if !errs.is_empty() {
                return Err(format!(
                    "node `{}`: output does not match `{key}`: {}",
                    node.id,
                    errs.join("; ")
                ));
            }
            let item = self.branches[branch].item.as_mut().expect("checked above");
            item.result = Some(value);
            return Ok(());
        }
        if self.reducer(key).is_some() {
            reduced.push(Reduced {
                key: key.clone(),
                branch,
                node: node.id.clone(),
                value,
            });
            return Ok(());
        }
        let errs = self.workflow.check_value(key, &value);
        if !errs.is_empty() {
            return Err(format!(
                "node `{}`: output does not match `{key}`: {}",
                node.id,
                errs.join("; ")
            ));
        }
        if self.untracked(key) {
            self.scratch.insert(key.clone(), value);
        } else {
            self.state.insert(key.clone(), value);
        }
        Ok(())
    }

    fn untracked(&self, key: &str) -> bool {
        self.flow.as_ref().is_some_and(|f| f.untracked().iter().any(|k| k == key))
    }

    fn view(&self) -> std::borrow::Cow<'_, Map<String, Value>> {
        if self.scratch.is_empty() {
            return std::borrow::Cow::Borrowed(&self.state);
        }
        let mut merged = self.state.clone();
        merged.extend(self.scratch.iter().map(|(k, v)| (k.clone(), v.clone())));
        std::borrow::Cow::Owned(merged)
    }

    fn reducer(&self, key: &str) -> Option<crate::flow::Reducer> {
        self.flow.as_ref().and_then(|f| f.reducer(key))
    }

    fn note_merge(&mut self, key: &str, writes: &[(&str, &Value)]) -> Result<(), String> {
        if !matches!(self.reducer(key), Some(crate::flow::Reducer::Merge)) {
            return Ok(());
        }
        let seen = self.merged.entry(key.to_string()).or_default();
        let mut fresh = Vec::new();
        for (node, value) in writes {
            for field in value.as_object().into_iter().flat_map(|o| o.keys()) {
                if let Some(other) = seen.get(field) {
                    return Err(format!(
                        "`merge`: nodes `{other}` and `{node}` both write `{field}` in one step; use `last` to keep the later write"
                    ));
                }
                fresh.push((field.clone(), node.to_string()));
            }
        }
        seen.extend(fresh);
        Ok(())
    }

    fn reduce(&mut self, reduced: Vec<Reduced>) -> Result<(), String> {
        assert!(
            reduced.windows(2).all(|w| w[0].branch < w[1].branch),
            "results are applied in ascending branch index"
        );
        let mut keys: Vec<String> = Vec::new();
        for r in &reduced {
            if !keys.contains(&r.key) {
                keys.push(r.key.clone());
            }
        }
        for key in keys {
            let reducer = self.reducer(&key).expect("only reducer keys are deferred");
            let group: Vec<&Reduced> = reduced.iter().filter(|r| r.key == key).collect();
            let last = group.last().expect("a reduced key has a write");
            let writes: Vec<(usize, Value)> = group.iter().map(|r| (r.branch, r.value.clone())).collect();
            let fail = |m: String| format!("node `{}`: reducing `{key}`: {m}", last.node);
            let value = reducer.reduce(self.state.get(&key), &writes).map_err(fail)?;
            let noted: Vec<(&str, &Value)> = group.iter().map(|r| (r.node.as_str(), &r.value)).collect();
            self.note_merge(&key, &noted).map_err(fail)?;
            let errs = self.workflow.check_value(&key, &value);
            if !errs.is_empty() {
                return Err(fail(format!("result does not match `{key}`: {}", errs.join("; "))));
            }
            self.state.insert(key, value);
        }
        Ok(())
    }

    async fn settle(&mut self, io: &dyn GraphIo) -> Result<bool, String> {
        let passes = 3 * self.workflow.nodes.len() + 8;
        for _ in 0..passes {
            let mut changed = false;
            let mut next = Vec::with_capacity(self.branches.len());
            for b in std::mem::take(&mut self.branches) {
                if b.at != At::Left || b.item.is_some() {
                    next.push(b);
                    continue;
                }
                changed = true;
                assert!(self.goto(&b.node).is_none(), "goto agents route on their choice, never by leaving");
                let out = self.workflow.next(&b.node);
                assert!(!out.is_empty(), "only end nodes have no out-edges");
                next.extend(out.into_iter().map(|e| Branch::enter(&e.to)));
            }
            self.branches = next;
            let window = self.openings();
            let mut entering = (0..self.branches.len())
                .filter(|&i| window[i] && self.branches[i].at == At::Enter && self.spends(&self.branches[i].node))
                .count();
            for i in 0..self.branches.len() {
                if self.branches[i].at != At::Enter {
                    continue;
                }
                let node = self.node(&self.branches[i].node).clone();
                let fail = |m: String| format!("node `{}`: {m}", node.id);
                if matches!(self.flow_kind(&node.id), Some(FlowKind::Map { .. })) {
                    self.expand_map(i)?;
                    changed = true;
                    break;
                }
                if let Some(FlowKind::Flow { flow: name, .. }) = self.flow_kind(&node.id) {
                    if !window[i] {
                        continue;
                    }
                    let name = name.clone();
                    let item = self.branches[i].item.clone();
                    let mut input = reads(&self.workflow, &self.state, &node.reads, item.as_deref()).map_err(fail)?;
                    input.retain(|_, v| !v.is_null());
                    let child = io.flow(&name).await.map_err(|e| fail(e.to_string()))?;
                    let limits = match self.share(&mut entering) {
                        Some(s) => Limits {
                            run: Some(s),
                            ..self.limits.clone()
                        },
                        None => node_limits(&self.limits, self.usage().cost_usd_micros, None, None),
                    };
                    let mut child = GraphRun::start_child(child, input, limits, io).await.map_err(|e| fail(e.to_string()))?;
                    child.run_id = RunId::derived(self.run_id, &format!("{}:{}:{i}", node.id, self.steps));
                    self.branches[i].at = At::Flow { run: Box::new(child) };
                    changed = true;
                    continue;
                }
                if let Some(FlowKind::Switch { on }) = self.flow_kind(&node.id) {
                    let to = self.switch_to(&node.id, on).map_err(fail)?;
                    self.branches[i] = Branch::enter(&to);
                    changed = true;
                    continue;
                }
                let item = self.branches[i].item.clone();
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
                        let label = f(&self.view(), &condition.args).map_err(fail)?;
                        let edge = self
                            .workflow
                            .next(&node.id)
                            .into_iter()
                            .find(|e| e.when.as_deref() == Some(label.as_str()))
                            .ok_or_else(|| fail(format!("no edge for label `{label}`")))?;
                        self.branches[i] = Branch::enter(&edge.to);
                    }
                    NodeKind::Human { .. } => {
                        let data = reads(&self.workflow, &self.state, &node.reads, item.as_deref()).map_err(fail)?;
                        self.branches[i].at = At::Human { data };
                    }
                    NodeKind::Agent { persona, prompt } => {
                        if !window[i] {
                            continue;
                        }
                        let data = reads(&self.workflow, &self.state, &node.reads, item.as_deref()).map_err(fail)?;
                        let p = io.persona(persona).await.map_err(|e| fail(e.to_string()))?;
                        let mut p = node_persona(self.flow.as_deref(), io, &node.id, p).map_err(fail)?;
                        let delegates = node_delegates(self.flow.as_deref(), &node.id).to_vec();
                        let tools = if delegates.is_empty() {
                            None
                        } else {
                            let (reg, names) = node_registry(io, &delegates, &p.tools).await.map_err(fail)?;
                            p.tools.extend(names);
                            let reg = Arc::new(reg);
                            self.registries.0.insert(node.id.clone(), reg.clone());
                            Some(reg)
                        };
                        let (call_cap, run_cap) =
                            (p.budget.call_usd_micros, p.budget.run_usd_micros);
                        let share = self
                            .share(&mut entering)
                            .map(|s| run_cap.map_or(s, |c| c.min(s)));
                        let mut limits = node_limits(
                            &self.limits,
                            self.usage().cost_usd_micros,
                            call_cap,
                            run_cap,
                        );
                        if share.is_some() {
                            limits.run = share;
                        }
                        let input = vec![agent_input(prompt.as_deref(), data)];
                        let run = match &tools {
                            None => AgentRun::new(&p, input, limits, io.agent()).await,
                            Some(reg) => {
                                let tio = ToolsIo {
                                    inner: io.agent(),
                                    tools: reg,
                                };
                                AgentRun::new(&p, input, limits, &tio).await
                            }
                        }
                        .map_err(|e| fail(e.to_string()))?;
                        self.branches[i].at = At::Agent {
                            run: Box::new(run),
                            call_cap,
                            run_cap,
                            share,
                        };
                    }
                }
                changed = true;
            }
            changed |= self.close_joins();
            changed |= self.collect_maps()?;
            if !changed {
                changed = self.close_any();
            }
            if !changed {
                if self.even() {
                    self.top_up(false)?;
                }
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
            (b.at == At::Joined && !self.is_any(&b.node) && parked == self.workflow.in_degree(&b.node))
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
                item: None,
            },
        );
        true
    }
}

fn apply_finished(b: &mut Branch, finished: Finished) -> Work {
    match finished {
        Finished::Tool { output } => {
            assert_eq!(b.at, At::Enter, "kept tool results apply to branches at their tool");
            Work::Tool(Ok(output))
        }
        Finished::Agent { run: kept, outcome } => {
            let At::Agent { run, .. } = &mut b.at else {
                unreachable!("kept agent results apply to agent branches, checked before");
            };
            *run = kept;
            Work::Agent(outcome.outcome())
        }
    }
}

fn node_persona(flow: Option<&Flow>, io: &dyn GraphIo, id: &str, mut p: Persona) -> Result<Persona, String> {
    let Some(FlowKind::Agent { goto, model, .. }) = flow.and_then(|f| f.kind(id)) else {
        return Ok(p);
    };
    if let Some(model) = model.as_ref().filter(|m| **m != p.model) {
        io.allow_model(id, &p, model).map_err(|reason| {
            format!("model override `{model}` is not allowed for persona `{}`: {reason}", p.name)
        })?;
        p.model = model.clone();
    }
    if !goto.is_empty() {
        let mut fields = toml::Table::new();
        fields.insert("next".into(), toml::Value::String(format!("enum({})", goto.join(", "))));
        fields.insert("reason".into(), toml::Value::String("string".into()));
        p.output = Some(compile_output(&fields).map_err(|e| format!("goto targets: {e}"))?);
    }
    Ok(p)
}

fn node_delegates<'f>(flow: Option<&'f Flow>, id: &str) -> &'f [String] {
    match flow.and_then(|f| f.kind(id)) {
        Some(FlowKind::Agent { delegates, .. }) => delegates,
        _ => &[],
    }
}

async fn node_registry(
    io: &dyn GraphIo,
    delegates: &[String],
    persona_tools: &[String],
) -> Result<(ToolRegistry, Vec<String>), String> {
    let shared = io
        .agent_arc()
        .ok_or("delegates need a GraphIo that shares its AgentIo (agent_arc)")?;
    let mut settings = DelegateSettings::default();
    if let Some(store) = io.checkpoints() {
        settings = settings.with_store(store);
    }
    let mut coworkers = Vec::with_capacity(delegates.len());
    for name in delegates {
        let member = io.persona(name).await.map_err(|e| format!("delegate `{name}`: {e}"))?;
        let description = format!("{}: {}", member.display_name, member.role);
        let tool = AgentTool::new(name, &description, member, shared.clone(), settings.clone())
            .map_err(|e| e.to_string())?;
        coworkers.push((name.clone(), tool));
    }
    let delegation = DelegationTools::new(coworkers, settings).map_err(|e| e.to_string())?;
    let mut reg = ToolRegistry::new();
    for name in persona_tools {
        let tool = io
            .agent()
            .tools()
            .get(name)
            .ok_or_else(|| format!("tool `{name}` is not registered"))?;
        reg.register(tool).map_err(|e| e.to_string())?;
    }
    let mut names = Vec::new();
    for tool in delegation.tools() {
        names.push(tool.spec().name.clone());
        reg.register(tool).map_err(|e| e.to_string())?;
    }
    assert!(
        reg.specs().len() == persona_tools.len() + names.len(),
        "the node registry holds only persona and delegation tools"
    );
    Ok((reg, names))
}

fn goto_choice(output: &Value, targets: &[String]) -> Result<String, String> {
    let next = output.get("next").and_then(Value::as_str).unwrap_or_default();
    targets
        .iter()
        .find(|t| t.as_str() == next)
        .cloned()
        .ok_or_else(|| format!("chose `{next}`, which is not one of its goto targets"))
}

fn run_plain(kind: &FlowKind) -> FlowKind {
    match kind.clone() {
        FlowKind::Agent { persona, prompt, .. } => FlowKind::Agent {
            persona,
            prompt,
            goto: Vec::new(),
            delegates: Vec::new(),
            model: None,
            defer: false,
        },
        FlowKind::Tool { tool, .. } => FlowKind::Tool { tool, defer: false },
        FlowKind::Join { .. } => FlowKind::Join { mode: JoinMode::All },
        other => other,
    }
}

const MAX_SUBFLOW_DEPTH: usize = 4;

fn check_children<'a>(
    flow: &'a Flow,
    io: &'a dyn GraphIo,
    path: &'a mut Vec<String>,
) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<(), GraphError>> + Send + 'a>> {
    Box::pin(async move {
        assert!(path.len() <= MAX_SUBFLOW_DEPTH + 1, "deeper paths are refused before recursing");
        for n in &flow.workflow().nodes {
            let Some(FlowKind::Flow { flow: name, returns, .. }) = flow.kind(&n.id) else {
                continue;
            };
            let setup = |m: String| GraphError::Setup(format!("node `{}`: {m}", n.id));
            if path.contains(name) {
                return Err(setup(format!("flow `{name}` reaches itself: {} > {name}", path.join(" > "))));
            }
            if path.len() > MAX_SUBFLOW_DEPTH {
                return Err(setup(format!(
                    "subflows nest deeper than {MAX_SUBFLOW_DEPTH}: {} > {name}",
                    path.join(" > ")
                )));
            }
            let child = io.flow(name).await.map_err(|e| setup(e.to_string()))?;
            if let Some(feature) = unsupported(&child) {
                return Err(setup(format!("flow `{name}`: {feature} is not run by this engine yet")));
            }
            if let Some(m) = untracked_misuse(&child) {
                return Err(setup(format!("flow `{name}`: {m}")));
            }
            if !child.workflow().state.contains_key(returns) {
                return Err(setup(format!("flow `{name}` has no state key `{returns}` to return")));
            }
            for c in &child.workflow().nodes {
                if let NodeKind::Agent { persona, .. } = &c.kind {
                    let p = io.persona(persona).await.map_err(|e| setup(format!("flow `{name}`: {e}")))?;
                    node_persona(Some(&child), io, &c.id, p)
                        .map_err(|e| setup(format!("flow `{name}`, node `{}`: {e}", c.id)))?;
                }
            }
            path.push(name.clone());
            check_children(&child, io, path).await?;
            path.pop();
        }
        Ok(())
    })
}

fn untracked_misuse(flow: &Flow) -> Option<String> {
    let secret = |k: &str| flow.untracked().iter().any(|u| u == k);
    if let Some(k) = flow.untracked().iter().find(|k| flow.reducer(k).is_some()) {
        return Some(format!("untracked key `{k}` cannot have a reducer: reduced values are kept per step"));
    }
    for n in &flow.workflow().nodes {
        let kind = flow.kind(&n.id);
        let keeps_reads = matches!(
            kind,
            Some(FlowKind::Agent { .. } | FlowKind::Human { .. } | FlowKind::Flow { .. })
        );
        if let Some(k) = n.reads.iter().find(|k| keeps_reads && secret(k)) {
            return Some(format!(
                "node `{}` cannot read untracked key `{k}`: agent, human and flow nodes keep what they read in the checkpoint",
                n.id
            ));
        }
        if let Some(FlowKind::Map { over, .. }) = kind {
            if secret(over) || n.writes.as_deref().is_some_and(secret) {
                return Some(format!("map node `{}` cannot run over or write an untracked key", n.id));
            }
        }
        if matches!(kind, Some(FlowKind::Human { .. })) && n.writes.as_deref().is_some_and(secret) {
            return Some(format!("human node `{}` cannot write untracked key: its answer is checkpointed", n.id));
        }
    }
    None
}

fn unsupported(flow: &Flow) -> Option<String> {
    let node = flow.workflow().nodes.iter().find(|n| match flow.kind(&n.id) {
        Some(FlowKind::Map { .. } | FlowKind::Switch { .. } | FlowKind::Flow { .. }) => false,
        kind => kind.map(run_plain) != Some(FlowKind::from_v1(&n.kind)),
    })?;
    Some(format!("node `{}` of a 0.5 kind or option", node.id))
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

fn reads(
    workflow: &Workflow,
    state: &Map<String, Value>,
    keys: &[String],
    item: Option<&Item>,
) -> Result<Map<String, Value>, String> {
    keys.iter()
        .map(|k| match item.filter(|it| &it.key == k).map(|it| &it.value).or_else(|| state.get(k)) {
            Some(v) => Ok((k.clone(), v.clone())),
            None if nullable(workflow, k) => Ok((k.clone(), Value::Null)),
            None => Err(format!("reads `{k}`, which is not set")),
        })
        .collect()
}

fn nullable(workflow: &Workflow, key: &str) -> bool {
    workflow.state.get(key).is_some_and(|schema| match &schema["type"] {
        Value::Array(types) => types.iter().any(|t| t == "null"),
        _ => false,
    })
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
    workflow: &Workflow,
    node: &Node,
    tool: &str,
    state: &Map<String, Value>,
    item: Option<&Item>,
    ctx: ToolContext,
) -> Result<Value, ToolFailure> {
    let args = reads(workflow, state, &node.reads, item).map_err(|message| ToolFailure {
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
    let result = io
        .agent()
        .tools()
        .call(tool, Value::Object(args), &ctx)
        .await;
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

fn windows(branches: &[Branch], cap: impl Fn(&str) -> usize) -> Vec<bool> {
    let mut open = vec![true; branches.len()];
    let mut maps: BTreeMap<&str, Vec<(usize, usize, bool)>> = BTreeMap::new();
    for (pos, b) in branches.iter().enumerate() {
        if let Some(it) = &b.item {
            maps.entry(it.map.as_str())
                .or_default()
                .push((it.index, pos, b.at != At::Left));
        }
    }
    for (map, mut items) in maps {
        let cap = cap(map);
        items.sort_unstable_by_key(|(index, ..)| *index);
        assert!(
            items.windows(2).all(|w| w[0].0 < w[1].0),
            "a map has one branch per item"
        );
        let mut ahead = 0usize;
        for (_, pos, unfinished) in items {
            open[pos] = ahead < cap;
            ahead += usize::from(unfinished);
        }
    }
    open
}

fn ready_map(branches: &[Branch]) -> Option<(String, usize)> {
    let mut maps: Vec<(&str, usize, bool)> = Vec::new();
    let mut seen: BTreeMap<&str, usize> = BTreeMap::new();
    for (pos, b) in branches.iter().enumerate() {
        let Some(it) = &b.item else {
            continue;
        };
        let left = b.at == At::Left;
        match seen.get(it.map.as_str()) {
            Some(&k) => maps[k].2 &= left,
            None => {
                seen.insert(it.map.as_str(), maps.len());
                maps.push((it.map.as_str(), pos, left));
            }
        }
    }
    maps.into_iter()
        .find(|(_, _, ready)| *ready)
        .map(|(map, first, _)| (map.to_string(), first))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn item(map: &str, index: usize, at: At) -> Branch {
        Branch {
            node: "w".into(),
            at,
            call: None,
            item: Some(Box::new(Item {
                map: map.into(),
                index,
                key: "x".into(),
                value: Value::Null,
                result: None,
            })),
        }
    }

    #[test]
    fn window_counts_unfinished_lower_items_in_item_order() {
        let branches = vec![
            item("m", 3, At::Enter),
            Branch::enter("t"),
            item("m", 1, At::Left),
            item("m", 2, At::Enter),
            item("m", 0, At::Joined),
            item("n", 0, At::Enter),
        ];
        let open = windows(&branches, |map| if map == "m" { 2 } else { 1 });
        assert_eq!(open, vec![false, true, true, true, true, true]);
    }

    #[test]
    fn ready_map_is_the_first_map_whose_items_all_left() {
        let branches = vec![
            Branch::enter("t"),
            item("m", 0, At::Left),
            item("n", 0, At::Left),
            item("m", 1, At::Enter),
            item("n", 1, At::Left),
        ];
        assert_eq!(ready_map(&branches), Some(("n".to_string(), 2)));
        assert_eq!(ready_map(&branches[..2]), Some(("m".to_string(), 1)));
        assert_eq!(ready_map(&branches[..1]), None);
    }
}
