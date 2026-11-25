use crate::agent::Agent;
use crate::built::Discard;
use crate::error::KitError;
use crate::laws::Laws;
use crate::storage::Storage;
use crate::tool::AsTool;
use async_trait::async_trait;
use serde_json::{Map, Value};
use std::collections::HashMap;
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;
use toml::Value as Toml;
use troupekit_agent::{
    AgentIo, EvalFn, InMemoryLedger, Limits, MemorySource, PersonaLoader, SimpleIo, UsageLedger,
};
use troupekit_core::clock::Clock;
use troupekit_core::event::Usage;
use troupekit_core::ids::RunId;
use troupekit_core::inmem::InMemCheckpoints;
use troupekit_core::store::CheckpointStore;
use troupekit_graph::{GraphOutcome, GraphRun, SimpleGraphIo, Workflow};
use troupekit_tools::{Tool, ToolContext, ToolError, ToolOutput, ToolRegistry, ToolSpec};

const MAX_NODES: usize = 64;
const MAX_GRAPH_RETRIES: u32 = 50;
const DEFAULT_GRAPH_RETRIES: u32 = 3;
const RETRY_WAIT_CHUNK_MS: i64 = 300_000;
const INPUT: &str = "input";
const END: &str = "end";

#[derive(Clone)]
enum Node {
    Agent(Box<Agent>),
    Step(Arc<dyn Tool>),
}

/// Builds a pipeline of agents and tool fns that run one after another.
/// Each node reads the output of the node before it, the first reads the
/// input; the last node's output is the answer. Fan-out, routers and
/// human nodes stay in workflow TOML.
#[derive(Clone)]
pub struct Graph {
    name: String,
    nodes: Vec<(String, Node)>,
    edges: Vec<(String, String)>,
    storage: Storage,
    limits: Limits,
    ledger: Option<Arc<dyn UsageLedger>>,
    max_steps: Option<u32>,
    retries: u32,
    law_paths: Vec<PathBuf>,
    inline_laws: Vec<String>,
    auto_laws: bool,
}

impl Graph {
    /// Starts a graph builder called `name`, 1 to 64 of `[a-z0-9-]`.
    /// `build` checks the name with the workflow rules.
    /// Every setting starts at its documented default.
    pub fn new(name: impl Into<String>) -> Self {
        Graph {
            name: name.into(),
            nodes: Vec::new(),
            edges: Vec::new(),
            storage: Storage::memory(),
            limits: Limits::default(),
            ledger: None,
            max_steps: None,
            retries: DEFAULT_GRAPH_RETRIES,
            law_paths: Vec::new(),
            inline_laws: Vec::new(),
            auto_laws: true,
        }
    }

    /// Adds agent node `id` running `agent` on the previous node's output.
    /// The agent keeps its own model binding; its tools and evals join the
    /// graph, and its storage, retries and events are the graph's. An agent
    /// with its own limits, laws or ledger fails `build`: set them on the
    /// graph, so no spend cap or counter is dropped.
    pub fn agent(mut self, id: impl Into<String>, agent: Agent) -> Self {
        self.nodes.push((id.into(), Node::Agent(Box::new(agent))));
        self
    }

    /// Adds fn node `id` calling `step`, a `#[tool]` fn or `tool(..)`.
    /// Its argument names must equal the ids of the nodes that feed it
    /// (`input` for the first node); `build` checks this.
    pub fn step(mut self, id: impl Into<String>, step: impl AsTool) -> Self {
        self.nodes.push((id.into(), Node::Step(step.into_arc())));
        self
    }

    /// Adds an edge from node `from` to node `to`.
    /// Edges must form one chain; with no edges the nodes run in the
    /// order they were added.
    pub fn edge(mut self, from: impl Into<String>, to: impl Into<String>) -> Self {
        self.edges.push((from.into(), to.into()));
        self
    }

    /// Sets where graph checkpoints live.
    /// Defaults to `Storage::memory()`, fresh per run.
    /// Returns the builder.
    pub fn checkpoints(mut self, storage: Storage) -> Self {
        self.storage = storage;
        self
    }

    /// Sets the spend limits for the whole graph run.
    /// Defaults to `Limits::default()`, no limit.
    /// Returns the builder.
    pub fn limits(mut self, limits: Limits) -> Self {
        self.limits = limits;
        self
    }

    /// Sets the usage ledger that period limits (`scope`, `user`) count in.
    /// Defaults to a fresh in-memory ledger per build; pass a durable one
    /// so period caps survive rebuilds and restarts.
    pub fn ledger(mut self, ledger: Arc<dyn UsageLedger>) -> Self {
        self.ledger = Some(ledger);
        self
    }

    /// Caps graph steps per run; 1 to 10000, checked at `build`.
    /// Defaults to the workflow default.
    /// Returns the builder.
    pub fn max_steps(mut self, steps: u32) -> Self {
        self.max_steps = Some(steps);
        self
    }

    /// Sets how often a transient failure is retried per run.
    /// At most 50, checked at `build`; defaults to 3.
    /// Returns the builder.
    pub fn retries(mut self, retries: u32) -> Self {
        self.retries = retries;
        self
    }

    /// Adds a `.bend` law file, checked at `build`.
    /// 0.2 stores laws; no run proves them yet.
    /// Returns the builder.
    pub fn laws(mut self, path: impl AsRef<Path>) -> Self {
        self.law_paths.push(path.as_ref().to_path_buf());
        self
    }

    /// Adds one inline law, checked at `build`.
    /// 0.2 stores laws; no run proves them yet.
    /// Returns the builder.
    pub fn law(mut self, law: impl Into<String>) -> Self {
        self.inline_laws.push(law.into());
        self
    }

    /// Asks for laws derived from the code as well. Defaults to `true`.
    /// 0.2 records the choice only.
    /// Returns the builder.
    pub fn auto_laws(mut self, on: bool) -> Self {
        self.auto_laws = on;
        self
    }

    /// Checks the nodes, edges, workflow, every agent and tool, and the
    /// storage, then returns a ready graph. Makes no model call.
    /// Fails with the first problem found.
    pub async fn build(self) -> Result<BuiltGraph, KitError> {
        let owner = format!("graph:{}", self.name);
        let fail = |m: String| KitError::config(&owner, m);
        if self.retries > MAX_GRAPH_RETRIES {
            return Err(fail(format!(
                "retries must be at most {MAX_GRAPH_RETRIES}, got {}",
                self.retries
            )));
        }
        let order = self.chain().map_err(fail)?;
        let laws = Laws::check(&owner, &self.law_paths, &self.inline_laws, self.auto_laws)?;
        let workflow = self.workflow(&order)?;
        let mut registry = ToolRegistry::new();
        let mut agent_io = SimpleIo::new(ToolRegistry::new());
        let mut files: HashMap<String, String> = HashMap::new();
        let mut keys: HashMap<String, (String, &Agent)> = HashMap::new();
        let mut evals: HashMap<String, EvalFn> = HashMap::new();
        for (i, id) in order.iter().enumerate() {
            let feeder = feeder(&order, i);
            match self.node(id) {
                Node::Agent(agent) => {
                    if let Some(what) = agent.graph_conflict() {
                        return Err(fail(format!(
                            "agent node `{id}` sets its own {what} on the graph"
                        )));
                    }
                    let key = match keys.get(agent.name()) {
                        Some((key, first)) if first.same_model(agent) => key.clone(),
                        Some(_) => {
                            return Err(fail(format!(
                                "two different agents are named `{}`",
                                agent.name()
                            )));
                        }
                        None => {
                            let key = format!("agent-{}", keys.len());
                            keys.insert(agent.name().to_string(), (key.clone(), agent));
                            key
                        }
                    };
                    let prepared = agent.prepare(Some(&key))?;
                    for tool in &prepared.tools {
                        share(&mut registry, tool.clone()).map_err(fail)?;
                    }
                    agent_io.models.entry(key).or_insert(prepared.handle);
                    for (eval, f) in prepared.eval_fns {
                        match evals.get(&eval) {
                            Some(first) if Arc::ptr_eq(first, &f) => {}
                            Some(_) => {
                                return Err(fail(format!(
                                    "eval `{eval}` has two different callbacks; pass one shared EvalFn"
                                )));
                            }
                            None => {
                                agent_io.evals.register(&eval, f.clone())?;
                                evals.insert(eval, f);
                            }
                        }
                    }
                    let file = format!("{}.toml", prepared.persona.name);
                    match files.get(&file) {
                        Some(text) if *text != prepared.persona_text => {
                            return Err(fail(format!(
                                "two different agents are named `{}`",
                                prepared.persona.name
                            )));
                        }
                        Some(_) => {}
                        None => {
                            files.insert(file, prepared.persona_text);
                        }
                    }
                }
                Node::Step(tool) => {
                    let args = arg_names(tool.spec());
                    if args != [feeder] {
                        return Err(fail(format!(
                            "step `{id}` takes ({}) but is fed by ({feeder}); name its arguments after the nodes that feed it",
                            args.join(", ")
                        )));
                    }
                    if !takes_text(&tool.spec().schema["properties"][feeder]) {
                        return Err(fail(format!(
                            "step `{id}` argument `{feeder}` must accept a string; every graph node is fed text"
                        )));
                    }
                    registry.register(Arc::new(StepTool::new(id, tool.clone())))?;
                }
            }
        }
        agent_io.tools = registry;
        agent_io.events = Arc::new(Discard);
        agent_io.ledger = self
            .ledger
            .clone()
            .unwrap_or_else(|| Arc::new(InMemoryLedger::default()));
        let agent_io: Arc<dyn AgentIo> = Arc::new(agent_io);
        let loader = PersonaLoader::new(vec![Arc::new(MemorySource::new("graph", files))]);
        let io = SimpleGraphIo::new(agent_io, loader);
        GraphRun::new(
            workflow.clone(),
            input_state("build check"),
            self.limits.clone(),
            &io,
        )
        .await?;
        let store = self.storage.open().await?;
        let last = order.last().expect("chain has a node").clone();
        Ok(BuiltGraph {
            inner: Arc::new(Inner {
                owner,
                workflow,
                io,
                limits: self.limits,
                store,
                last,
                laws,
            }),
        })
    }

    /// Builds the graph and runs it once on `input`.
    /// Returns the last node's output.
    /// Fails like `build`, then like `BuiltGraph::run`.
    pub async fn run(self, input: impl Into<String>) -> Result<Value, KitError> {
        self.build().await?.run(input).await
    }

    /// Builds the graph and runs it once on `input`.
    /// Returns the full report.
    /// Fails like `build`, then like `BuiltGraph::run_with`.
    pub async fn run_with(self, input: impl Into<String>) -> Result<GraphReport, KitError> {
        self.build().await?.run_with(input).await
    }

    fn node(&self, id: &str) -> &Node {
        let found = self.nodes.iter().find(|(n, _)| n == id);
        &found.expect("chain ids come from the nodes").1
    }

    fn chain(&self) -> Result<Vec<String>, String> {
        if self.nodes.is_empty() {
            return Err("needs at least one node".into());
        }
        if self.nodes.len() > MAX_NODES {
            return Err(format!(
                "takes at most {MAX_NODES} nodes, got {}",
                self.nodes.len()
            ));
        }
        let mut ids: Vec<&str> = Vec::with_capacity(self.nodes.len());
        for (id, _) in &self.nodes {
            if id == INPUT || id == END {
                return Err(format!("node id `{id}` is reserved"));
            }
            if ids.contains(&id.as_str()) {
                return Err(format!("duplicate node id `{id}`"));
            }
            ids.push(id);
        }
        if self.edges.is_empty() {
            return Ok(ids.iter().map(|s| s.to_string()).collect());
        }
        for (from, to) in &self.edges {
            if let Some(unknown) = [from, to].into_iter().find(|n| !ids.contains(&n.as_str())) {
                return Err(format!("edge names unknown node `{unknown}`"));
            }
        }
        let count = |pick: &dyn Fn(&(String, String)) -> &str, id: &str| {
            self.edges.iter().filter(|e| pick(e) == id).count()
        };
        for id in &ids {
            if count(&|e| e.0.as_str(), id) > 1 {
                return Err(format!(
                    "node `{id}` has two outgoing edges; fan-out needs workflow TOML"
                ));
            }
            if count(&|e| e.1.as_str(), id) > 1 {
                return Err(format!(
                    "node `{id}` has two incoming edges; joins need workflow TOML"
                ));
            }
        }
        let Some(start) = ids.iter().find(|id| count(&|e| e.1.as_str(), id) == 0) else {
            return Err("edges form a cycle".into());
        };
        let mut order = vec![start.to_string()];
        while order.len() <= ids.len() {
            let last = order.last().expect("order starts non-empty");
            let Some((_, next)) = self.edges.iter().find(|(from, _)| from == last) else {
                break;
            };
            if order.contains(next) {
                return Err("edges form a cycle".into());
            }
            order.push(next.clone());
        }
        if let Some(lone) = ids.iter().find(|id| !order.iter().any(|o| o == *id)) {
            return Err(format!("node `{lone}` is not on the chain"));
        }
        assert_eq!(order.len(), ids.len());
        Ok(order)
    }

    fn workflow(&self, order: &[String]) -> Result<Workflow, KitError> {
        let mut state = toml::Table::new();
        state.insert(INPUT.into(), Toml::String("string".into()));
        let mut nodes = Vec::with_capacity(order.len() + 1);
        for (i, id) in order.iter().enumerate() {
            state.insert(id.clone(), Toml::String("string".into()));
            let mut node = toml::Table::new();
            node.insert("id".into(), Toml::String(id.clone()));
            match self.node(id) {
                Node::Agent(agent) => {
                    node.insert("kind".into(), Toml::String("agent".into()));
                    node.insert("persona".into(), Toml::String(agent.name().to_string()));
                }
                Node::Step(_) => {
                    node.insert("kind".into(), Toml::String("tool".into()));
                    node.insert("tool".into(), Toml::String(step_name(id)));
                }
            }
            let reads = vec![Toml::String(feeder(order, i).to_string())];
            node.insert("reads".into(), Toml::Array(reads));
            node.insert("writes".into(), Toml::String(id.clone()));
            nodes.push(Toml::Table(node));
        }
        let mut end = toml::Table::new();
        end.insert("id".into(), Toml::String(END.into()));
        end.insert("kind".into(), Toml::String("end".into()));
        nodes.push(Toml::Table(end));
        let mut edges = Vec::with_capacity(order.len());
        for (i, from) in order.iter().enumerate() {
            let to = order.get(i + 1).map_or(END, String::as_str);
            let mut edge = toml::Table::new();
            edge.insert("from".into(), Toml::String(from.clone()));
            edge.insert("to".into(), Toml::String(to.to_string()));
            edges.push(Toml::Table(edge));
        }
        let mut table = toml::Table::new();
        table.insert("name".into(), Toml::String(self.name.clone()));
        table.insert("start".into(), Toml::String(order[0].clone()));
        if let Some(steps) = self.max_steps {
            table.insert("max_steps".into(), Toml::Integer(i64::from(steps)));
        }
        table.insert("max_retries".into(), Toml::Integer(i64::from(self.retries)));
        table.insert("state".into(), Toml::Table(state));
        table.insert("nodes".into(), Toml::Array(nodes));
        table.insert("edges".into(), Toml::Array(edges));
        let text = toml::to_string(&table).map_err(|e| {
            KitError::config(
                &format!("graph:{}", self.name),
                format!("workflow does not serialize: {e}"),
            )
        })?;
        Ok(Workflow::parse(&format!("graph:{}", self.name), &text)?)
    }
}

impl fmt::Debug for Graph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ids: Vec<&str> = self.nodes.iter().map(|(id, _)| id.as_str()).collect();
        f.debug_struct("Graph")
            .field("name", &self.name)
            .field("nodes", &ids)
            .field("edges", &self.edges)
            .field("storage", &self.storage)
            .field("retries", &self.retries)
            .field("ledger", &self.ledger.is_some())
            .finish_non_exhaustive()
    }
}

/// The result of one finished graph run: the last node's output, the
/// whole state, usage, steps and the run id.
/// Read it through the getters.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct GraphReport {
    output: Value,
    state: Map<String, Value>,
    usage: Usage,
    steps: u64,
    run_id: RunId,
}

impl GraphReport {
    /// Returns the last node's output, the answer of the graph.
    /// A string for text nodes.
    /// The same value `run` returns.
    pub fn output(&self) -> &Value {
        &self.output
    }

    /// Returns the final state: `input` plus one entry per node id.
    /// Each value is that node's output.
    /// Read only.
    pub fn state(&self) -> &Map<String, Value> {
        &self.state
    }

    /// Returns the tokens and cost every agent node used, summed.
    /// Cost is in millionths of a US dollar.
    /// Tool nodes add nothing.
    pub fn usage(&self) -> Usage {
        self.usage
    }

    /// Returns how many graph steps the run took.
    /// Bounded by the workflow's `max_steps`.
    /// At least one.
    pub fn steps(&self) -> u64 {
        self.steps
    }

    /// Returns the id of the run, for `resume` and logs.
    /// Unique per run.
    /// Stable across checkpoints.
    pub fn run_id(&self) -> RunId {
        self.run_id
    }
}

/// A graph that passed every `build` check, ready to run many times.
/// Cheap to clone and safe to share across tasks; in-memory checkpoints
/// are dropped after each run.
#[derive(Clone)]
pub struct BuiltGraph {
    inner: Arc<Inner>,
}

struct Inner {
    owner: String,
    workflow: Workflow,
    io: SimpleGraphIo,
    limits: Limits,
    store: Option<Arc<dyn CheckpointStore>>,
    last: String,
    laws: Laws,
}

impl BuiltGraph {
    /// Runs the graph once on `input` and returns the last node's output.
    /// Fails with `Failed` or `Paused` when the run ends without one.
    /// Transient failures are retried by the graph engine.
    pub async fn run(&self, input: impl Into<String>) -> Result<Value, KitError> {
        Ok(self.run_with(input).await?.output)
    }

    /// Runs the graph once on `input` and returns the full report.
    /// Same failures as `run`.
    /// The run id in the report works with `resume`.
    pub async fn run_with(&self, input: impl Into<String>) -> Result<GraphReport, KitError> {
        let inner = &self.inner;
        let mut run = GraphRun::new(
            inner.workflow.clone(),
            input_state(&input.into()),
            inner.limits.clone(),
            &inner.io,
        )
        .await?;
        let store: Arc<dyn CheckpointStore> = match &inner.store {
            Some(store) => store.clone(),
            None => Arc::new(InMemCheckpoints::default()),
        };
        self.finish(&mut run, store.as_ref()).await
    }

    /// Continues graph run `run_id` from its latest checkpoint and returns
    /// the last node's output; a finished run returns it again. Needs
    /// durable storage and a run this same workflow started.
    pub async fn resume(&self, run_id: RunId) -> Result<Value, KitError> {
        let inner = &self.inner;
        let Some(store) = &inner.store else {
            return Err(KitError::config(
                &inner.owner,
                "resume needs Storage::sqlite(..) or Storage::custom(..) checkpoints",
            ));
        };
        let Some(checkpoint) = store.latest(run_id).await? else {
            return Err(KitError::config(
                &inner.owner,
                format!("no checkpoint for run {run_id}"),
            ));
        };
        let mut run = GraphRun::from_checkpoint(&checkpoint)?;
        if run.workflow() != &inner.workflow {
            return Err(KitError::config(
                &inner.owner,
                format!(
                    "run {run_id} was started by a different workflow; resume it with the graph that started it"
                ),
            ));
        }
        Ok(self.finish(&mut run, store.as_ref()).await?.output)
    }

    /// Returns the validated workflow this graph runs.
    /// Built from the builder by `Workflow::parse`.
    /// Read only.
    pub fn workflow(&self) -> &Workflow {
        &self.inner.workflow
    }

    /// Returns the laws checked at `build`.
    /// 0.2 stores them; runs do not prove them yet.
    /// Read only.
    pub fn laws(&self) -> &Laws {
        &self.inner.laws
    }

    async fn finish(
        &self,
        run: &mut GraphRun,
        store: &dyn CheckpointStore,
    ) -> Result<GraphReport, KitError> {
        let inner = &self.inner;
        if store.latest(run.run_id()).await?.is_none() {
            store.save(run.checkpoint()).await?;
        }
        let cap =
            2 * u64::from(inner.workflow.max_steps) + u64::from(inner.workflow.max_retries) + 8;
        for _ in 0..cap {
            let before = run.steps();
            let outcome = run.step(&inner.io).await;
            if run.steps() != before {
                store.save(run.checkpoint()).await?;
            }
            match outcome {
                GraphOutcome::Continue => {}
                GraphOutcome::RetryAt { at_ms, .. } => {
                    wait_until(inner.io.agent.clock(), at_ms, RETRY_WAIT_CHUNK_MS).await;
                }
                GraphOutcome::Waiting(reason) => {
                    return Err(KitError::Paused {
                        run_id: run.run_id(),
                        reason,
                    })
                }
                GraphOutcome::Failed(reason) => {
                    return Err(KitError::Failed {
                        run_id: run.run_id(),
                        reason,
                    })
                }
                GraphOutcome::Done(state) => {
                    let output = state
                        .get(&inner.last)
                        .cloned()
                        .expect("a finished chain run wrote its last node");
                    return Ok(GraphReport {
                        output,
                        state,
                        usage: run.usage(),
                        steps: run.steps(),
                        run_id: run.run_id(),
                    });
                }
            }
        }
        Err(KitError::Failed {
            run_id: run.run_id(),
            reason: format!("did not finish within {cap} steps"),
        })
    }
}

impl fmt::Debug for BuiltGraph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let inner = &self.inner;
        f.debug_struct("BuiltGraph")
            .field("name", &inner.workflow.name)
            .field("nodes", &inner.workflow.nodes.len())
            .field("durable", &inner.store.is_some())
            .field("laws", &inner.laws)
            .finish_non_exhaustive()
    }
}

struct StepTool {
    spec: ToolSpec,
    inner: Arc<dyn Tool>,
}

impl StepTool {
    fn new(id: &str, inner: Arc<dyn Tool>) -> Self {
        let spec = ToolSpec {
            name: step_name(id),
            ..inner.spec().clone()
        };
        StepTool { spec, inner }
    }
}

#[async_trait]
impl Tool for StepTool {
    fn spec(&self) -> &ToolSpec {
        &self.spec
    }

    async fn call(&self, args: Value, ctx: &ToolContext) -> Result<ToolOutput, ToolError> {
        let out = self.inner.call(args, ctx).await?;
        if out.is_error {
            return Ok(out);
        }
        Ok(ToolOutput::text(Value::String(out.content).to_string()))
    }
}

async fn wait_until(clock: &dyn Clock, at_ms: i64, chunk_ms: i64) {
    assert!(chunk_ms > 0, "a wait chunk lasts some time");
    let first = at_ms.saturating_sub(clock.now_ms());
    let rounds = first.max(0) / chunk_ms + 2;
    for _ in 0..rounds {
        let left = at_ms.saturating_sub(clock.now_ms());
        if left <= 0 {
            return;
        }
        let ms = u64::try_from(left.min(chunk_ms)).expect("left is positive");
        tokio::time::sleep(Duration::from_millis(ms)).await;
    }
}

fn step_name(id: &str) -> String {
    format!("step__{id}")
}

fn feeder(order: &[String], i: usize) -> &str {
    if i == 0 {
        INPUT
    } else {
        &order[i - 1]
    }
}

fn input_state(text: &str) -> Map<String, Value> {
    let mut state = Map::new();
    state.insert(INPUT.into(), Value::String(text.to_string()));
    state
}

fn takes_text(schema: &Value) -> bool {
    match schema {
        Value::Bool(any) => *any,
        Value::Object(map) => match map.get("type") {
            Some(Value::String(t)) => t == "string",
            Some(Value::Array(types)) => types.iter().any(|t| t == "string"),
            Some(_) => false,
            None => !["$ref", "anyOf", "oneOf", "allOf", "enum", "const", "not"]
                .iter()
                .any(|k| map.contains_key(*k)),
        },
        _ => false,
    }
}

fn arg_names(spec: &ToolSpec) -> Vec<String> {
    let mut names: Vec<String> = spec.schema["properties"]
        .as_object()
        .map(|p| p.keys().cloned().collect())
        .unwrap_or_default();
    names.sort();
    names
}

fn share(registry: &mut ToolRegistry, tool: Arc<dyn Tool>) -> Result<(), String> {
    let name = tool.spec().name.clone();
    match registry.get(&name) {
        Some(existing) if Arc::ptr_eq(&existing, &tool) => Ok(()),
        Some(existing) if existing.spec() == tool.spec() => Err(format!(
            "tool `{name}` has two different implementations; pass one shared tool to both agents"
        )),
        Some(_) => Err(format!(
            "tool `{name}` is registered twice with different specs"
        )),
        None => registry.register(tool).map_err(|e| e.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::wait_until;
    use troupekit_core::clock::{Clock, SystemClock};

    #[tokio::test]
    async fn retry_wait_reaches_a_deadline_past_one_chunk() {
        let clock = SystemClock;
        let at = clock.now_ms() + 25;
        wait_until(&clock, at, 10).await;
        assert!(clock.now_ms() >= at);
        let before = clock.now_ms();
        wait_until(&clock, before - 1_000, 10).await;
        assert!(clock.now_ms() - before < 10);
    }
}
