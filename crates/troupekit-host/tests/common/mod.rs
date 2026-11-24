#![allow(dead_code)]

use serde_json::{json, Map, Value};
use std::sync::{Arc, Mutex};
use troupekit_agent::{Limits, MemorySource, PersonaLoader, SimpleIo};
use troupekit_core::clock::FixedClock;
use troupekit_core::error::{ProviderError, ProviderErrorKind};
use troupekit_core::ids::RunId;
use troupekit_core::inmem::InMemCheckpoints;
use troupekit_core::message::Role;
use troupekit_graph::testing::{text_reply, FnModel, FnTool};
use troupekit_graph::{GraphRun, SimpleGraphIo, Workflow};
use troupekit_host::{EventLog, InMemQueue, Runner};
use troupekit_providers::Price;
use troupekit_tools::{ToolContext, ToolError, ToolOutput, ToolRegistry};

pub const PRICE: Price = Price {
    input_usd_micros_per_mtok: 1_000_000,
    output_usd_micros_per_mtok: 2_000_000,
    verified: true,
    long_context: None,
    search_usd_micros: 0,
};

pub const LINEAR: &str = r#"
name = "linear"
start = "plan"
[state]
goal = "string"
plan = "{steps: list}"
note = "string"
[[nodes]]
id = "plan"
kind = "agent"
persona = "planner"
prompt = "Plan the week."
reads = ["goal"]
writes = "plan"
[[nodes]]
id = "save"
kind = "tool"
tool = "save_plan"
reads = ["plan"]
writes = "note"
[[nodes]]
id = "done"
kind = "end"
[[edges]]
from = "plan"
to = "save"
[[edges]]
from = "save"
to = "done"
"#;

pub const APPROVAL: &str = r#"
name = "approve"
start = "plan"
[state]
goal = "string"
plan = "{steps: list}"
approved = "bool"
[[nodes]]
id = "plan"
kind = "agent"
persona = "planner"
prompt = "Plan."
reads = ["goal"]
writes = "plan"
[[nodes]]
id = "approve"
kind = "human"
ask = "approval"
reads = ["plan"]
writes = "approved"
[[nodes]]
id = "route"
kind = "router"
condition = "flag(approved)"
[[nodes]]
id = "done"
kind = "end"
[[edges]]
from = "plan"
to = "approve"
[[edges]]
from = "approve"
to = "route"
[[edges]]
from = "route"
to = "done"
when = "true"
[[edges]]
from = "route"
to = "plan"
when = "false"
"#;

/// A model that plans from the goal in the last user message: a goal
/// holding `blocked` gets a rate limit, any other a two-step plan.
/// The reply depends only on the request.
pub fn planner_model() -> FnModel {
    FnModel::new("fake/m", |req| {
        let goal = req
            .messages
            .iter()
            .rev()
            .find(|m| m.role == Role::User)
            .map(|m| m.text_content())
            .unwrap_or_default();
        if goal.contains("blocked") {
            return Err(ProviderError::new(
                ProviderErrorKind::RateLimited,
                "slow down",
            ));
        }
        text_reply(r#"{"steps": ["warm up", "run"]}"#, 10)
    })
}

/// Registers `save_plan`, which logs each call's idempotency key and
/// answers `saved`, or the error output `boom` when `fail` is set.
/// Returns the registry and the key log.
pub fn saving_tools(fail: bool) -> (ToolRegistry, Arc<Mutex<Vec<String>>>) {
    let keys = Arc::new(Mutex::new(Vec::new()));
    let log = keys.clone();
    let mut tools = ToolRegistry::new();
    tools
        .register(FnTool::new("save_plan", move |_, ctx: ToolContext| {
            let log = log.clone();
            async move {
                log.lock().unwrap().push(ctx.idempotency_key.unwrap());
                if fail {
                    return Ok(ToolOutput::error("boom"));
                }
                Ok::<_, ToolError>(ToolOutput::text("saved"))
            }
        }))
        .unwrap();
    (tools, keys)
}

/// Everything a runner needs, in memory, on one fixed clock: queue,
/// checkpoints, graph io, event log and the model.
/// Several runners built from one stack share all of it.
pub struct Stack {
    pub queue: Arc<InMemQueue>,
    pub checkpoints: Arc<InMemCheckpoints>,
    pub io: Arc<SimpleGraphIo>,
    pub events: Arc<EventLog>,
    pub clock: Arc<FixedClock>,
    pub model: Arc<FnModel>,
}

/// Builds a stack serving `model` and `tools`, with the `planner`
/// persona, at clock time 1_000_000.
/// Returns the stack.
pub fn stack(model: FnModel, tools: ToolRegistry) -> Stack {
    let model = Arc::new(model);
    let clock = Arc::new(FixedClock::new(1_000_000));
    let events = Arc::new(EventLog::default());
    let mut agent = SimpleIo::new(tools).with_model("fake/m", model.clone(), PRICE);
    agent.events = events.clone();
    agent.clock = clock.clone();
    let personas = PersonaLoader::new(vec![Arc::new(MemorySource::new(
        "test",
        [(
            "planner.toml".to_string(),
            "name = 'planner'\ndisplay_name = 'P'\nrole = 'Plans.'\ngoal = 'A plan.'\ninstructions = 'Plan it.'\nmodel = 'fake/m'\n[output]\nsteps = 'list'\n".to_string(),
        )],
    ))]);
    Stack {
        queue: Arc::new(InMemQueue::new(clock.clone())),
        checkpoints: Arc::new(InMemCheckpoints::default()),
        io: Arc::new(SimpleGraphIo::new(Arc::new(agent), personas)),
        events,
        clock,
        model,
    }
}

impl Stack {
    /// Builds a runner over this stack's queue, checkpoints and io.
    /// Call it again for a second worker on the same stores.
    /// Returns the runner.
    pub fn runner(&self) -> Runner {
        Runner::new(
            self.queue.clone(),
            self.checkpoints.clone(),
            self.io.clone(),
        )
    }

    /// Starts a run of workflow `text` with `goal` and enqueues it.
    /// Returns its run id.
    /// Panics on a setup error, as tests pass valid workflows.
    pub async fn start(&self, text: &str, goal: &str) -> RunId {
        let wf = Workflow::parse("test/wf.toml", text).unwrap();
        let input: Map<String, Value> = [("goal".to_string(), json!(goal))].into_iter().collect();
        let run = GraphRun::new(wf, input, Limits::default(), self.io.as_ref())
            .await
            .unwrap();
        self.runner().enqueue(&run).await.unwrap();
        run.run_id()
    }

    /// Loads the latest saved state of `id`.
    /// Returns the run restored from it.
    /// Panics when there is none.
    pub async fn saved(&self, id: RunId) -> GraphRun {
        use troupekit_core::store::CheckpointStore;
        let cp = self.checkpoints.latest(id).await.unwrap().unwrap();
        GraphRun::from_checkpoint(&cp).unwrap()
    }
}
