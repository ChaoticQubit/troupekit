#![allow(dead_code)]

use async_trait::async_trait;
use serde_json::{json, Map, Value};
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use troupekit::agent::{Limits, PersonaLoader, SimpleIo};
use troupekit::core::clock::Clock;
use troupekit::core::ids::RunId;
use troupekit::graph::{GraphRun, GraphStatus, SimpleGraphIo, Workflow};
use troupekit::host::TickOptions;
use troupekit::Troupe;
use troupekit_tools::{Tool, ToolContext, ToolError, ToolOutput, ToolRegistry, ToolSpec};

pub const WORKFLOW: &str = r#"
name = "three"
start = "first"
[state]
goal = "string"
a = "string"
b = "string"
c = "string"
[[nodes]]
id = "first"
kind = "tool"
tool = "first"
reads = ["goal"]
writes = "a"
[[nodes]]
id = "second"
kind = "tool"
tool = "second"
reads = ["a"]
writes = "b"
[[nodes]]
id = "third"
kind = "tool"
tool = "third"
reads = ["b"]
writes = "c"
[[nodes]]
id = "done"
kind = "end"
[[edges]]
from = "first"
to = "second"
[[edges]]
from = "second"
to = "third"
[[edges]]
from = "third"
to = "done"
"#;

pub type Hook = Arc<dyn Fn() -> Pin<Box<dyn Future<Output = ()> + Send>> + Send + Sync>;

/// A pure tool: it joins the string value of its one argument with its own
/// name. `hook`, when set, runs before the answer, so a test can stop the
/// tool in the middle of a step.
pub struct Join {
    spec: ToolSpec,
    hook: Option<Hook>,
}

#[async_trait]
impl Tool for Join {
    fn spec(&self) -> &ToolSpec {
        &self.spec
    }

    async fn call(&self, args: Value, _ctx: &ToolContext) -> Result<ToolOutput, ToolError> {
        let input = args
            .as_object()
            .and_then(|m| m.values().next())
            .and_then(Value::as_str)
            .ok_or_else(|| ToolError::InvalidArgs("one string argument".into()))?
            .to_string();
        if let Some(hook) = &self.hook {
            hook().await;
        }
        let out = format!("{input}>{}", self.spec.name);
        Ok(ToolOutput::text(serde_json::to_string(&out).unwrap()))
    }
}

fn join(name: &str, hook: Option<Hook>) -> Arc<dyn Tool> {
    Arc::new(Join {
        spec: ToolSpec {
            name: name.into(),
            description: format!("joins its input with {name}"),
            schema: json!({"type": "object"}),
            hosted: None,
        },
        hook,
    })
}

/// The three tools of `WORKFLOW`; `second_hook` runs inside the second.
pub fn tools(second_hook: Option<Hook>) -> ToolRegistry {
    let mut tools = ToolRegistry::new();
    tools.register(join("first", None)).unwrap();
    tools.register(join("second", second_hook)).unwrap();
    tools.register(join("third", None)).unwrap();
    tools
}

/// A graph io over `tools` whose ledger is the troupe's and whose clock is
/// `clock`. No model and no persona: the workflow is tool-only.
pub fn graph_io(troupe: &Troupe, tools: ToolRegistry, clock: Arc<dyn Clock>) -> Arc<SimpleGraphIo> {
    let mut agent = SimpleIo::new(tools);
    agent.ledger = troupe.ledger();
    agent.clock = clock;
    Arc::new(SimpleGraphIo::new(
        Arc::new(agent),
        PersonaLoader::new(Vec::new()),
    ))
}

/// Creates a run of `WORKFLOW` with `goal` and enqueues it on `troupe`.
pub async fn start(troupe: &Troupe, io: Arc<SimpleGraphIo>, goal: &str) -> RunId {
    let wf = Workflow::parse("three.toml", WORKFLOW).unwrap();
    let input: Map<String, Value> = [("goal".to_string(), json!(goal))].into_iter().collect();
    let run = GraphRun::new(wf, input, Limits::default(), io.as_ref())
        .await
        .unwrap();
    troupe.runner(io).enqueue(&run).await.unwrap();
    run.run_id()
}

/// Loads the latest checkpoint of `id` as a run.
pub async fn saved(troupe: &Troupe, id: RunId) -> GraphRun {
    let cp = troupe.checkpoints().latest(id).await.unwrap().unwrap();
    GraphRun::from_checkpoint(&cp).unwrap()
}

/// Ticks until the run is done, at most 20 ticks. Returns the final run.
pub async fn finish(troupe: &Troupe, io: Arc<SimpleGraphIo>, id: RunId) -> GraphRun {
    let runner = troupe.runner(io);
    for _ in 0..20 {
        runner.tick(TickOptions::default()).await.unwrap();
        let run = saved(troupe, id).await;
        if run.status() == &GraphStatus::Done {
            return run;
        }
    }
    panic!("run {id} not done after 20 ticks");
}
