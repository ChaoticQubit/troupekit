use serde_json::{json, Map, Value};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use troupekit_agent::{Limits, MemorySource, PersonaLoader, SimpleIo};
use troupekit_core::clock::FixedClock;
use troupekit_core::store::Checkpoint;
use troupekit_graph::testing::{text_reply, FnModel, FnTool};
use troupekit_graph::{Conditions, GraphOutcome, GraphRun, SimpleGraphIo, Stop, Workflow};
use troupekit_providers::Price;
use troupekit_tools::{ToolError, ToolOutput, ToolRegistry};

const PRICE: Price = Price {
    input_usd_micros_per_mtok: 1_000_000,
    output_usd_micros_per_mtok: 2_000_000,
    verified: true,
    long_context: None,
    search_usd_micros: 0,
};

const LINEAR: &str = r#"
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

const APPROVAL: &str = r#"
name = "approve"
start = "plan"
[state]
goal = "string"
plan = "{steps: list}"
approved = "bool"
note = "string"
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
prompt = "Approve?"
reads = ["plan"]
writes = "approved"
[[nodes]]
id = "route"
kind = "router"
condition = "flag(approved)"
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
to = "approve"
[[edges]]
from = "approve"
to = "route"
[[edges]]
from = "route"
to = "save"
when = "true"
[[edges]]
from = "route"
to = "plan"
when = "false"
[[edges]]
from = "save"
to = "done"
"#;

const FAN: &str = r#"
name = "fan"
start = "seed"
[state]
goal = "string"
seed = "string"
answer = "string"
note = "string"
[[nodes]]
id = "seed"
kind = "tool"
tool = "seed"
reads = ["goal"]
writes = "seed"
[[nodes]]
id = "ask"
kind = "human"
ask = "question"
prompt = "Anything to add?"
reads = ["seed"]
writes = "answer"
[[nodes]]
id = "work"
kind = "tool"
tool = "work"
reads = ["seed"]
writes = "note"
[[nodes]]
id = "gather"
kind = "join"
[[nodes]]
id = "done"
kind = "end"
[[edges]]
from = "seed"
to = "ask"
[[edges]]
from = "seed"
to = "work"
[[edges]]
from = "ask"
to = "gather"
[[edges]]
from = "work"
to = "gather"
[[edges]]
from = "gather"
to = "done"
"#;

#[derive(Clone, Copy)]
enum Unblock {
    Nothing,
    Approve,
    RaiseBudget,
    Retry,
}

struct Golden {
    name: &'static str,
    unblock: Unblock,
}

const GOLDEN: [Golden; 6] = [
    Golden {
        name: "linear_between_steps",
        unblock: Unblock::Nothing,
    },
    Golden {
        name: "fan_out_left_and_tool_enter",
        unblock: Unblock::Nothing,
    },
    Golden {
        name: "approval_waiting",
        unblock: Unblock::Approve,
    },
    Golden {
        name: "agent_mid_run",
        unblock: Unblock::RaiseBudget,
    },
    Golden {
        name: "stopped",
        unblock: Unblock::Retry,
    },
    Golden {
        name: "failed_tool_with_call_key",
        unblock: Unblock::Retry,
    },
];

fn persona(name: &str, extra: &str) -> (String, String) {
    (
        format!("{name}.toml"),
        format!("name = '{name}'\ndisplay_name = 'P'\nrole = 'Works.'\ngoal = 'Done.'\ninstructions = 'Do it.'\nmodel = 'fake/m'\n{extra}"),
    )
}

fn tool(tools: &mut ToolRegistry, name: &str, reply: &'static str) {
    tools
        .register(FnTool::new(name, move |_, _| async move {
            Ok(ToolOutput::text(reply))
        }))
        .unwrap();
}

fn io(save_fails: usize, keys: Arc<Mutex<Vec<Option<String>>>>) -> SimpleGraphIo {
    let mut tools = ToolRegistry::new();
    tool(&mut tools, "seed", "seeded");
    tool(&mut tools, "work", "worked");
    let left = Arc::new(AtomicUsize::new(save_fails));
    tools
        .register(FnTool::new("save_plan", move |_, ctx| {
            let (keys, left) = (keys.clone(), left.clone());
            async move {
                keys.lock().unwrap().push(ctx.idempotency_key.clone());
                if left.load(Ordering::SeqCst) > 0 {
                    left.fetch_sub(1, Ordering::SeqCst);
                    return Err(ToolError::Failed("disk full".into()));
                }
                Ok(ToolOutput::text("saved"))
            }
        }))
        .unwrap();
    let model = Arc::new(FnModel::new("fake/m", |_| {
        text_reply(r#"{"steps": ["warm up", "run"]}"#, 10)
    }));
    let mut agent = SimpleIo::new(tools).with_model("fake/m", model, PRICE);
    agent.clock = Arc::new(FixedClock::new(1_000_000));
    let personas = PersonaLoader::new(vec![Arc::new(MemorySource::new(
        "test",
        [persona("planner", "[output]\nsteps = 'list'")],
    ))]);
    let mut io = SimpleGraphIo::new(Arc::new(agent), personas);
    io.conditions = Conditions::new();
    io
}

fn goal() -> Map<String, Value> {
    [("goal".to_string(), json!("g"))].into_iter().collect()
}

async fn start(text: &str, limits: Limits, io: &SimpleGraphIo) -> GraphRun {
    let workflow = Workflow::parse("golden/wf.toml", text).unwrap();
    GraphRun::new(workflow, goal(), limits, io).await.unwrap()
}

async fn drive(run: &mut GraphRun, io: &SimpleGraphIo) -> GraphOutcome {
    for _ in 0..100 {
        let out = run.step(io).await;
        if out != GraphOutcome::Continue {
            return out;
        }
    }
    panic!("run did not settle in 100 steps");
}

fn branches(cp: &Checkpoint) -> Vec<(String, String)> {
    cp.state["branches"]
        .as_array()
        .expect("a graph run holds branches")
        .iter()
        .map(|b| {
            (
                b["node"].as_str().unwrap().to_string(),
                b["at"]["at"].as_str().unwrap().to_string(),
            )
        })
        .collect()
}

fn pair(node: &str, at: &str) -> (String, String) {
    (node.to_string(), at.to_string())
}

async fn capture(name: &str) -> Checkpoint {
    let keys = Arc::new(Mutex::new(Vec::new()));
    let plain = io(0, keys.clone());
    match name {
        "linear_between_steps" => {
            let mut run = start(LINEAR, Limits::default(), &plain).await;
            for _ in 0..10 {
                assert_eq!(run.step(&plain).await, GraphOutcome::Continue);
                let cp = run.checkpoint();
                if branches(&cp) == vec![pair("save", "enter")] {
                    return cp;
                }
            }
            panic!("the linear run never reached the tool `save`");
        }
        "fan_out_left_and_tool_enter" => {
            let mut run = start(FAN, Limits::default(), &plain).await;
            assert_eq!(run.step(&plain).await, GraphOutcome::Continue);
            run.resume("ask", json!("more")).unwrap();
            let cp = run.checkpoint();
            assert_eq!(
                branches(&cp),
                vec![pair("ask", "left"), pair("work", "enter")]
            );
            cp
        }
        "approval_waiting" => {
            let mut run = start(APPROVAL, Limits::default(), &plain).await;
            assert!(matches!(
                drive(&mut run, &plain).await,
                GraphOutcome::Waiting(_)
            ));
            let cp = run.checkpoint();
            assert_eq!(branches(&cp), vec![pair("approve", "human")]);
            assert_eq!(cp.state["status"]["status"], "waiting_human");
            cp
        }
        "agent_mid_run" => {
            let tight = Limits {
                run: Some(1),
                ..Limits::default()
            };
            let mut run = start(LINEAR, tight, &plain).await;
            assert!(matches!(run.step(&plain).await, GraphOutcome::Waiting(_)));
            let cp = run.checkpoint();
            assert_eq!(branches(&cp), vec![pair("plan", "agent")]);
            cp
        }
        "stopped" => {
            let mut run = start(LINEAR, Limits::default(), &plain).await;
            assert_eq!(run.step(&plain).await, GraphOutcome::Continue);
            run.stop(Stop::Cancelled).unwrap();
            let cp = run.checkpoint();
            assert_eq!(cp.state["stopped"], json!({"kind": "cancelled"}));
            cp
        }
        "failed_tool_with_call_key" => {
            let flaky = io(2, keys.clone());
            let mut run = start(LINEAR, Limits::default(), &flaky).await;
            assert!(matches!(
                drive(&mut run, &flaky).await,
                GraphOutcome::Failed(_)
            ));
            run.retry(None).unwrap();
            assert!(matches!(
                drive(&mut run, &flaky).await,
                GraphOutcome::Failed(_)
            ));
            let cp = run.checkpoint();
            assert_eq!(branches(&cp), vec![pair("save", "enter")]);
            assert!(cp.state["branches"][0]["call"].is_object());
            assert_eq!(cp.state["failure"], "tool");
            let keys = keys.lock().unwrap();
            assert_eq!(keys.len(), 2);
            assert_eq!(keys[0], keys[1]);
            cp
        }
        other => panic!("no golden scenario `{other}`"),
    }
}

fn fixture_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/v1")
        .join(format!("{name}.json"))
}

fn blessing() -> bool {
    std::env::var_os("TROUPEKIT_BLESS").is_some()
}

async fn fixture(name: &str, bless: bool) -> Checkpoint {
    let path = fixture_path(name);
    if bless {
        let cp = capture(name).await;
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let text = serde_json::to_string_pretty(&cp).unwrap();
        std::fs::write(&path, format!("{text}\n")).unwrap();
        return cp;
    }
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "{}: {e}; bless it with TROUPEKIT_BLESS=1 at a commit whose run.rs is unedited since 2343a4f",
            path.display()
        )
    });
    serde_json::from_str(&text).unwrap()
}

#[tokio::test]
async fn golden_v1_checkpoints_restore_and_reserialize_identically() {
    for g in &GOLDEN {
        let cp = fixture(g.name, blessing()).await;
        assert_eq!(cp.state["format"], 1, "{}", g.name);
        let run = GraphRun::from_checkpoint(&cp).unwrap_or_else(|e| panic!("{}: {e}", g.name));
        let again = run.checkpoint();
        assert_eq!(again.run_id, cp.run_id, "{}", g.name);
        assert_eq!(again.step, cp.step, "{}", g.name);
        assert_eq!(again.state, cp.state, "{}", g.name);
    }
}

#[tokio::test]
async fn golden_v1_checkpoints_continue_to_done() {
    if blessing() {
        return;
    }
    for g in &GOLDEN {
        let cp = fixture(g.name, false).await;
        let keys = Arc::new(Mutex::new(Vec::new()));
        let io = io(0, keys.clone());
        let mut run = GraphRun::from_checkpoint(&cp).unwrap();
        match g.unblock {
            Unblock::Nothing => {}
            Unblock::Approve => run.resume("approve", json!(true)).unwrap(),
            Unblock::RaiseBudget => run.raise_budget(Limits::default()).unwrap(),
            Unblock::Retry => run.retry(None).unwrap(),
        }
        let out = drive(&mut run, &io).await;
        assert!(matches!(out, GraphOutcome::Done(_)), "{}: {out:?}", g.name);
        if g.name == "failed_tool_with_call_key" {
            let call = &cp.state["branches"][0]["call"];
            let key = format!("{}:{}:{}", cp.run_id, call["step"], call["index"]);
            assert_eq!(keys.lock().unwrap().as_slice(), &[Some(key)], "{}", g.name);
        }
    }
}
