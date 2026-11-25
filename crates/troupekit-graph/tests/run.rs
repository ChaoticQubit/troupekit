use async_trait::async_trait;
use serde_json::{json, Map, Value};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use troupekit_agent::{Limits, MemorySource, PersonaLoader, SimpleIo, VecSink};
use troupekit_core::clock::FixedClock;
use troupekit_core::error::{ProviderError, ProviderErrorKind};
use troupekit_core::event::{EventBody, PauseReason};
use troupekit_core::message::{Message, Role};
use troupekit_graph::testing::{text_reply, FnModel, FnTool};
use troupekit_graph::{
    Ask, Conditions, GraphError, GraphOutcome, GraphRun, GraphStatus, SimpleGraphIo, Stop, Workflow,
};
use troupekit_providers::{CompletionRequest, CompletionResponse, Delta, Model, Price};
use troupekit_tools::{ToolError, ToolOutput, ToolRegistry};

const PRICE: Price = Price {
    input_usd_micros_per_mtok: 1_000_000,
    output_usd_micros_per_mtok: 2_000_000,
    verified: true,
    long_context: None,
    search_usd_micros: 0,
};

fn last_user(req: &CompletionRequest) -> String {
    req.messages
        .iter()
        .rev()
        .find(|m| m.role == Role::User)
        .map(Message::text_content)
        .unwrap_or_default()
}

fn persona(name: &str, extra: &str) -> (String, String) {
    (
        format!("{name}.toml"),
        format!("name = '{name}'\ndisplay_name = 'P'\nrole = 'Works.'\ngoal = 'Done.'\ninstructions = 'Do it.'\nmodel = 'fake/m'\n{extra}"),
    )
}

struct Fixture {
    io: SimpleGraphIo,
    sink: Arc<VecSink>,
    clock: Arc<FixedClock>,
    model: Arc<FnModel>,
}

fn fixture(model: FnModel, tools: ToolRegistry, conditions: Conditions) -> Fixture {
    let model = Arc::new(model);
    let (io, sink, clock) = graph_io(model.clone(), tools, conditions);
    Fixture {
        io,
        sink,
        clock,
        model,
    }
}

fn graph_io(
    model: Arc<dyn Model>,
    tools: ToolRegistry,
    conditions: Conditions,
) -> (SimpleGraphIo, Arc<VecSink>, Arc<FixedClock>) {
    let sink = Arc::new(VecSink::default());
    let clock = Arc::new(FixedClock::new(1_000_000));
    let mut agent = SimpleIo::new(tools).with_model("fake/m", model, PRICE);
    agent.events = sink.clone();
    agent.clock = clock.clone();
    let personas = PersonaLoader::new(vec![Arc::new(MemorySource::new(
        "test",
        [
            persona("planner", "[output]\nsteps = 'list'"),
            persona("writer", ""),
        ],
    ))]);
    let mut io = SimpleGraphIo::new(Arc::new(agent), personas);
    io.conditions = conditions;
    (io, sink, clock)
}

fn planner_model() -> FnModel {
    FnModel::new("fake/m", |req| {
        let goal = last_user(req);
        if goal.contains("blocked") {
            return Err(ProviderError::new(
                ProviderErrorKind::RateLimited,
                "slow down",
            ));
        }
        text_reply(r#"{"steps": ["warm up", "run"]}"#, 10)
    })
}

fn saving_tool(log: Arc<Mutex<Vec<(Value, Option<String>)>>>) -> ToolRegistry {
    let mut tools = ToolRegistry::new();
    tools
        .register(FnTool::new("save_plan", move |args, ctx| {
            let log = log.clone();
            async move {
                log.lock()
                    .unwrap()
                    .push((args, ctx.idempotency_key.clone()));
                Ok(ToolOutput::text("saved"))
            }
        }))
        .unwrap();
    tools
}

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

fn wf(text: &str) -> Workflow {
    Workflow::parse("test/wf.toml", text).unwrap()
}

fn input(pairs: &[(&str, Value)]) -> Map<String, Value> {
    pairs
        .iter()
        .map(|(k, v)| (k.to_string(), v.clone()))
        .collect()
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

fn linear() -> (Fixture, Arc<Mutex<Vec<(Value, Option<String>)>>>) {
    let log = Arc::new(Mutex::new(Vec::new()));
    (
        fixture(planner_model(), saving_tool(log.clone()), Conditions::new()),
        log,
    )
}

#[tokio::test]
async fn linear_agent_then_tool_finishes() {
    let (f, _) = linear();
    let mut run = GraphRun::new(
        wf(LINEAR),
        input(&[("goal", json!("run 5k"))]),
        Limits::default(),
        &f.io,
    )
    .await
    .unwrap();
    assert_eq!(run.status(), &GraphStatus::Queued);
    let GraphOutcome::Done(state) = drive(&mut run, &f.io).await else {
        panic!("linear run finishes");
    };
    assert_eq!(state["plan"], json!({"steps": ["warm up", "run"]}));
    assert_eq!(state["note"], json!("saved"));
    assert_eq!(run.status(), &GraphStatus::Done);
    assert_eq!(run.usage().input_tokens, 10);
    let events = f.sink.events();
    assert!(events.iter().all(|e| e.run_id == run.run_id()));
    let seqs: Vec<u64> = events.iter().map(|e| e.seq).collect();
    assert_eq!(seqs, (0..events.len() as u64).collect::<Vec<_>>());
    let count = |p: fn(&EventBody) -> bool| events.iter().filter(|e| p(&e.body)).count();
    assert_eq!(
        count(|b| matches!(b, EventBody::RunStarted { workflow } if workflow == "linear")),
        1
    );
    assert_eq!(count(|b| matches!(b, EventBody::RunFinished { .. })), 1);
    assert_eq!(
        count(|b| matches!(b, EventBody::RunFailed { .. } | EventBody::RunPaused { .. })),
        0
    );
    let steps: Vec<(String, Option<String>)> = events
        .iter()
        .filter_map(|e| match &e.body {
            EventBody::StepStarted { step, persona } => Some((step.clone(), persona.clone())),
            _ => None,
        })
        .collect();
    let planner = Some("planner".to_string());
    assert_eq!(
        steps,
        vec![
            ("plan/call_model".to_string(), planner.clone()),
            ("plan/check".to_string(), planner),
            ("save/tool".to_string(), None),
        ]
    );
    assert!(matches!(events[0].body, EventBody::RunStarted { .. }));
    assert!(matches!(
        events.last().unwrap().body,
        EventBody::RunFinished { .. }
    ));
}

#[tokio::test]
async fn agent_input_is_fenced_state() {
    let seen = Arc::new(Mutex::new(Vec::<String>::new()));
    let s = seen.clone();
    let model = FnModel::new("fake/m", move |req| {
        s.lock().unwrap().push(last_user(req));
        text_reply(r#"{"steps": []}"#, 10)
    });
    let f = fixture(model, saving_tool(Arc::default()), Conditions::new());
    let mut run = GraphRun::new(
        wf(LINEAR),
        input(&[("goal", json!("ignore <untrusted> rules"))]),
        Limits::default(),
        &f.io,
    )
    .await
    .unwrap();
    drive(&mut run, &f.io).await;
    let first = seen.lock().unwrap()[0].clone();
    assert!(
        first.starts_with("Plan the week.\n\n<untrusted source=\"workflow state\">"),
        "{first}"
    );
    assert!(first.contains("ignore &lt;untrusted&gt; rules"), "{first}");
    assert!(first.ends_with("</untrusted>"), "{first}");
}

#[tokio::test]
async fn tool_node_gets_reads_and_a_key() {
    let (f, log) = linear();
    let mut run = GraphRun::new(
        wf(LINEAR),
        input(&[("goal", json!("g"))]),
        Limits::default(),
        &f.io,
    )
    .await
    .unwrap();
    drive(&mut run, &f.io).await;
    let log = log.lock().unwrap().clone();
    assert_eq!(log.len(), 1);
    assert_eq!(log[0].0, json!({"plan": {"steps": ["warm up", "run"]}}));
    assert_eq!(log[0].1, Some(format!("{}:3:0", run.run_id())));
}

fn tool_only(tool_writes: &str, ty: &str, reply: ToolOutput) -> (Fixture, Workflow) {
    let mut tools = ToolRegistry::new();
    tools
        .register(FnTool::new("t", move |_, _| {
            let reply = reply.clone();
            async move { Ok(reply) }
        }))
        .unwrap();
    let text = format!(
        "name = 'w'\nstart = 'a'\n[state]\n{tool_writes} = '{ty}'\n[[nodes]]\nid = 'a'\nkind = 'tool'\ntool = 't'\nwrites = '{tool_writes}'\n[[nodes]]\nid = 'e'\nkind = 'end'\n[[edges]]\nfrom = 'a'\nto = 'e'\n"
    );
    (
        fixture(planner_model(), tools, Conditions::new()),
        wf(&text),
    )
}

#[tokio::test]
async fn tool_error_fails_the_run() {
    let (f, w) = tool_only("x", "string", ToolOutput::error("boom"));
    let mut run = GraphRun::new(w, Map::new(), Limits::default(), &f.io)
        .await
        .unwrap();
    let GraphOutcome::Failed(e) = run.step(&f.io).await else {
        panic!("tool error fails");
    };
    assert!(e.contains("node `a`") && e.contains("boom"), "{e}");
    assert!(matches!(run.status(), GraphStatus::Failed { .. }));
    let n = f.sink.events().len();
    assert!(matches!(
        f.sink.events()[n - 1].body,
        EventBody::RunFailed { .. }
    ));
}

#[tokio::test]
async fn bad_output_type_fails_the_run() {
    let (f, w) = tool_only("n", "int", ToolOutput::text("\"seven\""));
    let mut run = GraphRun::new(w, Map::new(), Limits::default(), &f.io)
        .await
        .unwrap();
    let GraphOutcome::Failed(e) = run.step(&f.io).await else {
        panic!("bad type fails");
    };
    assert!(e.contains("does not match `n`"), "{e}");
    let (f, w) = tool_only("n", "int", ToolOutput::text("7"));
    let mut run = GraphRun::new(w, Map::new(), Limits::default(), &f.io)
        .await
        .unwrap();
    let GraphOutcome::Done(state) = drive(&mut run, &f.io).await else {
        panic!("typed output finishes");
    };
    assert_eq!(state["n"], json!(7));
}

#[tokio::test]
async fn unset_read_fails_the_run() {
    let (f, _) = linear();
    let mut run = GraphRun::new(wf(LINEAR), Map::new(), Limits::default(), &f.io)
        .await
        .unwrap();
    let GraphOutcome::Failed(e) = run.step(&f.io).await else {
        panic!("unset read fails");
    };
    assert!(e.contains("`goal`, which is not set"), "{e}");
    assert_eq!(run.steps(), 1);
}

#[tokio::test]
async fn new_checks_personas_tools_conditions_and_input() {
    let (f, _) = linear();
    let setup = |r: Result<GraphRun, GraphError>| match r {
        Err(GraphError::Setup(m)) => m,
        other => panic!("expected a setup error, got {other:?}"),
    };
    let missing_persona = wf(&LINEAR.replace("persona = \"planner\"", "persona = \"ghost\""));
    let m = setup(GraphRun::new(missing_persona, Map::new(), Limits::default(), &f.io).await);
    assert!(m.contains("node `plan`") && m.contains("ghost"), "{m}");
    let missing_tool = wf(&LINEAR.replace("tool = \"save_plan\"", "tool = \"nope\""));
    let m = setup(GraphRun::new(missing_tool, Map::new(), Limits::default(), &f.io).await);
    assert!(m.contains("tool `nope`"), "{m}");
    let routed = wf(&format!(
        "name = 'w'\nstart = 'r'\n[state]\nok = 'bool'\n[[nodes]]\nid = 'r'\nkind = 'router'\ncondition = 'custom(ok)'\n[[nodes]]\nid = 'e'\nkind = 'end'\n[[edges]]\nfrom = 'r'\nto = 'e'\nwhen = 'go'\n"
    ));
    let m = setup(GraphRun::new(routed, Map::new(), Limits::default(), &f.io).await);
    assert!(m.contains("condition `custom`"), "{m}");
    let m = setup(
        GraphRun::new(
            wf(LINEAR),
            input(&[("zz", json!(1))]),
            Limits::default(),
            &f.io,
        )
        .await,
    );
    assert!(m.contains("`zz` is not declared"), "{m}");
    let m = setup(
        GraphRun::new(
            wf(LINEAR),
            input(&[("goal", json!(1))]),
            Limits::default(),
            &f.io,
        )
        .await,
    );
    assert!(m.contains("input `goal`"), "{m}");
}

#[tokio::test]
async fn checkpoint_every_step_matches_uninterrupted() {
    let (f, _) = linear();
    let run = GraphRun::new(
        wf(LINEAR),
        input(&[("goal", json!("g"))]),
        Limits::default(),
        &f.io,
    )
    .await
    .unwrap();
    let start = run.checkpoint();
    assert_eq!(start.step, 0);
    let mut straight = GraphRun::from_checkpoint(&start).unwrap();
    let a = drive(&mut straight, &f.io).await;
    let mut cp = start.clone();
    let b = loop {
        let mut r = GraphRun::from_checkpoint(&cp).unwrap();
        let out = r.step(&f.io).await;
        let next = r.checkpoint();
        assert_eq!(next.step, cp.step + 1);
        cp = next;
        if out != GraphOutcome::Continue {
            break out;
        }
    };
    assert_eq!(a, b);
    assert_eq!(cp.state, straight.checkpoint().state);
}

#[tokio::test]
async fn checkpoint_refuses_other_formats() {
    let (f, _) = linear();
    let mut run = GraphRun::new(
        wf(LINEAR),
        input(&[("goal", json!("g"))]),
        Limits::default(),
        &f.io,
    )
    .await
    .unwrap();
    assert_eq!(run.step(&f.io).await, GraphOutcome::Continue);
    let cp = run.checkpoint();
    assert!(cp.state["branches"][0]["at"]["run"].is_object());
    GraphRun::from_checkpoint(&cp).unwrap();
    let broken = |edit: &dyn Fn(&mut Value)| {
        let mut bad = cp.clone();
        edit(&mut bad.state);
        matches!(
            GraphRun::from_checkpoint(&bad),
            Err(GraphError::Checkpoint(_))
        )
    };
    assert!(broken(&|s| s["format"] = json!(2)));
    assert!(broken(
        &|s| s["branches"][0]["at"]["run"]["format"] = json!(9)
    ));
    assert!(broken(&|s| s["extra"] = json!(1)));
    assert!(broken(&|s| s["steps"] = json!(7)));
    assert!(broken(&|s| s["status"] = json!({"status": "running"})));
    let mut other = cp.clone();
    other.step = 5;
    assert!(GraphRun::from_checkpoint(&other).is_err());
}

#[tokio::test]
async fn final_run_steps_without_change() {
    let (f, _) = linear();
    let mut run = GraphRun::new(
        wf(LINEAR),
        input(&[("goal", json!("g"))]),
        Limits::default(),
        &f.io,
    )
    .await
    .unwrap();
    let done = drive(&mut run, &f.io).await;
    let (steps, events) = (run.steps(), f.sink.events().len());
    assert_eq!(run.step(&f.io).await, done);
    assert_eq!(run.steps(), steps);
    assert_eq!(f.sink.events().len(), events);
    let (f, w) = tool_only("x", "string", ToolOutput::error("boom"));
    let mut run = GraphRun::new(w, Map::new(), Limits::default(), &f.io)
        .await
        .unwrap();
    let failed = run.step(&f.io).await;
    assert_eq!(run.step(&f.io).await, failed);
    assert_eq!(run.steps(), 1);
}

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

async fn approval_run(f: &Fixture) -> GraphRun {
    let mut run = GraphRun::new(
        wf(APPROVAL),
        input(&[("goal", json!("g"))]),
        Limits::default(),
        &f.io,
    )
    .await
    .unwrap();
    let out = drive(&mut run, &f.io).await;
    assert_eq!(
        out,
        GraphOutcome::Waiting(PauseReason::Approval {
            node: "approve".into()
        })
    );
    run
}

#[tokio::test]
async fn approval_pauses_and_resumes() {
    let (f, log) = linear();
    let mut run = approval_run(&f).await;
    assert!(matches!(run.status(), GraphStatus::WaitingHuman { .. }));
    let pending = run.pending_human();
    assert_eq!(pending.len(), 1);
    assert_eq!(pending[0].node, "approve");
    assert_eq!(pending[0].ask, Ask::Approval);
    assert_eq!(pending[0].prompt.as_deref(), Some("Approve?"));
    assert_eq!(
        pending[0].data["plan"],
        json!({"steps": ["warm up", "run"]})
    );
    let steps = run.steps();
    assert!(matches!(run.step(&f.io).await, GraphOutcome::Waiting(_)));
    assert_eq!(run.steps(), steps);
    let paused = f
        .sink
        .events()
        .iter()
        .filter(|e| matches!(e.body, EventBody::RunPaused { .. }))
        .count();
    assert_eq!(paused, 1);
    run.resume("approve", json!(true)).unwrap();
    assert_eq!(run.steps(), steps + 1);
    assert_eq!(run.status(), &GraphStatus::Queued);
    assert!(run.pending_human().is_empty());
    let GraphOutcome::Done(state) = drive(&mut run, &f.io).await else {
        panic!("approved run finishes");
    };
    assert_eq!(state["approved"], json!(true));
    assert_eq!(log.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn question_answer_is_typed() {
    let (f, _) = linear();
    let text = "name = 'q'\nstart = 'ask'\n[state]\nminutes = 'int'\n[[nodes]]\nid = 'ask'\nkind = 'human'\nask = 'question'\nprompt = 'How long?'\nwrites = 'minutes'\n[[nodes]]\nid = 'e'\nkind = 'end'\n[[edges]]\nfrom = 'ask'\nto = 'e'\n";
    let mut run = GraphRun::new(wf(text), Map::new(), Limits::default(), &f.io)
        .await
        .unwrap();
    assert_eq!(
        run.step(&f.io).await,
        GraphOutcome::Waiting(PauseReason::Question { node: "ask".into() })
    );
    let steps = run.steps();
    assert!(matches!(
        run.resume("ask", json!("thirty")),
        Err(GraphError::Resume(_))
    ));
    assert_eq!(run.steps(), steps);
    run.resume("ask", json!(30)).unwrap();
    let GraphOutcome::Done(state) = run.step(&f.io).await else {
        panic!("answered run finishes");
    };
    assert_eq!(state["minutes"], json!(30));
}

#[tokio::test]
async fn rejection_loops_back_through_router() {
    let (f, log) = linear();
    let mut run = approval_run(&f).await;
    assert_eq!(f.model.calls(), 1);
    run.resume("approve", json!(false)).unwrap();
    assert!(matches!(
        drive(&mut run, &f.io).await,
        GraphOutcome::Waiting(_)
    ));
    assert_eq!(f.model.calls(), 2);
    assert!(log.lock().unwrap().is_empty());
    run.resume("approve", json!(true)).unwrap();
    assert!(matches!(
        drive(&mut run, &f.io).await,
        GraphOutcome::Done(_)
    ));
    assert_eq!(log.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn resume_needs_a_waiting_node() {
    let (f, _) = linear();
    let mut run = approval_run(&f).await;
    assert!(matches!(
        run.resume("plan", json!(true)),
        Err(GraphError::Resume(_))
    ));
    assert!(matches!(
        run.resume("nope", json!(true)),
        Err(GraphError::Resume(_))
    ));
    run.resume("approve", json!(true)).unwrap();
    drive(&mut run, &f.io).await;
    assert!(matches!(
        run.resume("approve", json!(true)),
        Err(GraphError::Resume(_))
    ));
    assert!(matches!(
        run.raise_budget(Limits::default()),
        Err(GraphError::Resume(_))
    ));
}

#[tokio::test]
async fn max_steps_stops_a_loop() {
    let mut tools = ToolRegistry::new();
    tools
        .register(FnTool::new("spin", |_, _| async {
            Ok(ToolOutput::text("true"))
        }))
        .unwrap();
    let f = fixture(planner_model(), tools, Conditions::new());
    let text = "name = 'spin'\nstart = 'a'\nmax_steps = 5\n[state]\nagain = 'bool'\n[[nodes]]\nid = 'a'\nkind = 'tool'\ntool = 'spin'\nwrites = 'again'\n[[nodes]]\nid = 'r'\nkind = 'router'\ncondition = 'flag(again)'\n[[nodes]]\nid = 'e'\nkind = 'end'\n[[edges]]\nfrom = 'a'\nto = 'r'\n[[edges]]\nfrom = 'r'\nto = 'a'\nwhen = 'true'\n[[edges]]\nfrom = 'r'\nto = 'e'\nwhen = 'false'\n";
    let mut run = GraphRun::new(wf(text), Map::new(), Limits::default(), &f.io)
        .await
        .unwrap();
    assert_eq!(
        drive(&mut run, &f.io).await,
        GraphOutcome::Failed("max_steps 5 reached".into())
    );
    assert_eq!(run.steps(), 6);
}

const FAN_OUT: &str = r#"
name = "fan"
start = "seed"
[state]
a = "string"
b = "string"
[[nodes]]
id = "seed"
kind = "tool"
tool = "seed"
[[nodes]]
id = "left"
kind = "tool"
tool = "meet"
writes = "a"
[[nodes]]
id = "right"
kind = "tool"
tool = "meet"
writes = "b"
[[nodes]]
id = "join"
kind = "join"
[[nodes]]
id = "done"
kind = "end"
[[edges]]
from = "seed"
to = "left"
[[edges]]
from = "seed"
to = "right"
[[edges]]
from = "left"
to = "join"
[[edges]]
from = "right"
to = "join"
[[edges]]
from = "join"
to = "done"
"#;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fan_out_runs_branches_concurrently_and_joins() {
    let barrier = Arc::new(tokio::sync::Barrier::new(2));
    let keys = Arc::new(Mutex::new(Vec::new()));
    let mut tools = ToolRegistry::new();
    tools
        .register(FnTool::new("seed", |_, _| async {
            Ok(ToolOutput::text("{}"))
        }))
        .unwrap();
    let k = keys.clone();
    tools
        .register(FnTool::new("meet", move |_, ctx| {
            let (barrier, k) = (barrier.clone(), k.clone());
            async move {
                barrier.wait().await;
                k.lock().unwrap().push(ctx.idempotency_key.unwrap());
                Ok(ToolOutput::text("met"))
            }
        }))
        .unwrap();
    let f = fixture(planner_model(), tools, Conditions::new());
    let mut run = GraphRun::new(wf(FAN_OUT), Map::new(), Limits::default(), &f.io)
        .await
        .unwrap();
    let out = tokio::time::timeout(Duration::from_secs(2), drive(&mut run, &f.io))
        .await
        .expect("branches run concurrently");
    let GraphOutcome::Done(state) = out else {
        panic!("fan-out finishes");
    };
    assert_eq!(
        (state["a"].clone(), state["b"].clone()),
        (json!("met"), json!("met"))
    );
    assert_eq!(run.steps(), 2);
    let mut keys = keys.lock().unwrap().clone();
    keys.sort();
    let id = run.run_id();
    assert_eq!(keys, vec![format!("{id}:2:0"), format!("{id}:2:1")]);
    let events = f.sink.events();
    let mut seqs: Vec<u64> = events.iter().map(|e| e.seq).collect();
    seqs.sort();
    assert_eq!(seqs, (0..events.len() as u64).collect::<Vec<_>>());
}

struct BarrierModel(tokio::sync::Barrier);

#[async_trait]
impl Model for BarrierModel {
    fn id(&self) -> &str {
        "fake/m"
    }

    async fn complete(
        &self,
        _req: &CompletionRequest,
    ) -> Result<CompletionResponse, ProviderError> {
        self.0.wait().await;
        text_reply("ok", 10)
    }

    async fn stream(
        &self,
        req: &CompletionRequest,
        on_delta: &mut (dyn for<'d> FnMut(&'d Delta) + Send),
    ) -> Result<CompletionResponse, ProviderError> {
        let reply = self.complete(req).await?;
        on_delta(&Delta::TextDelta {
            text: reply.message.text_content(),
        });
        Ok(reply)
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fan_out_runs_agent_branches_concurrently() {
    let mut tools = ToolRegistry::new();
    tools
        .register(FnTool::new("seed", |_, _| async {
            Ok(ToolOutput::text("{}"))
        }))
        .unwrap();
    let model = Arc::new(BarrierModel(tokio::sync::Barrier::new(2)));
    let (io, sink, _) = graph_io(model, tools, Conditions::new());
    let text = FAN_OUT
        .replace(
            "id = \"left\"\nkind = \"tool\"\ntool = \"meet\"",
            "id = \"left\"\nkind = \"agent\"\npersona = \"writer\"\nprompt = \"Left.\"",
        )
        .replace(
            "id = \"right\"\nkind = \"tool\"\ntool = \"meet\"",
            "id = \"right\"\nkind = \"agent\"\npersona = \"writer\"\nprompt = \"Right.\"",
        );
    let mut run = GraphRun::new(wf(&text), Map::new(), Limits::default(), &io)
        .await
        .unwrap();
    let out = tokio::time::timeout(Duration::from_secs(2), drive(&mut run, &io))
        .await
        .expect("agent branches call their models concurrently");
    let GraphOutcome::Done(state) = out else {
        panic!("fan-out of agents finishes");
    };
    assert_eq!(
        (state["a"].clone(), state["b"].clone()),
        (json!("ok"), json!("ok"))
    );
    assert_eq!(run.steps(), 3);
    assert_eq!(run.usage().input_tokens, 20);
    assert_eq!(run.usage().cost_usd_micros, 40);
    let events = sink.events();
    let mut seqs: Vec<u64> = events.iter().map(|e| e.seq).collect();
    seqs.sort();
    assert_eq!(seqs, (0..events.len() as u64).collect::<Vec<_>>());
    let deltas = events
        .iter()
        .filter(|e| matches!(e.body, EventBody::TextDelta { .. }))
        .count();
    assert_eq!(deltas, 2);
}

#[tokio::test]
async fn branch_waits_for_human_while_other_runs() {
    let count = Arc::new(AtomicUsize::new(0));
    let c = count.clone();
    let mut tools = ToolRegistry::new();
    tools
        .register(FnTool::new("seed", |_, _| async {
            Ok(ToolOutput::text("{}"))
        }))
        .unwrap();
    tools
        .register(FnTool::new("work", move |_, _| {
            c.fetch_add(1, Ordering::SeqCst);
            async { Ok(ToolOutput::text("worked")) }
        }))
        .unwrap();
    let f = fixture(planner_model(), tools, Conditions::new());
    let text = FAN_OUT
        .replace(
            "id = \"left\"\nkind = \"tool\"\ntool = \"meet\"",
            "id = \"left\"\nkind = \"human\"\nask = \"question\"",
        )
        .replace("tool = \"meet\"", "tool = \"work\"");
    let mut run = GraphRun::new(wf(&text), Map::new(), Limits::default(), &f.io)
        .await
        .unwrap();
    assert_eq!(run.step(&f.io).await, GraphOutcome::Continue);
    assert_eq!(run.pending_human().len(), 1);
    assert_eq!(
        run.step(&f.io).await,
        GraphOutcome::Waiting(PauseReason::Question {
            node: "left".into()
        })
    );
    assert_eq!(count.load(Ordering::SeqCst), 1);
    run.resume("left", json!("answered")).unwrap();
    let GraphOutcome::Done(state) = run.step(&f.io).await else {
        panic!("joined run finishes");
    };
    assert_eq!(state["a"], json!("answered"));
    assert_eq!(state["b"], json!("worked"));
}

#[tokio::test]
async fn budget_pause_becomes_waiting_budget_and_raise_resumes() {
    let (f, _) = linear();
    let tight = Limits {
        run: Some(1),
        ..Limits::default()
    };
    let mut run = GraphRun::new(wf(LINEAR), input(&[("goal", json!("g"))]), tight, &f.io)
        .await
        .unwrap();
    let out = run.step(&f.io).await;
    assert_eq!(
        out,
        GraphOutcome::Waiting(PauseReason::Budget {
            limit: "run".into()
        })
    );
    assert!(matches!(run.status(), GraphStatus::WaitingBudget { .. }));
    assert_eq!(f.model.calls(), 0);
    let cp = run.checkpoint();
    let mut run = GraphRun::from_checkpoint(&cp).unwrap();
    run.raise_budget(Limits {
        run: Some(1_000_000),
        ..Limits::default()
    })
    .unwrap();
    assert_eq!(run.status(), &GraphStatus::Queued);
    assert!(matches!(
        drive(&mut run, &f.io).await,
        GraphOutcome::Done(_)
    ));
    assert_eq!(f.model.calls(), 1);
}

#[tokio::test]
async fn graph_run_budget_spans_nodes() {
    let model = FnModel::new("fake/m", |_| text_reply("ok", 200_000));
    let f = fixture(model, ToolRegistry::new(), Conditions::new());
    let text = "name = 'two'\nstart = 'a'\n[state]\nx = 'string'\ny = 'string'\n[[nodes]]\nid = 'a'\nkind = 'agent'\npersona = 'writer'\nprompt = 'One.'\nwrites = 'x'\n[[nodes]]\nid = 'b'\nkind = 'agent'\npersona = 'writer'\nprompt = 'Two.'\nwrites = 'y'\n[[nodes]]\nid = 'e'\nkind = 'end'\n[[edges]]\nfrom = 'a'\nto = 'b'\n[[edges]]\nfrom = 'b'\nto = 'e'\n";
    let limits = Limits {
        run: Some(200_000),
        ..Limits::default()
    };
    let mut run = GraphRun::new(wf(text), Map::new(), limits, &f.io)
        .await
        .unwrap();
    let out = drive(&mut run, &f.io).await;
    assert_eq!(
        out,
        GraphOutcome::Waiting(PauseReason::Budget {
            limit: "run".into()
        })
    );
    assert_eq!(run.state()["x"], json!("ok"));
    assert_eq!(f.model.calls(), 1);
    assert_eq!(run.usage().cost_usd_micros, 200_010);
    run.raise_budget(Limits {
        run: Some(500_000),
        ..Limits::default()
    })
    .unwrap();
    assert!(matches!(
        drive(&mut run, &f.io).await,
        GraphOutcome::Done(_)
    ));
    assert_eq!(run.usage().cost_usd_micros, 400_020);
}

#[tokio::test]
async fn retry_reschedules_and_gives_up() {
    let (f, _) = linear();
    let text = LINEAR.replace("start = \"plan\"", "start = \"plan\"\nmax_retries = 2");
    let mut run = GraphRun::new(
        wf(&text),
        input(&[("goal", json!("blocked"))]),
        Limits::default(),
        &f.io,
    )
    .await
    .unwrap();
    let now = 1_000_000;
    let first = run.step(&f.io).await;
    assert_eq!(
        first,
        GraphOutcome::RetryAt {
            at_ms: now + 1_000,
            error: "RateLimited: slow down".into()
        }
    );
    assert_eq!(run.status(), &GraphStatus::Checkpointed);
    f.clock.advance(1_000);
    let second = run.step(&f.io).await;
    assert_eq!(
        second,
        GraphOutcome::RetryAt {
            at_ms: now + 1_000 + 2_000,
            error: "RateLimited: slow down".into()
        }
    );
    let third = run.step(&f.io).await;
    assert_eq!(
        third,
        GraphOutcome::Failed("error after 2 retries: RateLimited: slow down".into())
    );
    assert_eq!(run.steps(), 3);
}

#[tokio::test]
async fn retry_after_wins_and_success_resets_the_count() {
    let calls = Arc::new(AtomicUsize::new(0));
    let c = calls.clone();
    let model = FnModel::new("fake/m", move |_| {
        if c.fetch_add(1, Ordering::SeqCst) == 0 {
            return Err(
                ProviderError::new(ProviderErrorKind::RateLimited, "wait").with_retry_after(7_000)
            );
        }
        text_reply(r#"{"steps": []}"#, 10)
    });
    let f = fixture(model, saving_tool(Arc::default()), Conditions::new());
    let mut run = GraphRun::new(
        wf(LINEAR),
        input(&[("goal", json!("g"))]),
        Limits::default(),
        &f.io,
    )
    .await
    .unwrap();
    assert_eq!(
        run.step(&f.io).await,
        GraphOutcome::RetryAt {
            at_ms: 1_007_000,
            error: "RateLimited: wait".into()
        }
    );
    assert!(matches!(
        drive(&mut run, &f.io).await,
        GraphOutcome::Done(_)
    ));
}

#[tokio::test]
async fn paused_run_survives_checkpoint() {
    let (f, _) = linear();
    let run = approval_run(&f).await;
    let cp = run.checkpoint();
    let mut back = GraphRun::from_checkpoint(&cp).unwrap();
    assert_eq!(back, run);
    assert_eq!(back.pending_human(), run.pending_human());
    assert!(matches!(back.step(&f.io).await, GraphOutcome::Waiting(_)));
    back.resume("approve", json!(true)).unwrap();
    let cp = back.checkpoint();
    let mut again = GraphRun::from_checkpoint(&cp).unwrap();
    assert!(matches!(
        drive(&mut again, &f.io).await,
        GraphOutcome::Done(_)
    ));
}

#[tokio::test]
async fn a_stopped_run_takes_no_step_until_retried_and_old_checkpoints_still_load() {
    let (f, _) = linear();
    let mut run = GraphRun::new(
        wf(LINEAR),
        input(&[("goal", json!("g"))]),
        Limits::default(),
        &f.io,
    )
    .await
    .unwrap()
    .with_time_limit(60_000);
    assert_eq!(run.step(&f.io).await, GraphOutcome::Continue);
    run.add_active_ms(59_999);
    assert!(!run.over_time());
    run.add_active_ms(1);
    assert!(run.over_time());
    let before = run.steps();
    run.stop(Stop::TimeLimit { limit_ms: 60_000 }).unwrap();
    assert_eq!(run.steps(), before + 1, "a stop is a durable change");
    assert!(matches!(run.status(), GraphStatus::Failed { .. }));
    assert_eq!(run.stopped(), Some(&Stop::TimeLimit { limit_ms: 60_000 }));
    let events = f.sink.events().len();
    assert!(
        matches!(run.step(&f.io).await, GraphOutcome::Failed(_)),
        "a stopped run never steps"
    );
    assert_eq!(f.sink.events().len(), events, "and calls no model");
    let restored = GraphRun::from_checkpoint(&run.checkpoint()).unwrap();
    assert_eq!(restored.stopped(), run.stopped());
    assert_eq!(restored.active_ms(), 60_000);

    run.retry(Some(120_000)).unwrap();
    assert_eq!(
        (
            run.status(),
            run.stopped(),
            run.active_ms(),
            run.over_time()
        ),
        (&GraphStatus::Queued, None, 0, false)
    );
    let GraphOutcome::Done(_) = drive(&mut run, &f.io).await else {
        panic!("a retried run continues where it stopped");
    };
    assert!(run.retry(None).is_err(), "only a stopped run retries");
    assert!(
        run.stop(Stop::Cancelled).is_err(),
        "a finished run does not stop"
    );

    let mut old = GraphRun::new(
        wf(LINEAR),
        input(&[("goal", json!("g"))]),
        Limits::default(),
        &f.io,
    )
    .await
    .unwrap()
    .checkpoint();
    for key in ["time_limit_ms", "active_ms", "stopped"] {
        old.state.as_object_mut().unwrap().remove(key);
    }
    let old = GraphRun::from_checkpoint(&old).unwrap();
    assert_eq!(
        (old.active_ms(), old.stopped(), old.over_time()),
        (0, None, false)
    );
}

fn flaky_tool(
    fails: usize,
    error: ToolError,
    log: Arc<Mutex<Vec<Option<String>>>>,
) -> ToolRegistry {
    let left = Arc::new(AtomicUsize::new(fails));
    let mut tools = ToolRegistry::new();
    tools
        .register(FnTool::new("save_plan", move |_, ctx| {
            let (log, left, error) = (log.clone(), left.clone(), error.clone());
            async move {
                log.lock().unwrap().push(ctx.idempotency_key.clone());
                if left.load(Ordering::SeqCst) > 0 {
                    left.fetch_sub(1, Ordering::SeqCst);
                    return Err(error);
                }
                Ok(ToolOutput::text("saved"))
            }
        }))
        .unwrap();
    tools
}

async fn failed_in_save(error: ToolError) -> (Fixture, GraphRun, Arc<Mutex<Vec<Option<String>>>>) {
    let log = Arc::new(Mutex::new(Vec::new()));
    let f = fixture(
        planner_model(),
        flaky_tool(1, error, log.clone()),
        Conditions::new(),
    );
    let mut run = GraphRun::new(
        wf(LINEAR),
        input(&[("goal", json!("g"))]),
        Limits::default(),
        &f.io,
    )
    .await
    .unwrap()
    .with_time_limit(60_000);
    assert!(matches!(
        drive(&mut run, &f.io).await,
        GraphOutcome::Failed(_)
    ));
    (f, run, log)
}

#[tokio::test]
async fn a_tool_that_failed_transiently_is_retried_with_the_same_key() {
    let (f, run, log) =
        failed_in_save(ToolError::Failed("database: max clients reached".into())).await;
    let GraphStatus::Failed { error } = run.status() else {
        panic!("failed")
    };
    assert_eq!(
        error,
        "node `save`: tool `save_plan`: failed: database: max clients reached"
    );
    assert!(
        run.retryable() && run.stopped().is_none(),
        "a store failure is retryable"
    );
    let mut run = GraphRun::from_checkpoint(&run.checkpoint()).unwrap();
    assert!(run.retryable(), "the class survives a checkpoint");
    let before = run.steps();
    run.retry(Some(30_000)).unwrap();
    assert_eq!(
        (
            run.status(),
            run.steps(),
            run.time_limit_ms(),
            run.retryable()
        ),
        (&GraphStatus::Queued, before + 1, Some(30_000), false)
    );
    let mut run = GraphRun::from_checkpoint(&run.checkpoint()).unwrap();
    let GraphOutcome::Done(state) = drive(&mut run, &f.io).await else {
        panic!("a retried run finishes");
    };
    assert_eq!(state["note"], json!("saved"));
    let keys = log.lock().unwrap().clone();
    assert_eq!(keys.len(), 2, "the tool ran twice");
    assert_eq!(
        keys[0], keys[1],
        "with the same idempotency key, so it writes the same rows"
    );
    assert_eq!(f.model.calls(), 1, "the planner is not asked again");
    assert!(run.retry(None).is_err(), "a finished run does not retry");
}

#[tokio::test]
async fn a_failed_run_retries_only_for_transient_errors() {
    for error in [
        ToolError::InvalidArgs("bad".into()),
        ToolError::UnknownTool("x".into()),
        ToolError::Config("x".into()),
        ToolError::Blocked("x".into()),
        ToolError::Decode("x".into()),
    ] {
        let (_, mut run, _) = failed_in_save(error.clone()).await;
        assert!(!run.retryable(), "{error} is not retryable");
        assert!(
            matches!(run.retry(None), Err(GraphError::Resume(_))),
            "{error}"
        );
        let restored = GraphRun::from_checkpoint(&run.checkpoint()).unwrap();
        assert!(!restored.retryable(), "{error} after a checkpoint");
    }
    let (_, run, _) = failed_in_save(ToolError::Http("503".into())).await;
    assert!(run.retryable(), "an http failure is retryable");
    let (f, w) = tool_only("x", "string", ToolOutput::error("boom"));
    let mut run = GraphRun::new(w, Map::new(), Limits::default(), &f.io)
        .await
        .unwrap();
    assert!(matches!(run.step(&f.io).await, GraphOutcome::Failed(_)));
    assert!(!run.retryable(), "an error reply of the tool is final");
    let (f, w) = tool_only("x", "int", ToolOutput::text("\"text\""));
    let mut run = GraphRun::new(w, Map::new(), Limits::default(), &f.io)
        .await
        .unwrap();
    assert!(matches!(run.step(&f.io).await, GraphOutcome::Failed(_)));
    assert!(!run.retryable(), "a type mismatch is final");
}

#[tokio::test]
async fn a_run_out_of_transient_retries_retries_with_a_fresh_count() {
    let (f, _) = linear();
    let text = LINEAR.replace("start = \"plan\"", "start = \"plan\"\nmax_retries = 1");
    let mut run = GraphRun::new(
        wf(&text),
        input(&[("goal", json!("blocked"))]),
        Limits::default(),
        &f.io,
    )
    .await
    .unwrap();
    assert!(matches!(
        run.step(&f.io).await,
        GraphOutcome::RetryAt { .. }
    ));
    assert!(matches!(run.step(&f.io).await, GraphOutcome::Failed(_)));
    assert!(run.retryable());
    run.retry(None).unwrap();
    assert!(
        matches!(run.step(&f.io).await, GraphOutcome::RetryAt { .. }),
        "the retry count starts again"
    );
}

#[tokio::test]
async fn a_failed_checkpoint_from_before_the_class_is_classified_from_its_error() {
    let (f, run, log) =
        failed_in_save(ToolError::Failed("database: max clients reached".into())).await;
    let mut cp = run.checkpoint();
    assert!(cp
        .state
        .as_object_mut()
        .unwrap()
        .remove("failure")
        .is_some());
    let mut old = GraphRun::from_checkpoint(&cp).unwrap();
    assert!(old.retryable(), "a store failure saved without its class");
    old.retry(None).unwrap();
    assert!(matches!(
        drive(&mut old, &f.io).await,
        GraphOutcome::Done(_)
    ));
    let keys = log.lock().unwrap().clone();
    assert_eq!(keys[0], keys[1]);

    for (error, retryable) in [
        ("node `save`: tool `save_plan`: http: 503", true),
        ("error after 3 retries: RateLimited: slow down", true),
        (
            "node `save`: tool `save_plan`: invalid arguments: bad",
            false,
        ),
        ("node `save`: tool `save_plan` failed: boom", false),
        ("max_steps 50 reached", false),
    ] {
        let mut cp = run.checkpoint();
        let state = cp.state.as_object_mut().unwrap();
        state.remove("failure");
        state.insert("status".into(), json!({"status": "failed", "error": error}));
        assert_eq!(
            GraphRun::from_checkpoint(&cp).unwrap().retryable(),
            retryable,
            "{error}"
        );
    }
}
