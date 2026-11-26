use serde_json::{json, Map, Value};
use std::sync::{Arc, Mutex};
use troupekit_agent::{Limits, MemorySource, PersonaLoader, SimpleIo};
use troupekit_graph::testing::{text_reply, FnModel, FnTool};
use troupekit_graph::{Flow, GraphError, GraphIo, GraphOutcome, GraphRun, SimpleGraphIo};
use troupekit_providers::Price;
use troupekit_tools::{ToolOutput, ToolRegistry};

const PRICE: Price = Price {
    input_usd_micros_per_mtok: 1_000_000,
    output_usd_micros_per_mtok: 2_000_000,
    verified: true,
    long_context: None,
    search_usd_micros: 0,
};

const WRITER: &str = "name = 'writer'\ndisplay_name = 'W'\nrole = 'Writes.'\ngoal = 'Done.'\ninstructions = 'Do it.'\nmodel = 'fake/m'\n";

fn io(models: &[(&str, Arc<FnModel>)]) -> SimpleGraphIo {
    let mut tools = ToolRegistry::new();
    for name in ["l", "r", "x"] {
        tools
            .register(FnTool::new(name, move |_, _| async move {
                Ok(ToolOutput::text(format!("{name}!")))
            }))
            .unwrap();
    }
    let mut agent = SimpleIo::new(tools);
    for (id, model) in models {
        agent = agent.with_model(id, model.clone(), PRICE);
    }
    let personas = PersonaLoader::new(vec![Arc::new(MemorySource::new(
        "test",
        [("writer.toml".to_string(), WRITER.to_string())],
    ))]);
    SimpleGraphIo::new(Arc::new(agent), personas)
}

async fn drive(run: &mut GraphRun, io: &SimpleGraphIo) -> GraphOutcome {
    for _ in 0..40 {
        let out = run.step(io).await;
        if out != GraphOutcome::Continue {
            return out;
        }
    }
    panic!("run did not settle in 40 steps");
}

async fn run(text: &str, input: Map<String, Value>, io: &SimpleGraphIo) -> GraphOutcome {
    let flow = Flow::parse("test/r.toml", text).unwrap();
    let mut run = GraphRun::start(flow, input, Limits::default(), io)
        .await
        .unwrap();
    drive(&mut run, io).await
}

fn switch_flow(ty: &str, routes: &[(&str, &str)]) -> String {
    let mut text = format!(
        "name = 'sw'\nstart = 's'\n[state]\nk = '{ty}'\nl = 'string?'\nr = 'string?'\n[[nodes]]\nid = 's'\nkind = 'switch'\non = 'k'\n[[nodes]]\nid = 'l'\nkind = 'tool'\ntool = 'l'\nwrites = 'l'\n[[nodes]]\nid = 'r'\nkind = 'tool'\ntool = 'r'\nwrites = 'r'\n[[nodes]]\nid = 'e'\nkind = 'end'\n[[edges]]\nfrom = 'l'\nto = 'e'\n[[edges]]\nfrom = 'r'\nto = 'e'\n"
    );
    for (to, when) in routes {
        text.push_str(&format!(
            "[[edges]]\nfrom = 's'\nto = '{to}'\nwhen = '{when}'\n"
        ));
    }
    text
}

fn input(k: Value) -> Map<String, Value> {
    let mut m = Map::new();
    m.insert("k".into(), k);
    m
}

fn took(out: GraphOutcome) -> (bool, bool) {
    let GraphOutcome::Done(state) = out else {
        panic!("{out:?}");
    };
    (
        state.get("l") == Some(&json!("l!")),
        state.get("r") == Some(&json!("r!")),
    )
}

#[tokio::test]
async fn switch_routes_on_enum_bool_and_string() {
    let io = io(&[]);
    let cases = [
        (
            "enum(left, right)",
            &[("l", "left"), ("r", "right")],
            json!("right"),
            (false, true),
        ),
        (
            "bool",
            &[("l", "true"), ("r", "false")],
            json!(true),
            (true, false),
        ),
        (
            "bool",
            &[("l", "true"), ("r", "false")],
            json!(false),
            (false, true),
        ),
        (
            "string",
            &[("l", "go"), ("r", "stop")],
            json!("go"),
            (true, false),
        ),
    ];
    for (ty, routes, value, want) in cases {
        let out = run(&switch_flow(ty, routes), input(value.clone()), &io).await;
        assert_eq!(took(out), want, "{ty} = {value}");
    }
}

#[tokio::test]
async fn switch_without_matching_label_fails_naming_the_value() {
    let io = io(&[]);
    let text = switch_flow("string", &[("l", "go"), ("r", "stop")]);
    let GraphOutcome::Failed(e) = run(&text, input(json!("wait")), &io).await else {
        panic!("an unmatched label fails the run");
    };
    assert!(e.contains("node `s`") && e.contains("`wait`"), "{e}");
    let GraphOutcome::Failed(e) = run(&text, Map::new(), &io).await else {
        panic!("an unset key fails the run");
    };
    assert!(e.contains("node `s`") && e.contains("`k`"), "{e}");
}

fn goto_flow(extra: &str) -> String {
    format!(
        "name = 'gt'\nstart = 'a'\n{extra}[state]\npick = '{{next: string, reason: string}}?'\nl = 'string?'\nr = 'string?'\n[[nodes]]\nid = 'a'\nkind = 'agent'\npersona = 'writer'\nprompt = 'Pick.'\ngoto = ['l', 'r']\nwrites = 'pick'\n[[nodes]]\nid = 'l'\nkind = 'tool'\ntool = 'l'\nwrites = 'l'\n[[nodes]]\nid = 'r'\nkind = 'tool'\ntool = 'r'\nwrites = 'r'\n[[nodes]]\nid = 'e'\nkind = 'end'\n[[edges]]\nfrom = 'l'\nto = 'e'\n[[edges]]\nfrom = 'r'\nto = 'e'\n"
    )
}

#[tokio::test]
async fn goto_agent_output_is_forced_to_its_targets() {
    let schemas: Arc<Mutex<Vec<Option<Value>>>> = Arc::default();
    let seen = schemas.clone();
    let model = Arc::new(FnModel::new("fake/m", move |req| {
        seen.lock().unwrap().push(req.output_schema.clone());
        text_reply(r#"{"next": "r", "reason": "right fits"}"#, 10)
    }));
    let right = io(&[("fake/m", model)]);
    let GraphOutcome::Done(state) = run(&goto_flow(""), Map::new(), &right).await else {
        panic!("the goto run finishes");
    };
    assert_eq!(state["r"], json!("r!"));
    assert_eq!(state.get("l"), None, "only the chosen target runs");
    assert_eq!(state["pick"], json!({"next": "r", "reason": "right fits"}));
    let schemas = schemas.lock().unwrap();
    let schema = schemas[0]
        .as_ref()
        .expect("the goto agent asks for typed output");
    assert_eq!(schema["properties"]["next"]["enum"], json!(["l", "r"]));

    let wrong = Arc::new(FnModel::new("fake/m", |_| {
        text_reply(r#"{"next": "zz", "reason": "nowhere"}"#, 10)
    }));
    let io = io(&[("fake/m", wrong)]);
    let GraphOutcome::Failed(e) = run(&goto_flow(""), Map::new(), &io).await else {
        panic!("a choice outside the targets fails the node");
    };
    assert!(e.contains("node `a`"), "{e}");
}

#[tokio::test]
async fn goto_cycle_is_bounded_by_max_steps() {
    let text = "name = 'loop'\nstart = 'a'\nmax_steps = 5\n[state]\n[[nodes]]\nid = 'a'\nkind = 'agent'\npersona = 'writer'\nprompt = 'Pick.'\ngoto = ['a', 'e']\n[[nodes]]\nid = 'e'\nkind = 'end'\n";
    let model = Arc::new(FnModel::new("fake/m", |_| {
        text_reply(r#"{"next": "a", "reason": "again"}"#, 10)
    }));
    let io = io(&[("fake/m", model.clone())]);
    let GraphOutcome::Failed(e) = run(text, Map::new(), &io).await else {
        panic!("a goto cycle stops at max_steps");
    };
    assert!(e.contains("max_steps 5"), "{e}");
    assert!(model.calls() <= 5, "{} model calls", model.calls());
}

fn writer_flow(node_extra: &str) -> String {
    format!(
        "name = 'ov'\nstart = 'a'\n[state]\nout = 'string?'\n[[nodes]]\nid = 'a'\nkind = 'agent'\npersona = 'writer'\nprompt = 'Write.'\nwrites = 'out'\n{node_extra}[[nodes]]\nid = 'e'\nkind = 'end'\n[[edges]]\nfrom = 'a'\nto = 'e'\n"
    )
}

#[tokio::test]
async fn node_model_override_replaces_persona_model() {
    let base = Arc::new(FnModel::new("fake/m", |_| text_reply("base", 10)));
    let other = Arc::new(FnModel::new("fake/other", |_| text_reply("other", 10)));
    let io = io(&[("fake/m", base.clone()), ("fake/other", other.clone())]);
    let GraphOutcome::Done(state) =
        run(&writer_flow("model = 'fake/other'\n"), Map::new(), &io).await
    else {
        panic!("the overridden run finishes");
    };
    assert_eq!(state["out"], json!("other"));
    assert_eq!((base.calls(), other.calls()), (0, 1));
}

#[tokio::test]
async fn node_override_never_mutates_the_loaded_persona() {
    let model = Arc::new(FnModel::new("fake/m", |_| {
        text_reply(r#"{"next": "l", "reason": "left"}"#, 10)
    }));
    let other = Arc::new(FnModel::new("fake/other", |_| text_reply("other", 10)));
    let io = io(&[("fake/m", model), ("fake/other", other)]);
    let before = io.persona("writer").await.unwrap();
    assert!(matches!(
        run(&goto_flow(""), Map::new(), &io).await,
        GraphOutcome::Done(_)
    ));
    assert!(matches!(
        run(&writer_flow("model = 'fake/other'\n"), Map::new(), &io).await,
        GraphOutcome::Done(_)
    ));
    let after = io.persona("writer").await.unwrap();
    assert_eq!(after, before);
    assert_eq!((after.model.as_str(), after.output), ("fake/m", None));
}

const HELPER: &str = "name = 'helper'\ndisplay_name = 'H'\nrole = 'Helps.'\ngoal = 'Answer.'\ninstructions = 'Help.'\nmodel = 'fake/h'\n";

fn leader_model() -> Arc<FnModel> {
    use troupekit_core::message::{ContentPart, Message, Role};
    use troupekit_providers::{CompletionResponse, StopReason};
    Arc::new(FnModel::new("fake/m", |req| {
        let result = req
            .messages
            .iter()
            .flat_map(|m| &m.parts)
            .find_map(|p| match p {
                ContentPart::ToolResult { content, .. } => Some(content.clone()),
                _ => None,
            });
        match result {
            Some(content) => text_reply(&content, 10),
            None => Ok(CompletionResponse {
                message: Message {
                    role: Role::Assistant,
                    parts: vec![ContentPart::ToolCall {
                        id: "c1".into(),
                        name: "delegate_work".into(),
                        arguments: json!({"coworker": "helper", "task": "say hi"}),
                    }],
                },
                usage: troupekit_core::event::Usage {
                    input_tokens: 10,
                    output_tokens: 5,
                    cost_usd_micros: 0,
                },
                stop: StopReason::ToolUse,
            }),
        }
    }))
}

struct Delegating {
    io: SimpleGraphIo,
    sink: Arc<troupekit_agent::VecSink>,
}

struct Unshared(SimpleGraphIo);

#[async_trait::async_trait]
impl GraphIo for Unshared {
    fn agent(&self) -> &dyn troupekit_agent::AgentIo {
        self.0.agent()
    }

    async fn persona(
        &self,
        name: &str,
    ) -> Result<troupekit_agent::Persona, troupekit_agent::AgentError> {
        self.0.persona(name).await
    }

    fn conditions(&self) -> &troupekit_graph::Conditions {
        self.0.conditions()
    }
}

fn delegating_io() -> Delegating {
    let sink = Arc::new(troupekit_agent::VecSink::default());
    let helper = Arc::new(FnModel::new("fake/h", |_| text_reply("member says hi", 10)));
    let mut agent = SimpleIo::new(ToolRegistry::new())
        .with_model("fake/m", leader_model(), PRICE)
        .with_model("fake/h", helper, PRICE);
    agent.events = sink.clone();
    let personas = PersonaLoader::new(vec![Arc::new(MemorySource::new(
        "test",
        [
            ("writer.toml".to_string(), WRITER.to_string()),
            ("helper.toml".to_string(), HELPER.to_string()),
        ],
    ))]);
    let agent: Arc<dyn troupekit_agent::AgentIo> = Arc::new(agent);
    Delegating {
        io: SimpleGraphIo::new(agent, personas),
        sink,
    }
}

fn delegate_flow() -> String {
    writer_flow("delegates = ['helper']\n")
}

#[tokio::test]
async fn delegate_node_calls_member_and_gets_fenced_answer() {
    let d = delegating_io();
    let GraphOutcome::Done(state) = run(&delegate_flow(), Map::new(), &d.io).await else {
        panic!("the delegating run finishes");
    };
    let out = state["out"].as_str().unwrap();
    assert!(out.contains("<untrusted source="), "{out}");
    assert!(out.contains("member says hi"), "{out}");
}

#[tokio::test]
async fn member_events_stream_through_graph_sink() {
    let d = delegating_io();
    let flow = Flow::parse("test/r.toml", &delegate_flow()).unwrap();
    let mut graph = GraphRun::start(flow, Map::new(), Limits::default(), &d.io)
        .await
        .unwrap();
    assert!(matches!(
        drive(&mut graph, &d.io).await,
        GraphOutcome::Done(_)
    ));
    let events = d.sink.events();
    let member: Vec<_> = events
        .iter()
        .filter(|e| e.run_id != graph.run_id())
        .collect();
    assert!(!member.is_empty(), "member events reach the sink");
    assert!(
        member.iter().all(|e| e.run_id == member[0].run_id),
        "one member run, one id"
    );
    assert!(
        events.iter().any(|e| e.run_id == graph.run_id()),
        "graph events too"
    );
}

#[tokio::test]
async fn delegates_without_agent_arc_fail_at_start() {
    let io = Unshared(delegating_io().io);
    let flow = Flow::parse("test/r.toml", &delegate_flow()).unwrap();
    let err = GraphRun::start(flow, Map::new(), Limits::default(), &io)
        .await
        .unwrap_err();
    assert!(
        matches!(&err, GraphError::Setup(m) if m.contains("delegates need a GraphIo that shares its AgentIo (agent_arc)")),
        "{err}"
    );
}

#[tokio::test]
async fn node_registry_holds_only_persona_and_delegation_tools() {
    let tools: Arc<Mutex<Vec<Vec<String>>>> = Arc::default();
    let seen = tools.clone();
    let d = delegating_io();
    let spy = Arc::new(FnModel::new("fake/m", move |req| {
        seen.lock()
            .unwrap()
            .push(req.tools.iter().map(|t| t.name.clone()).collect());
        text_reply("done", 10)
    }));
    let mut registry = ToolRegistry::new();
    registry
        .register(FnTool::new("stray", |_, _| async {
            Ok(ToolOutput::text("no"))
        }))
        .unwrap();
    let mut agent = SimpleIo::new(registry)
        .with_model("fake/m", spy, PRICE)
        .with_model(
            "fake/h",
            Arc::new(FnModel::new("fake/h", |_| text_reply("hi", 10))),
            PRICE,
        );
    agent.events = d.sink.clone();
    let io = SimpleGraphIo::new(Arc::new(agent), d.io.personas);
    assert!(matches!(
        run(&delegate_flow(), Map::new(), &io).await,
        GraphOutcome::Done(_)
    ));
    let tools = tools.lock().unwrap();
    assert_eq!(
        tools[0],
        vec!["delegate_work", "ask_question"],
        "no stray tool, no member tool"
    );
}

struct Privacy {
    inner: SimpleGraphIo,
    private: Vec<&'static str>,
}

#[async_trait::async_trait]
impl GraphIo for Privacy {
    fn agent(&self) -> &dyn troupekit_agent::AgentIo {
        self.inner.agent()
    }

    async fn persona(
        &self,
        name: &str,
    ) -> Result<troupekit_agent::Persona, troupekit_agent::AgentError> {
        self.inner.persona(name).await
    }

    fn conditions(&self) -> &troupekit_graph::Conditions {
        self.inner.conditions()
    }

    fn allow_model(
        &self,
        _node: &str,
        persona: &troupekit_agent::Persona,
        selector: &str,
    ) -> Result<(), String> {
        if self.private.contains(&persona.name.as_str()) && selector != "fake/local" {
            return Err(format!("`{selector}` is not an on-device handle"));
        }
        Ok(())
    }
}

fn privacy_io(private: Vec<&'static str>) -> (Privacy, Arc<FnModel>, Arc<FnModel>) {
    let cloud = Arc::new(FnModel::new("fake/cloud", |_| text_reply("cloud", 10)));
    let local = Arc::new(FnModel::new("fake/local", |_| text_reply("local", 10)));
    let base = Arc::new(FnModel::new("fake/m", |_| text_reply("base", 10)));
    let inner = io(&[
        ("fake/m", base),
        ("fake/cloud", cloud.clone()),
        ("fake/local", local.clone()),
    ]);
    (Privacy { inner, private }, cloud, local)
}

async fn start_on(io: &Privacy, text: &str) -> Result<GraphRun, GraphError> {
    let flow = Flow::parse("test/r.toml", text).unwrap();
    GraphRun::start(flow, Map::new(), Limits::default(), io).await
}

#[tokio::test]
async fn private_persona_override_onto_a_cloud_handle_is_refused_at_start() {
    let (io, cloud, _) = privacy_io(vec!["writer"]);
    let err = start_on(&io, &writer_flow("model = 'fake/cloud'\n"))
        .await
        .unwrap_err();
    assert!(
        matches!(&err, GraphError::Setup(m)
            if m.contains("node `a`")
                && m.contains("model override `fake/cloud` is not allowed for persona `writer`")
                && m.contains("not an on-device handle")),
        "{err}"
    );
    assert_eq!(cloud.calls(), 0, "nothing reaches the cloud handle");
}

#[tokio::test]
async fn private_persona_override_that_stays_on_device_runs() {
    let (io, cloud, local) = privacy_io(vec!["writer"]);
    let mut run = start_on(&io, &writer_flow("model = 'fake/local'\n"))
        .await
        .unwrap();
    let mut out = GraphOutcome::Continue;
    for _ in 0..10 {
        out = run.step(&io).await;
        if out != GraphOutcome::Continue {
            break;
        }
    }
    let GraphOutcome::Done(state) = out else {
        panic!("{out:?}");
    };
    assert_eq!(state["out"], json!("local"));
    assert_eq!((cloud.calls(), local.calls()), (0, 1));
}

#[tokio::test]
async fn override_on_a_non_private_persona_is_kept() {
    let (io, cloud, _) = privacy_io(vec![]);
    let mut run = start_on(&io, &writer_flow("model = 'fake/cloud'\n"))
        .await
        .unwrap();
    for _ in 0..10 {
        if run.step(&io).await != GraphOutcome::Continue {
            break;
        }
    }
    assert_eq!(cloud.calls(), 1);
}

#[tokio::test]
async fn an_override_equal_to_the_persona_model_never_asks_the_hook() {
    let (io, _, _) = privacy_io(vec!["writer"]);
    assert!(start_on(&io, &writer_flow("model = 'fake/m'\n"))
        .await
        .is_ok());
}

mod fields {
    use super::{io, run};
    use serde_json::{json, Map, Value};
    use std::sync::{Arc, Mutex};
    use troupekit_graph::testing::{text_reply, FnModel};
    use troupekit_graph::{Flow, GraphOutcome};

    fn handoff(node_extra: &str, state_extra: &str) -> String {
        format!(
            "name = 'ho'\nstart = 'a'\n[state]\noutput = 'string?'\ndone = 'bool?'\ntasks = 'list?'\nl = 'string?'\nr = 'string?'\n{state_extra}[[nodes]]\nid = 'a'\nkind = 'agent'\npersona = 'writer'\nprompt = 'Pick.'\ngoto = ['l', 'r']\n{node_extra}[[nodes]]\nid = 'l'\nkind = 'tool'\ntool = 'l'\nwrites = 'l'\n[[nodes]]\nid = 'r'\nkind = 'tool'\ntool = 'r'\nwrites = 'r'\n[[nodes]]\nid = 'e'\nkind = 'end'\n[[edges]]\nfrom = 'l'\nto = 'e'\n[[edges]]\nfrom = 'r'\nto = 'e'\n"
        )
    }

    fn plain(node_extra: &str) -> String {
        format!(
            "name = 'pl'\nstart = 'a'\n[state]\noutput = 'string?'\ndone = 'bool?'\ntasks = 'list?'\n[[nodes]]\nid = 'a'\nkind = 'agent'\npersona = 'writer'\nprompt = 'Work.'\n{node_extra}[[nodes]]\nid = 'e'\nkind = 'end'\n[[edges]]\nfrom = 'a'\nto = 'e'\n"
        )
    }

    fn replying(text: &'static str) -> (Arc<FnModel>, Arc<Mutex<Vec<Option<Value>>>>) {
        let schemas: Arc<Mutex<Vec<Option<Value>>>> = Arc::default();
        let seen = schemas.clone();
        let model = Arc::new(FnModel::new("fake/m", move |req| {
            seen.lock().unwrap().push(req.output_schema.clone());
            text_reply(text, 10)
        }));
        (model, schemas)
    }

    #[tokio::test]
    async fn agent_fields_shape_the_reply_and_sets_write_each_key() {
        let (model, schemas) = replying(r#"{"answer": "hi", "done": true, "tasks": ["a", "b"]}"#);
        let io = io(&[("fake/m", model)]);
        let text = plain(
            "fields = { answer = 'string', done = 'bool', tasks = 'list' }\nsets = { answer = 'output', done = 'done', tasks = 'tasks' }\n",
        );
        let GraphOutcome::Done(state) = run(&text, Map::new(), &io).await else {
            panic!("the run finishes");
        };
        assert_eq!(state["output"], json!("hi"));
        assert_eq!(state["done"], json!(true));
        assert_eq!(state["tasks"], json!(["a", "b"]));
        let schema = schemas.lock().unwrap()[0].clone().expect("typed output");
        assert_eq!(schema["properties"]["done"]["type"], json!("boolean"));
        assert!(
            schema["properties"].get("next").is_none(),
            "no goto, no next field"
        );
    }

    #[tokio::test]
    async fn goto_agent_maps_answer_while_routing_on_next() {
        let (model, schemas) = replying(r#"{"next": "r", "reason": "fits", "answer": "hello"}"#);
        let io = io(&[("fake/m", model)]);
        let text = handoff(
            "fields = { answer = 'string' }\nsets = { answer = 'output' }\n",
            "",
        );
        let GraphOutcome::Done(state) = run(&text, Map::new(), &io).await else {
            panic!("the handoff finishes");
        };
        assert_eq!(
            state["output"],
            json!("hello"),
            "the answer field lands in the string key"
        );
        assert_eq!(state["r"], json!("r!"), "the run routed on next");
        assert_eq!(state.get("l"), None);
        let schema = schemas.lock().unwrap()[0].clone().expect("typed output");
        assert_eq!(schema["properties"]["next"]["enum"], json!(["l", "r"]));
        assert_eq!(schema["properties"]["answer"]["type"], json!("string"));
    }

    #[tokio::test]
    async fn set_value_of_the_wrong_type_fails_the_run() {
        let (model, _) = replying(r#"{"answer": "hi", "done": "yes"}"#);
        let io = io(&[("fake/m", model)]);
        let text =
            plain("fields = { answer = 'string', done = 'string' }\nsets = { done = 'done' }\n");
        let GraphOutcome::Failed(e) = run(&text, Map::new(), &io).await else {
            panic!("a mistyped set fails");
        };
        assert!(e.contains("node `a`") && e.contains("`done`"), "{e}");
    }

    #[test]
    fn fields_and_sets_are_checked_at_parse() {
        let bad = |text: &str, want: &str| {
            let err = Flow::parse("test/f.toml", text).unwrap_err().to_string();
            assert!(err.contains(want), "{want}: {err}");
        };
        bad(
            &plain("sets = { answer = 'output' }\n"),
            "`sets` needs `fields`",
        );
        bad(
            &plain("fields = { answer = 'string' }\nsets = { other = 'output' }\n"),
            "`other` is not a field",
        );
        bad(
            &plain("fields = { answer = 'string' }\nsets = { answer = 'nope' }\n"),
            "`nope` is not declared in [state]",
        );
        bad(&plain("fields = { answer = 'strang' }\n"), "fields");
        bad(
            &handoff("fields = { next = 'string' }\n", ""),
            "`next` is set by goto",
        );
        bad(
            &plain(
                "fields = { answer = 'string' }\nsets = { answer = 'output' }\nwrites = 'output'\n",
            ),
            "`output` is written twice",
        );
        assert!(
            Flow::parse(
                "test/f.toml",
                &handoff("sets = { next = 'output' }\nfields = {}\n", "")
            )
            .is_ok(),
            "goto fields can be set"
        );
    }

    #[test]
    fn parallel_sets_of_one_key_need_a_reducer() {
        let text = "name = 'p'\nstart = 's'\n[state]\nseed = 'string?'\nout = 'string?'\n[[nodes]]\nid = 's'\nkind = 'tool'\ntool = 'l'\nwrites = 'seed'\n[[nodes]]\nid = 'a'\nkind = 'agent'\npersona = 'writer'\nprompt = 'A.'\nfields = { answer = 'string' }\nsets = { answer = 'out' }\n[[nodes]]\nid = 'b'\nkind = 'tool'\ntool = 'r'\nwrites = 'out'\n[[nodes]]\nid = 'j'\nkind = 'join'\n[[nodes]]\nid = 'e'\nkind = 'end'\n[[edges]]\nfrom = 's'\nto = 'a'\n[[edges]]\nfrom = 's'\nto = 'b'\n[[edges]]\nfrom = 'a'\nto = 'j'\n[[edges]]\nfrom = 'b'\nto = 'j'\n[[edges]]\nfrom = 'j'\nto = 'e'\n";
        let err = Flow::parse("test/p.toml", text).unwrap_err().to_string();
        assert!(
            err.contains("two branches of the fan-out write `out`"),
            "{err}"
        );
    }
}
