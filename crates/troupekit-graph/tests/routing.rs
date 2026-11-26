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
    (state["l"] == json!("l!"), state["r"] == json!("r!"))
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
        "name = 'gt'\nstart = 'a'\n{extra}[state]\npick = '{{next: string, reason: string}}?'\nl = 'string?'\nr = 'string?'\n[[nodes]]\nid = 'a'\nkind = 'agent'\npersona = 'writer'\ngoto = ['l', 'r']\nwrites = 'pick'\n[[nodes]]\nid = 'l'\nkind = 'tool'\ntool = 'l'\nwrites = 'l'\n[[nodes]]\nid = 'r'\nkind = 'tool'\ntool = 'r'\nwrites = 'r'\n[[nodes]]\nid = 'e'\nkind = 'end'\n[[edges]]\nfrom = 'l'\nto = 'e'\n[[edges]]\nfrom = 'r'\nto = 'e'\n"
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
    let text = "name = 'loop'\nstart = 'a'\nmax_steps = 5\n[state]\n[[nodes]]\nid = 'a'\nkind = 'agent'\npersona = 'writer'\ngoto = ['a', 'e']\n[[nodes]]\nid = 'e'\nkind = 'end'\n";
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
        "name = 'ov'\nstart = 'a'\n[state]\nout = 'string?'\n[[nodes]]\nid = 'a'\nkind = 'agent'\npersona = 'writer'\nwrites = 'out'\n{node_extra}[[nodes]]\nid = 'e'\nkind = 'end'\n[[edges]]\nfrom = 'a'\nto = 'e'\n"
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

#[tokio::test]
async fn goto_and_override_flows_still_refuse_unbuilt_options() {
    let io = io(&[]);
    let text = writer_flow("delegates = ['writer']\n");
    let flow = Flow::parse("test/r.toml", &text).unwrap();
    let err = GraphRun::start(flow, Map::new(), Limits::default(), &io)
        .await
        .unwrap_err();
    assert!(
        matches!(&err, GraphError::Setup(m) if m.contains("node `a` of a 0.5 kind or option")),
        "{err}"
    );
}
