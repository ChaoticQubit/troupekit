use async_trait::async_trait;
use serde_json::{json, Map, Value};
use std::collections::BTreeMap;
use std::sync::Arc;
use troupekit_agent::{
    AgentError, AgentIo, Limits, MemorySource, Persona, PersonaLoader, SimpleIo,
};
use troupekit_graph::testing::{text_reply, FnModel, GatedTool};
use troupekit_graph::{
    Conditions, Flow, GraphError, GraphIo, GraphOutcome, GraphRun, SimpleGraphIo,
};
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

struct Flows {
    inner: SimpleGraphIo,
    flows: BTreeMap<String, String>,
}

#[async_trait]
impl GraphIo for Flows {
    fn agent(&self) -> &dyn AgentIo {
        self.inner.agent()
    }

    async fn persona(&self, name: &str) -> Result<Persona, AgentError> {
        self.inner.persona(name).await
    }

    fn conditions(&self) -> &Conditions {
        self.inner.conditions()
    }

    async fn flow(&self, name: &str) -> Result<Flow, GraphError> {
        let text = self
            .flows
            .get(name)
            .ok_or_else(|| GraphError::Setup(format!("no flow `{name}`")))?;
        Flow::parse(&format!("test/{name}.toml"), text)
            .map_err(|e| GraphError::Setup(e.to_string()))
    }
}

fn io(gates: &GatedTool, model: Arc<FnModel>, flows: &[(&str, String)]) -> Flows {
    let mut tools = ToolRegistry::new();
    for name in ["c1", "c2"] {
        let reply = format!("{name}!");
        tools
            .register(gates.tool(name, move |_| ToolOutput::text(reply.clone())))
            .unwrap();
    }
    let agent = SimpleIo::new(tools).with_model("fake/m", model, PRICE);
    let personas = PersonaLoader::new(vec![Arc::new(MemorySource::new(
        "test",
        [("writer.toml".to_string(), WRITER.to_string())],
    ))]);
    Flows {
        inner: SimpleGraphIo::new(Arc::new(agent), personas),
        flows: flows
            .iter()
            .map(|(n, t)| (n.to_string(), t.clone()))
            .collect(),
    }
}

fn model() -> Arc<FnModel> {
    Arc::new(FnModel::new("fake/m", |_| text_reply("polished", 10)))
}

fn parent(child: &str) -> String {
    format!(
        "name = 'parent'\nstart = 'sub'\n[state]\ntopic = 'string'\ny = 'string?'\n[[nodes]]\nid = 'sub'\nkind = 'flow'\nflow = '{child}'\nreads = ['topic']\nreturns = 'summary'\nwrites = 'y'\n[[nodes]]\nid = 'e'\nkind = 'end'\n[[edges]]\nfrom = 'sub'\nto = 'e'\n"
    )
}

const CHILD: &str = "name = 'polish'\nstart = 'c1'\n[state]\ntopic = 'string'\nc1 = 'string?'\nsummary = 'string?'\n[[nodes]]\nid = 'c1'\nkind = 'tool'\ntool = 'c1'\nreads = ['topic']\nwrites = 'c1'\n[[nodes]]\nid = 'w'\nkind = 'agent'\npersona = 'writer'\nreads = ['c1']\nwrites = 'summary'\n[[nodes]]\nid = 'e'\nkind = 'end'\n[[edges]]\nfrom = 'c1'\nto = 'w'\n[[edges]]\nfrom = 'w'\nto = 'e'\n";

fn nests(name: &str, child: &str) -> String {
    parent(child)
        .replace("name = 'parent'", &format!("name = '{name}'"))
        .replace("y = 'string?'", "y = 'string?'\nsummary = 'string?'")
}

fn topic() -> Map<String, Value> {
    let mut m = Map::new();
    m.insert("topic".into(), json!("rust"));
    m
}

async fn drive(run: &mut GraphRun, io: &Flows) -> GraphOutcome {
    for _ in 0..40 {
        let out = run.step(io).await;
        if out != GraphOutcome::Continue {
            return out;
        }
    }
    panic!("run did not settle in 40 steps");
}

async fn start(name: &str, io: &Flows) -> Result<GraphRun, GraphError> {
    let flow = io.flow(name).await.unwrap();
    GraphRun::start(flow, topic(), Limits::default(), io).await
}

#[tokio::test]
async fn subflow_runs_child_and_writes_its_return_key() {
    let gates = GatedTool::new();
    gates.open("c1");
    let io = io(
        &gates,
        model(),
        &[("parent", parent("polish")), ("polish", CHILD.into())],
    );
    let mut run = start("parent", &io).await.unwrap();
    let GraphOutcome::Done(state) = drive(&mut run, &io).await else {
        panic!("the parent finishes");
    };
    assert_eq!(state["y"], json!("polished"));
    assert!(!state.contains_key("c1"), "child state stays in the child");
    assert_eq!(gates.calls("c1").len(), 1);
}

#[tokio::test]
async fn subflow_child_is_durable_inside_parent_checkpoint() {
    let gates = GatedTool::new();
    gates.open("c1");
    let model = model();
    let io = io(
        &gates,
        model.clone(),
        &[("parent", parent("polish")), ("polish", CHILD.into())],
    );
    let mut run = start("parent", &io).await.unwrap();
    for _ in 0..40 {
        if gates.calls("c1").len() == 1 {
            break;
        }
        assert_eq!(run.step(&io).await, GraphOutcome::Continue);
    }
    assert_eq!(gates.calls("c1").len(), 1, "the child ran its first node");
    let cp = run.checkpoint();
    drop(run);
    let mut run = GraphRun::from_checkpoint(&cp).unwrap();
    let GraphOutcome::Done(state) = drive(&mut run, &io).await else {
        panic!("the restored parent finishes");
    };
    assert_eq!(state["y"], json!("polished"));
    assert_eq!(gates.calls("c1").len(), 1, "the child resumes where it was");
    assert_eq!(model.calls(), 1);
}

#[tokio::test]
async fn subflow_cycle_is_refused_at_start() {
    let gates = GatedTool::new();
    let io = io(
        &gates,
        model(),
        &[("a", nests("a", "b")), ("b", nests("b", "a"))],
    );
    let err = start("a", &io).await.unwrap_err();
    assert!(
        matches!(&err, GraphError::Setup(m) if m.contains("reaches itself") && m.contains("a > b > a")),
        "{err}"
    );
    assert!(gates.calls("c1").is_empty());
}

#[tokio::test]
async fn subflow_depth_cap_holds() {
    let gates = GatedTool::new();
    gates.open("c1");
    let mut flows: Vec<(&str, String)> = vec![("polish", CHILD.into())];
    let names = ["f0", "f1", "f2", "f3", "f4", "f5"];
    for (i, name) in names.iter().enumerate() {
        let child = names.get(i + 1).copied().unwrap_or("polish");
        flows.push((name, nests(name, child)));
    }
    let io = io(&gates, model(), &flows);
    let err = start("f0", &io).await.unwrap_err();
    assert!(
        matches!(&err, GraphError::Setup(m) if m.contains("deeper than 4")),
        "{err}"
    );
    assert!(
        start("f3", &io).await.is_ok(),
        "four levels below the root are allowed"
    );
}

#[tokio::test]
async fn child_usage_counts_against_parent_budget() {
    let gates = GatedTool::new();
    gates.open("c1");
    let io = io(
        &gates,
        model(),
        &[("parent", parent("polish")), ("polish", CHILD.into())],
    );
    let mut run = start("parent", &io).await.unwrap();
    assert!(matches!(drive(&mut run, &io).await, GraphOutcome::Done(_)));
    assert_eq!(
        run.usage().input_tokens,
        10,
        "the child's model call counts"
    );
    let tight = Limits {
        run: Some(1),
        ..Limits::default()
    };
    let flow = io.flow("parent").await.unwrap();
    let mut run = GraphRun::start(flow, topic(), tight, &io).await.unwrap();
    let out = drive(&mut run, &io).await;
    assert!(
        matches!(&out, GraphOutcome::Waiting(_) | GraphOutcome::Failed(_)),
        "a child over the parent budget stops: {out:?}"
    );
}

struct PrivateFlows(Flows);

#[async_trait]
impl GraphIo for PrivateFlows {
    fn agent(&self) -> &dyn AgentIo {
        self.0.agent()
    }

    async fn persona(&self, name: &str) -> Result<Persona, AgentError> {
        self.0.persona(name).await
    }

    fn conditions(&self) -> &Conditions {
        self.0.conditions()
    }

    async fn flow(&self, name: &str) -> Result<Flow, GraphError> {
        self.0.flow(name).await
    }

    fn allow_model(&self, _node: &str, persona: &Persona, selector: &str) -> Result<(), String> {
        Err(format!(
            "persona `{}` is private; `{selector}` is a cloud handle",
            persona.name
        ))
    }
}

#[tokio::test]
async fn private_override_inside_a_subflow_is_refused_at_start() {
    let gates = GatedTool::new();
    let child = CHILD.replace(
        "persona = 'writer'\n",
        "persona = 'writer'\nmodel = 'fake/cloud'\n",
    );
    let io = PrivateFlows(io(
        &gates,
        model(),
        &[("parent", parent("polish")), ("polish", child)],
    ));
    let flow = io.flow("parent").await.unwrap();
    let err = GraphRun::start(flow, topic(), Limits::default(), &io)
        .await
        .unwrap_err();
    assert!(
        matches!(&err, GraphError::Setup(m) if m.contains("flow `polish`, node `w`") && m.contains("is private")),
        "{err}"
    );
    assert!(gates.calls("c1").is_empty(), "nothing in the child ran");
}

#[tokio::test]
async fn child_retry_at_keeps_the_parent_retrying_not_failed() {
    let gates = GatedTool::new();
    gates.open("c1");
    let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let seen = calls.clone();
    let model = Arc::new(FnModel::new("fake/m", move |_| {
        if seen.fetch_add(1, std::sync::atomic::Ordering::SeqCst) == 0 {
            return Err(troupekit_core::error::ProviderError::new(
                troupekit_core::error::ProviderErrorKind::RateLimited,
                "wait",
            ));
        }
        text_reply("polished", 10)
    }));
    let io = io(
        &gates,
        model,
        &[("parent", parent("polish")), ("polish", CHILD.into())],
    );
    let mut run = start("parent", &io).await.unwrap();
    let out = drive(&mut run, &io).await;
    assert!(
        matches!(&out, GraphOutcome::RetryAt { error, .. } if error.contains("RateLimited")),
        "{out:?}"
    );
    let GraphOutcome::Done(state) = drive(&mut run, &io).await else {
        panic!("the parent finishes after the child's retry");
    };
    assert_eq!(state["y"], json!("polished"));
    assert_eq!(gates.calls("c1").len(), 1, "the child's tool ran once");
}

#[tokio::test]
async fn retryable_child_failure_makes_the_parent_retryable() {
    let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let seen = calls.clone();
    let mut tools = ToolRegistry::new();
    tools
        .register(troupekit_graph::testing::FnTool::new("c1", move |_, _| {
            let first = seen.fetch_add(1, std::sync::atomic::Ordering::SeqCst) == 0;
            async move {
                if first {
                    return Err(troupekit_tools::ToolError::Failed(
                        "database: max clients reached".into(),
                    ));
                }
                Ok(ToolOutput::text("c1!"))
            }
        }))
        .unwrap();
    let agent = SimpleIo::new(tools).with_model("fake/m", model(), PRICE);
    let personas = PersonaLoader::new(vec![Arc::new(MemorySource::new(
        "test",
        [("writer.toml".to_string(), WRITER.to_string())],
    ))]);
    let io = Flows {
        inner: SimpleGraphIo::new(Arc::new(agent), personas),
        flows: [("parent", parent("polish")), ("polish", CHILD.into())]
            .into_iter()
            .map(|(n, t)| (n.to_string(), t))
            .collect(),
    };
    let mut run = start("parent", &io).await.unwrap();
    let out = drive(&mut run, &io).await;
    assert!(
        matches!(&out, GraphOutcome::Failed(e) if e.contains("subflow")),
        "{out:?}"
    );
    assert!(run.retryable(), "a child's store failure is retryable");
    let mut run = GraphRun::from_checkpoint(&run.checkpoint()).unwrap();
    assert!(run.retryable(), "the class survives a checkpoint");
    run.retry(None).unwrap();
    let GraphOutcome::Done(state) = drive(&mut run, &io).await else {
        panic!("the retried parent finishes");
    };
    assert_eq!(state["y"], json!("polished"));
    assert_eq!(
        calls.load(std::sync::atomic::Ordering::SeqCst),
        2,
        "the child's tool re-ran once"
    );
}

#[tokio::test]
async fn child_with_an_unregistered_tool_is_refused_at_start() {
    let gates = GatedTool::new();
    gates.open("c2");
    let root = "name = 'parent'\nstart = 'r'\n[state]\ntopic = 'string'\nr = 'string?'\ny = 'string?'\n[[nodes]]\nid = 'r'\nkind = 'tool'\ntool = 'c2'\nwrites = 'r'\n[[nodes]]\nid = 'sub'\nkind = 'flow'\nflow = 'polish'\nreads = ['topic']\nreturns = 'summary'\nwrites = 'y'\n[[nodes]]\nid = 'e'\nkind = 'end'\n[[edges]]\nfrom = 'r'\nto = 'sub'\n[[edges]]\nfrom = 'sub'\nto = 'e'\n";
    let child = CHILD.replace("tool = 'c1'", "tool = 'nope'");
    let io = io(
        &gates,
        model(),
        &[("parent", root.to_string()), ("polish", child)],
    );
    let err = start("parent", &io).await.unwrap_err();
    assert!(
        matches!(&err, GraphError::Setup(m) if m.contains("flow `polish`, node `c1`") && m.contains("tool `nope` is not registered")),
        "{err}"
    );
    assert!(gates.calls("c2").is_empty(), "no root tool ran");
}
