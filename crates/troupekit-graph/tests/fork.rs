use serde_json::{json, Map, Value};
use std::sync::Arc;
use troupekit_agent::{Limits, PersonaLoader, SimpleIo};
use troupekit_core::event::PauseReason;
use troupekit_core::inmem::InMemCheckpoints;
use troupekit_core::store::{Checkpoint, CheckpointStore};
use troupekit_graph::testing::GatedTool;
use troupekit_graph::{Flow, GraphError, GraphOutcome, GraphRun, GraphStatus, SimpleGraphIo};
use troupekit_tools::{ToolOutput, ToolRegistry};

const CHAIN: &str = "name = 'chain'\nstart = 't1'\n[state]\nx = 'string'\ny = 'string'\nok = 'bool'\n[[nodes]]\nid = 't1'\nkind = 'tool'\ntool = 't1'\nwrites = 'x'\n[[nodes]]\nid = 't2'\nkind = 'tool'\ntool = 't2'\nreads = ['x']\nwrites = 'y'\n[[nodes]]\nid = 'h'\nkind = 'human'\nask = 'approval'\nwrites = 'ok'\n[[nodes]]\nid = 'e'\nkind = 'end'\n[[edges]]\nfrom = 't1'\nto = 't2'\n[[edges]]\nfrom = 't2'\nto = 'h'\n[[edges]]\nfrom = 'h'\nto = 'e'\n";

fn io(gates: &GatedTool) -> SimpleGraphIo {
    gates.open("t1");
    gates.open("t2");
    let mut tools = ToolRegistry::new();
    tools
        .register(gates.tool("t1", |_| ToolOutput::text("one")))
        .unwrap();
    tools
        .register(gates.tool("t2", |args| {
            ToolOutput::text(format!("{}!", args["x"].as_str().unwrap_or("none")))
        }))
        .unwrap();
    SimpleGraphIo::new(Arc::new(SimpleIo::new(tools)), PersonaLoader::new(vec![]))
}

async fn start(io: &SimpleGraphIo) -> GraphRun {
    let flow = Flow::parse("test/chain.toml", CHAIN).unwrap();
    GraphRun::start(flow, Map::new(), Limits::default(), io)
        .await
        .unwrap()
}

async fn history(io: &SimpleGraphIo) -> (GraphRun, Vec<Checkpoint>, GraphOutcome) {
    let mut run = start(io).await;
    let mut cps = vec![run.checkpoint()];
    for _ in 0..20 {
        let out = run.step(io).await;
        cps.push(run.checkpoint());
        if out != GraphOutcome::Continue {
            return (run, cps, out);
        }
    }
    panic!("run did not settle in 20 steps");
}

async fn drive(run: &mut GraphRun, io: &SimpleGraphIo) -> GraphOutcome {
    for _ in 0..20 {
        let out = run.step(io).await;
        if out != GraphOutcome::Continue {
            return out;
        }
    }
    panic!("run did not settle in 20 steps");
}

fn approval() -> GraphOutcome {
    GraphOutcome::Waiting(PauseReason::Approval { node: "h".into() })
}

fn after_t1(cps: &[Checkpoint]) -> &Checkpoint {
    cps.iter()
        .find(|cp| cp.state["state"]["x"] == json!("one"))
        .expect("a checkpoint after t1")
}

#[tokio::test]
async fn fork_gets_new_run_id_step_zero_and_parent_link() {
    let gates = GatedTool::new();
    let io = io(&gates);
    let (_, cps, _) = history(&io).await;
    let cp = after_t1(&cps);
    let fork = GraphRun::fork(cp).unwrap();
    assert_ne!(fork.run_id(), cp.run_id);
    assert_eq!(fork.steps(), 0);
    assert_eq!(fork.parent(), Some((cp.run_id, cp.step)));
    assert_eq!(fork.status(), &GraphStatus::Queued);
    let again = GraphRun::from_checkpoint(&fork.checkpoint()).unwrap();
    assert_eq!(
        again.parent(),
        Some((cp.run_id, cp.step)),
        "the link is durable"
    );
    assert_eq!(start(&io).await.parent(), None);
}

#[tokio::test]
async fn forked_run_diverges_and_original_history_is_unchanged() {
    let gates = GatedTool::new();
    let io = io(&gates);
    let (original, cps, out) = history(&io).await;
    assert_eq!(out, approval());
    let before: Vec<Value> = cps.iter().map(|c| c.state.clone()).collect();
    let mut fork = GraphRun::fork(after_t1(&cps)).unwrap();
    fork.update_state("x", json!("two"), Some("t1")).unwrap();
    assert_eq!(drive(&mut fork, &io).await, approval());
    assert_eq!(
        fork.state()["y"],
        json!("two!"),
        "the fork ran t2 on the edited value"
    );
    assert_eq!(original.state()["y"], json!("one!"));
    let after: Vec<Value> = cps.iter().map(|c| c.state.clone()).collect();
    assert_eq!(before, after, "forking copies, it never writes the parent");
}

#[tokio::test]
async fn fork_rekeys_tool_calls_and_agent_runs() {
    let gates = GatedTool::new();
    let io = io(&gates);
    let (_, cps, _) = history(&io).await;
    let original = gates.calls("t2");
    assert_eq!(original.len(), 1);
    let mut fork = GraphRun::fork(after_t1(&cps)).unwrap();
    drive(&mut fork, &io).await;
    let all = gates.calls("t2");
    assert_eq!(all.len(), 2, "the fork called t2 again");
    assert!(
        all.iter().filter(|k| original.contains(k)).count() == 1,
        "no idempotency key of the fork equals one of the original: {all:?}"
    );
}

#[tokio::test]
async fn fork_keeps_pending_approvals() {
    let gates = GatedTool::new();
    let io = io(&gates);
    let (_, cps, out) = history(&io).await;
    assert_eq!(out, approval());
    let waiting = cps.last().unwrap();
    let mut fork = GraphRun::fork(waiting).unwrap();
    assert!(matches!(fork.status(), GraphStatus::WaitingHuman { .. }));
    assert_eq!(
        fork.step(&io).await,
        approval(),
        "the fork waits for the same human node"
    );
    assert_eq!(gates.calls("t2").len(), 1, "nothing ran again");
}

#[tokio::test]
async fn edit_state_writes_a_checked_value_and_moves_the_step() {
    let gates = GatedTool::new();
    let io = io(&gates);
    let mut run = start(&io).await;
    assert_eq!(run.step(&io).await, GraphOutcome::Continue);
    let steps = run.steps();
    run.update_state("y", json!("set"), None).unwrap();
    assert_eq!(run.state()["y"], json!("set"));
    assert_eq!(run.steps(), steps + 1, "the edit is a durable change");
    assert_eq!(run.status(), &GraphStatus::Queued);
    let cp = run.checkpoint();
    assert_eq!(cp.step, steps + 1);
    assert_eq!(
        GraphRun::from_checkpoint(&cp).unwrap().state()["y"],
        json!("set")
    );
}

#[tokio::test]
async fn edit_state_cannot_write_an_approval_key() {
    let gates = GatedTool::new();
    let io = io(&gates);
    let mut run = start(&io).await;
    let steps = run.steps();
    let err = run.update_state("ok", json!(true), None).unwrap_err();
    assert!(
        matches!(&err, GraphError::Resume(m) if m.contains("`ok`") && m.contains("approval node `h`")),
        "{err}"
    );
    let err = run.update_state("ok", json!(true), Some("h")).unwrap_err();
    assert!(matches!(&err, GraphError::Resume(_)), "{err}");
    assert_eq!(run.steps(), steps, "a refused edit changes nothing");
    assert!(!run.state().contains_key("ok"));
}

#[tokio::test]
async fn edit_state_refuses_an_undeclared_or_mistyped_key() {
    let gates = GatedTool::new();
    let io = io(&gates);
    let mut run = start(&io).await;
    let err = run.update_state("nope", json!("v"), None).unwrap_err();
    assert!(
        matches!(&err, GraphError::Resume(m) if m.contains("`nope` is not declared")),
        "{err}"
    );
    let err = run.update_state("x", json!(7), None).unwrap_err();
    assert!(
        matches!(&err, GraphError::Resume(m) if m.contains("`x`")),
        "{err}"
    );
    let err = run.update_state("x", json!("v"), Some("t2")).unwrap_err();
    assert!(
        matches!(&err, GraphError::Resume(m) if m.contains("node `t2` does not write `x`")),
        "{err}"
    );
    let err = run.update_state("x", json!("v"), Some("zz")).unwrap_err();
    assert!(
        matches!(&err, GraphError::Resume(m) if m.contains("no node `zz`")),
        "{err}"
    );
    let (mut done, _, _) = history(&io).await;
    done.resume("h", json!(true)).unwrap();
    assert!(matches!(drive(&mut done, &io).await, GraphOutcome::Done(_)));
    let err = done.update_state("x", json!("v"), None).unwrap_err();
    assert!(
        matches!(&err, GraphError::Resume(m) if m.contains("finished")),
        "{err}"
    );
}

#[tokio::test]
async fn checkpoint_before_finds_the_step_that_entered_a_node() {
    let gates = GatedTool::new();
    let io = io(&gates);
    let (_, cps, _) = history(&io).await;
    let at = |cp: &Checkpoint, node: &str| {
        cp.state["branches"]
            .as_array()
            .unwrap()
            .iter()
            .any(|b| b["node"] == json!(node))
    };
    let before = GraphRun::checkpoint_before(&cps, "t2").expect("t2 was entered");
    let i = cps.iter().position(|c| c.step == before.step).unwrap();
    assert!(
        !at(before, "t2"),
        "the checkpoint before has no branch at t2"
    );
    assert!(at(&cps[i + 1], "t2"), "the next one entered t2");
    assert!(
        GraphRun::checkpoint_before(&cps, "t1").is_none(),
        "the start node has no checkpoint before it"
    );
    assert!(GraphRun::checkpoint_before(&cps, "zz").is_none());
}

#[tokio::test]
async fn replay_from_node_reruns_from_there() {
    let gates = GatedTool::new();
    let io = io(&gates);
    let (_, cps, _) = history(&io).await;
    let from = GraphRun::checkpoint_before(&cps, "h").unwrap();
    let mut replay = GraphRun::fork(from).unwrap();
    assert_eq!(drive(&mut replay, &io).await, approval());
    assert_eq!(
        replay.state()["y"],
        json!("one!"),
        "state up to the node is kept"
    );
    assert_eq!(
        gates.calls("t1").len(),
        1,
        "nodes before the replay point never run again"
    );
}

#[tokio::test]
async fn lineage_walks_parents_to_the_root() {
    let gates = GatedTool::new();
    let io = io(&gates);
    let store = InMemCheckpoints::default();
    let (root, cps, _) = history(&io).await;
    store.save(root.checkpoint()).await.unwrap();
    let a = GraphRun::fork(after_t1(&cps)).unwrap();
    store.save(a.checkpoint()).await.unwrap();
    let b = GraphRun::fork(&a.checkpoint()).unwrap();
    store.save(b.checkpoint()).await.unwrap();
    let chain = GraphRun::lineage(&store, b.run_id()).await.unwrap();
    assert_eq!(
        chain,
        vec![(a.run_id(), 0), (root.run_id(), after_t1(&cps).step)]
    );
    assert!(GraphRun::lineage(&store, root.run_id())
        .await
        .unwrap()
        .is_empty());
}

fn run_ids(v: &Value, out: &mut Vec<String>) {
    match v {
        Value::Object(m) => {
            for (k, x) in m {
                match (k.as_str(), x) {
                    ("run_id", Value::String(id)) => out.push(id.clone()),
                    _ => run_ids(x, out),
                }
            }
        }
        Value::Array(a) => a.iter().for_each(|x| run_ids(x, out)),
        _ => {}
    }
}

#[tokio::test]
async fn fork_rekeys_a_live_agent_run() {
    use troupekit_agent::MemorySource;
    use troupekit_graph::testing::{text_reply, FnModel};
    use troupekit_providers::Price;
    let price = Price {
        input_usd_micros_per_mtok: 1_000_000,
        output_usd_micros_per_mtok: 2_000_000,
        verified: true,
        long_context: None,
        search_usd_micros: 0,
    };
    let writer = "name = 'writer'\ndisplay_name = 'W'\nrole = 'Writes.'\ngoal = 'Done.'\ninstructions = 'Do it.'\nmodel = 'fake/m'\n";
    let model = Arc::new(FnModel::new("fake/m", |_| text_reply("ok", 10)));
    let agent = SimpleIo::new(ToolRegistry::new()).with_model("fake/m", model.clone(), price);
    let personas = PersonaLoader::new(vec![Arc::new(MemorySource::new(
        "test",
        [("writer.toml".to_string(), writer.to_string())],
    ))]);
    let io = SimpleGraphIo::new(Arc::new(agent), personas);
    let text = "name = 'ag'\nstart = 'a'\n[state]\nx = 'string'\n[[nodes]]\nid = 'a'\nkind = 'agent'\npersona = 'writer'\nprompt = 'Go.'\nwrites = 'x'\n[[nodes]]\nid = 'e'\nkind = 'end'\n[[edges]]\nfrom = 'a'\nto = 'e'\n";
    let tight = Limits {
        run: Some(1),
        ..Limits::default()
    };
    let flow = Flow::parse("test/ag.toml", text).unwrap();
    let mut run = GraphRun::start(flow, Map::new(), tight, &io).await.unwrap();
    assert!(
        matches!(run.step(&io).await, GraphOutcome::Waiting(_)),
        "the agent waits on its budget"
    );
    let cp = run.checkpoint();
    let mut before = Vec::new();
    run_ids(&cp.state, &mut before);
    assert!(
        before.len() >= 2,
        "the graph and its agent run carry run ids: {before:?}"
    );
    let fork = GraphRun::fork(&cp).unwrap();
    let mut after = Vec::new();
    run_ids(&fork.checkpoint().state, &mut after);
    assert_eq!(after.len(), before.len());
    assert!(
        after.iter().all(|id| !before.contains(id)),
        "no run id of the fork equals one of the original: {before:?} {after:?}"
    );
    let mut fork = fork;
    fork.raise_budget(Limits::default()).unwrap();
    assert!(matches!(drive(&mut fork, &io).await, GraphOutcome::Done(_)));
    assert_eq!(model.calls(), 1, "only the fork called the model");
}
