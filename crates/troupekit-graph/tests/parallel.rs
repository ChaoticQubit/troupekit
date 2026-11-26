use crate::pending::bounded;
use serde_json::{json, Map, Value};
use std::sync::Arc;
use troupekit_agent::{Limits, PersonaLoader, SimpleIo};
use troupekit_graph::testing::GatedTool;
use troupekit_graph::{Flow, GraphOutcome, GraphRun, SimpleGraphIo, Workflow, GRAPH_FORMAT_FLOW};
use troupekit_tools::{ToolOutput, ToolRegistry};

const ORDERS: [[usize; 3]; 6] = [
    [0, 1, 2],
    [0, 2, 1],
    [1, 0, 2],
    [1, 2, 0],
    [2, 0, 1],
    [2, 1, 0],
];

fn io(gates: &GatedTool, replies: [Value; 3]) -> SimpleGraphIo {
    let mut tools = ToolRegistry::new();
    tools
        .register(gates.tool("s", |_| ToolOutput::text("seeded")))
        .unwrap();
    for (i, reply) in replies.into_iter().enumerate() {
        let text = reply.to_string();
        tools
            .register(gates.tool(&format!("b{i}"), move |_| ToolOutput::text(text.clone())))
            .unwrap();
    }
    SimpleGraphIo::new(Arc::new(SimpleIo::new(tools)), PersonaLoader::new(vec![]))
}

fn three_branches(ty: &str, reducer: Option<&str>) -> String {
    let reducers = reducer
        .map(|r| format!("[reducers]\nk = '{r}'\n"))
        .unwrap_or_default();
    let mut text =
        format!("name = 'par'\nstart = 's'\n[state]\nseed = 'string'\nk = '{ty}'\n{reducers}");
    text.push_str("[[nodes]]\nid = 's'\nkind = 'tool'\ntool = 's'\nwrites = 'seed'\n");
    for i in 0..3 {
        text.push_str(&format!(
            "[[nodes]]\nid = 'b{i}'\nkind = 'tool'\ntool = 'b{i}'\nwrites = 'k'\n"
        ));
    }
    text.push_str("[[nodes]]\nid = 'j'\nkind = 'join'\n[[nodes]]\nid = 'e'\nkind = 'end'\n");
    for i in 0..3 {
        text.push_str(&format!(
            "[[edges]]\nfrom = 's'\nto = 'b{i}'\n[[edges]]\nfrom = 'b{i}'\nto = 'j'\n"
        ));
    }
    text.push_str("[[edges]]\nfrom = 'j'\nto = 'e'\n");
    text
}

async fn run_in_order(
    text: &str,
    replies: [Value; 3],
    order: [usize; 3],
) -> (GraphOutcome, Vec<String>) {
    let gates = GatedTool::new();
    let io = io(&gates, replies);
    let flow = Flow::parse("test/par.toml", text).unwrap();
    let mut run = GraphRun::start(flow, Map::new(), Limits::default(), &io)
        .await
        .unwrap();
    let opener = async {
        gates.open("s");
        while (0..3)
            .map(|i| gates.in_flight(&format!("b{i}")))
            .sum::<usize>()
            < 3
        {
            tokio::task::yield_now().await;
        }
        for (done, i) in order.iter().enumerate() {
            gates.open(&format!("b{i}"));
            while gates.finished().len() < done + 2 {
                tokio::task::yield_now().await;
            }
        }
    };
    let drive = async {
        for _ in 0..20 {
            let out = run.step(&io).await;
            if out != GraphOutcome::Continue {
                return out;
            }
        }
        panic!("run did not settle in 20 steps");
    };
    let (out, ()) = bounded("the gated run", || started(&gates), async {
        tokio::join!(drive, opener)
    })
    .await;
    (out, gates.finished())
}

#[tokio::test]
async fn reducer_result_is_independent_of_finish_order() {
    let cases: [(&str, &str, [Value; 3], Value); 7] = [
        (
            "append",
            "list",
            [json!("a"), json!("b"), json!("c")],
            json!(["a", "b", "c"]),
        ),
        (
            "extend",
            "list<int>",
            [json!([1]), json!([2, 3]), json!([4])],
            json!([1, 2, 3, 4]),
        ),
        (
            "merge",
            "{a: int, b: int, c: int}",
            [json!({"a": 1}), json!({"b": 2}), json!({"c": 3})],
            json!({"a": 1, "b": 2, "c": 3}),
        ),
        ("sum", "int", [json!(1), json!(2), json!(4)], json!(7)),
        ("min", "int", [json!(5), json!(3), json!(9)], json!(3)),
        ("max", "int", [json!(5), json!(3), json!(9)], json!(9)),
        (
            "last",
            "string",
            [json!("a"), json!("b"), json!("c")],
            json!("c"),
        ),
    ];
    for (reducer, ty, replies, want) in cases {
        let text = three_branches(ty, Some(reducer));
        for order in ORDERS {
            let (out, finished) = run_in_order(&text, replies.clone(), order).await;
            let expect: Vec<String> = std::iter::once("s".to_string())
                .chain(order.iter().map(|i| format!("b{i}")))
                .collect();
            assert_eq!(
                finished, expect,
                "{reducer} {order:?}: the gates fixed the finish order"
            );
            let GraphOutcome::Done(state) = out else {
                panic!("{reducer} {order:?}: {out:?}");
            };
            assert_eq!(state["k"], want, "{reducer} {order:?}");
        }
    }
}

#[tokio::test]
async fn reduced_value_that_breaks_the_schema_fails_the_run() {
    let text = three_branches("int", Some("sum"));
    let (out, _) = run_in_order(&text, [json!(1), json!(2.5), json!(1)], [0, 1, 2]).await;
    let GraphOutcome::Failed(e) = out else {
        panic!("a sum that is not an int fails the run: {out:?}");
    };
    assert!(e.contains("node `b2`") && e.contains("reducing `k`"), "{e}");
    assert!(e.contains("does not match `k`"), "{e}");
    let (out, _) = run_in_order(&text, [json!(1), json!("x"), json!(1)], [2, 1, 0]).await;
    let GraphOutcome::Failed(e) = out else {
        panic!("a non-number write fails the run: {out:?}");
    };
    assert!(e.contains("reducing `k`") && e.contains("branch 1"), "{e}");
}

#[tokio::test]
async fn v1_fan_out_behaviour_is_unchanged() {
    let text = "name = 'par'\nstart = 's'\n[state]\nseed = 'string'\nx = 'string'\ny = 'string'\n[[nodes]]\nid = 's'\nkind = 'tool'\ntool = 's'\nwrites = 'seed'\n[[nodes]]\nid = 'b0'\nkind = 'tool'\ntool = 'b0'\nwrites = 'x'\n[[nodes]]\nid = 'b1'\nkind = 'tool'\ntool = 'b1'\nwrites = 'y'\n[[nodes]]\nid = 'j'\nkind = 'join'\n[[nodes]]\nid = 'e'\nkind = 'end'\n[[edges]]\nfrom = 's'\nto = 'b0'\n[[edges]]\nfrom = 's'\nto = 'b1'\n[[edges]]\nfrom = 'b0'\nto = 'j'\n[[edges]]\nfrom = 'b1'\nto = 'j'\n[[edges]]\nfrom = 'j'\nto = 'e'\n";
    let gates = GatedTool::new();
    for name in ["s", "b0", "b1", "b2"] {
        gates.open(name);
    }
    let io = io(&gates, [json!("left"), json!("right"), json!("unused")]);
    let workflow = Workflow::parse("test/par.toml", text).unwrap();
    let mut old = GraphRun::new(workflow.clone(), Map::new(), Limits::default(), &io)
        .await
        .unwrap();
    let mut new = GraphRun::start(Flow::from(workflow), Map::new(), Limits::default(), &io)
        .await
        .unwrap();
    let mut runs = Vec::new();
    for run in [&mut old, &mut new] {
        let mut trail = Vec::new();
        let out = loop {
            let out = run.step(&io).await;
            let mut state = run.checkpoint().state;
            assert_eq!(state["format"], 1);
            assert!(state.get("flow").is_none(), "a 0.1 run stores no flow");
            state.as_object_mut().unwrap().remove("run_id");
            trail.push(state);
            if out != GraphOutcome::Continue {
                break out;
            }
        };
        runs.push((out, trail));
    }
    assert_eq!(runs[0], runs[1]);
    let GraphOutcome::Done(state) = &runs[0].0 else {
        panic!("the fan-out finishes: {:?}", runs[0].0);
    };
    assert_eq!(
        (&state["x"], &state["y"]),
        (&json!("left"), &json!("right"))
    );
}

#[tokio::test]
async fn flow_with_a_reducer_writes_format_2_and_restores() {
    let gates = GatedTool::new();
    let io = io(&gates, [json!(1), json!(2), json!(3)]);
    let flow = Flow::parse("test/par.toml", &three_branches("int", Some("sum"))).unwrap();
    let run = GraphRun::start(flow.clone(), Map::new(), Limits::default(), &io)
        .await
        .unwrap();
    let cp = run.checkpoint();
    assert_eq!(cp.state["format"], GRAPH_FORMAT_FLOW);
    let back = GraphRun::from_checkpoint(&cp).unwrap();
    assert_eq!(back.flow(), flow);
    assert_eq!(back.checkpoint(), cp);
    let mut forged = cp.clone();
    forged.state["format"] = json!(1);
    assert!(GraphRun::from_checkpoint(&forged).is_err());
}

fn map_flow(map_extra: &str, reducer: Option<&str>) -> String {
    let reducers = reducer
        .map(|r| format!("[reducers]\nout = '{r}'\n"))
        .unwrap_or_default();
    format!(
        "name = 'm'\nstart = 'm'\n[state]\nxs = 'list'\nx = 'string'\ny = 'string'\nout = 'list'\n{reducers}[[nodes]]\nid = 'm'\nkind = 'map'\nover = 'xs'\nitem = 'x'\nbody = 'w'\nwrites = 'out'\n{map_extra}\n[[nodes]]\nid = 'w'\nkind = 'tool'\ntool = 'w'\nreads = ['x']\nwrites = 'y'\n[[nodes]]\nid = 'e'\nkind = 'end'\n[[edges]]\nfrom = 'm'\nto = 'e'\n"
    )
}

fn started(gates: &GatedTool) -> String {
    format!(
        "finished: {:?}, max in flight: {}",
        gates.finished(),
        gates.max_in_flight()
    )
}

fn map_io(gates: &GatedTool) -> SimpleGraphIo {
    let mut tools = ToolRegistry::new();
    tools
        .register(gates.tool_per_item("w", "x", |args| {
            ToolOutput::text(args["x"].as_str().unwrap().to_uppercase())
        }))
        .unwrap();
    SimpleGraphIo::new(Arc::new(SimpleIo::new(tools)), PersonaLoader::new(vec![]))
}

fn map_input(items: &[&str], out: Option<Value>) -> Map<String, Value> {
    let mut input = Map::new();
    input.insert("xs".into(), json!(items));
    if let Some(out) = out {
        input.insert("out".into(), out);
    }
    input
}

async fn map_run(text: &str, input: Map<String, Value>, io: &SimpleGraphIo) -> GraphRun {
    let flow = Flow::parse("test/m.toml", text).unwrap();
    GraphRun::start(flow, input, Limits::default(), io)
        .await
        .unwrap()
}

async fn drive_plain(run: &mut GraphRun, io: &SimpleGraphIo) -> GraphOutcome {
    for _ in 0..40 {
        let out = run.step(io).await;
        if out != GraphOutcome::Continue {
            return out;
        }
    }
    panic!("run did not settle in 40 steps");
}

async fn open_in_order(gates: &GatedTool, order: &[&str]) {
    for (done, v) in order.iter().enumerate() {
        let gate = format!("w/{v}");
        while gates.in_flight(&gate) == 0 {
            tokio::task::yield_now().await;
        }
        gates.open(&gate);
        while gates.finished().len() < done + 1 {
            tokio::task::yield_now().await;
        }
    }
}

#[tokio::test]
async fn map_runs_one_branch_per_item_in_item_order() {
    let gates = GatedTool::new();
    let io = map_io(&gates);
    let mut run = map_run(&map_flow("", None), map_input(&["a", "b", "c"], None), &io).await;
    let (out, ()) = bounded("the map run", || started(&gates), async {
        tokio::join!(drive_plain(&mut run, &io), async {
            while gates.calls("w/a").len() + gates.calls("w/b").len() + gates.calls("w/c").len() < 3
            {
                tokio::task::yield_now().await;
            }
            open_in_order(&gates, &["c", "b", "a"]).await;
        })
    })
    .await;
    let GraphOutcome::Done(state) = out else {
        panic!("{out:?}");
    };
    assert_eq!(gates.finished(), vec!["w/c", "w/b", "w/a"]);
    assert_eq!(state["out"], json!(["A", "B", "C"]));
}

#[tokio::test]
async fn map_concurrency_cap_holds() {
    let gates = GatedTool::new();
    let io = map_io(&gates);
    let items = ["a", "b", "c", "d", "e"];
    let mut run = map_run(
        &map_flow("concurrency = 2", None),
        map_input(&items, None),
        &io,
    )
    .await;
    let opener = async {
        for (done, v) in items.iter().enumerate() {
            let gate = format!("w/{v}");
            while gates.in_flight(&gate) == 0 {
                tokio::task::yield_now().await;
            }
            let live: usize = items
                .iter()
                .map(|i| gates.in_flight(&format!("w/{i}")))
                .sum();
            assert!(live <= 2, "{live} items in flight over a cap of 2");
            gates.open(&gate);
            while gates.finished().len() < done + 1 {
                tokio::task::yield_now().await;
            }
        }
    };
    let (out, ()) = bounded("the capped map run", || started(&gates), async {
        tokio::join!(drive_plain(&mut run, &io), opener)
    })
    .await;
    let GraphOutcome::Done(state) = out else {
        panic!("{out:?}");
    };
    assert_eq!(state["out"], json!(["A", "B", "C", "D", "E"]));
    assert_eq!(
        gates.max_in_flight(),
        2,
        "the cap is used and never exceeded"
    );
    assert_eq!(gates.finished().len(), 5);
}

#[tokio::test]
async fn map_items_are_durable_one_by_one() {
    use crate::pending::Store;
    use troupekit_core::store::CheckpointStore;
    let (store, gates) = (Store::default(), GatedTool::new());
    let io = map_io(&gates);
    let mut run = map_run(&map_flow("", None), map_input(&["a", "b", "c"], None), &io).await;
    let id = run.run_id();
    store.save(run.checkpoint()).await.unwrap();
    gates.open("w/a");
    let a_kept = async {
        while !store.has(id, 1, 0) {
            tokio::task::yield_now().await;
        }
    };
    bounded("the first map mark", || started(&gates), async {
        tokio::select! {
            biased;
            out = run.step_durable(&io, &store) => panic!("the step must not finish while b and c wait: {out:?}"),
            () = a_kept => {}
        }
    })
    .await;
    let mut run = GraphRun::restore(&store, id).await.unwrap();
    gates.open("w/b");
    gates.open("w/c");
    let mut out = GraphOutcome::Continue;
    for _ in 0..40 {
        out = run.step_durable(&io, &store).await.unwrap();
        store.save(run.checkpoint()).await.unwrap();
        if out != GraphOutcome::Continue {
            break;
        }
    }
    let GraphOutcome::Done(state) = out else {
        panic!("{out:?}");
    };
    assert_eq!(state["out"], json!(["A", "B", "C"]));
    assert_eq!(
        gates.calls("w/a").len(),
        1,
        "a finished item never runs again"
    );
    assert_eq!(gates.calls("w/b").len(), 2);
    assert_eq!(gates.calls("w/b")[0], gates.calls("w/b")[1]);
}

#[tokio::test]
async fn map_over_more_than_max_items_fails_cleanly() {
    let gates = GatedTool::new();
    let io = map_io(&gates);
    let mut run = map_run(
        &map_flow("max_items = 2", None),
        map_input(&["a", "b", "c"], None),
        &io,
    )
    .await;
    let GraphOutcome::Failed(e) = run.step(&io).await else {
        panic!("a list over the cap fails the run");
    };
    assert!(
        e.contains("node `m`") && e.contains("3 items") && e.contains("cap of 2"),
        "{e}"
    );
    assert!(gates.finished().is_empty() && gates.calls("w/a").is_empty());
}

#[tokio::test]
async fn map_over_empty_list_writes_empty_result() {
    let gates = GatedTool::new();
    let io = map_io(&gates);
    let mut run = map_run(&map_flow("", None), map_input(&[], None), &io).await;
    let GraphOutcome::Done(state) = drive_plain(&mut run, &io).await else {
        panic!("an empty map finishes");
    };
    assert_eq!(state["out"], json!([]));
}

#[tokio::test]
async fn map_with_reducer_reduces_in_item_order() {
    let gates = GatedTool::new();
    let io = map_io(&gates);
    let input = map_input(&["a", "b", "c"], Some(json!(["z"])));
    let mut run = map_run(&map_flow("", Some("append")), input, &io).await;
    let (out, ()) = bounded("the reduced map run", || started(&gates), async {
        tokio::join!(drive_plain(&mut run, &io), async {
            while gates.calls("w/a").len() + gates.calls("w/b").len() + gates.calls("w/c").len() < 3
            {
                tokio::task::yield_now().await;
            }
            open_in_order(&gates, &["b", "c", "a"]).await;
        })
    })
    .await;
    let GraphOutcome::Done(state) = out else {
        panic!("{out:?}");
    };
    assert_eq!(state["out"], json!(["z", "A", "B", "C"]));
}

#[tokio::test]
async fn item_key_is_branch_local() {
    let gates = GatedTool::new();
    for v in ["a", "b"] {
        gates.open(&format!("w/{v}"));
    }
    let io = map_io(&gates);
    let mut run = map_run(&map_flow("", None), map_input(&["a", "b"], None), &io).await;
    let first = run.step(&io).await;
    assert_eq!(first, GraphOutcome::Done(run.state().clone()));
    let state = run.state();
    assert_eq!(state["out"], json!(["A", "B"]));
    assert!(
        !state.contains_key("x"),
        "the item key never reaches shared state"
    );
    assert!(
        !state.contains_key("y"),
        "the body's result never reaches shared state"
    );
    assert_eq!(gates.calls("w/a").len(), 1);
    assert_eq!(gates.calls("w/b").len(), 1);
}

#[tokio::test]
async fn reduced_write_survives_a_transient_sibling_failure() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use troupekit_graph::testing::FnTool;
    use troupekit_tools::ToolError;
    let text = "name = 'sum'\nstart = 's'\n[state]\nseed = 'string'\nn = 'number'\n[reducers]\nn = 'sum'\n[[nodes]]\nid = 's'\nkind = 'tool'\ntool = 's'\nwrites = 'seed'\n[[nodes]]\nid = 'a'\nkind = 'tool'\ntool = 'a'\nwrites = 'n'\n[[nodes]]\nid = 'b'\nkind = 'tool'\ntool = 'b'\nwrites = 'n'\n[[nodes]]\nid = 'j'\nkind = 'join'\n[[nodes]]\nid = 'e'\nkind = 'end'\n[[edges]]\nfrom = 's'\nto = 'a'\n[[edges]]\nfrom = 's'\nto = 'b'\n[[edges]]\nfrom = 'a'\nto = 'j'\n[[edges]]\nfrom = 'b'\nto = 'j'\n[[edges]]\nfrom = 'j'\nto = 'e'\n";
    let mut tools = ToolRegistry::new();
    tools
        .register(FnTool::new("s", |_, _| async {
            Ok(ToolOutput::text("seeded"))
        }))
        .unwrap();
    tools
        .register(FnTool::new("a", |_, _| async { Ok(ToolOutput::text("2")) }))
        .unwrap();
    let b_calls = Arc::new(AtomicUsize::new(0));
    tools
        .register(FnTool::new("b", move |_, _| {
            let first = b_calls.fetch_add(1, Ordering::SeqCst) == 0;
            async move {
                if first {
                    Err(ToolError::Failed("database: max clients reached".into()))
                } else {
                    Ok(ToolOutput::text("3"))
                }
            }
        }))
        .unwrap();
    let io = SimpleGraphIo::new(Arc::new(SimpleIo::new(tools)), PersonaLoader::new(vec![]));
    let mut input = Map::new();
    input.insert("n".into(), json!(10));
    let flow = Flow::parse("test/sum.toml", text).unwrap();
    let mut run = GraphRun::start(flow, input, Limits::default(), &io)
        .await
        .unwrap();
    let GraphOutcome::Failed(_) = drive_plain(&mut run, &io).await else {
        panic!("b's first call fails the run");
    };
    assert!(run.retryable(), "a store failure is retryable");
    run.retry(None).unwrap();
    let GraphOutcome::Done(state) = drive_plain(&mut run, &io).await else {
        panic!("the retried run finishes");
    };
    assert_eq!(
        state["n"],
        json!(15),
        "a's 2 and b's 3 both reach the total"
    );
}

#[tokio::test]
async fn merge_from_a_map_and_a_sibling_in_one_step_refuses_a_shared_field() {
    let text = "name = 'mm'\nstart = 's'\n[state]\nseed = 'string'\nxs = 'list'\nx = 'string'\ny = '{a: string}'\nobj = '{a: string}'\n[reducers]\nobj = 'merge'\n[[nodes]]\nid = 's'\nkind = 'tool'\ntool = 's'\nwrites = 'seed'\n[[nodes]]\nid = 'm'\nkind = 'map'\nover = 'xs'\nitem = 'x'\nbody = 'w'\nwrites = 'obj'\n[[nodes]]\nid = 'w'\nkind = 'tool'\ntool = 'w'\nreads = ['x']\nwrites = 'y'\n[[nodes]]\nid = 't'\nkind = 'tool'\ntool = 't'\nwrites = 'obj'\n[[nodes]]\nid = 'j'\nkind = 'join'\n[[nodes]]\nid = 'e'\nkind = 'end'\n[[edges]]\nfrom = 's'\nto = 'm'\n[[edges]]\nfrom = 's'\nto = 't'\n[[edges]]\nfrom = 'm'\nto = 'j'\n[[edges]]\nfrom = 't'\nto = 'j'\n[[edges]]\nfrom = 'j'\nto = 'e'\n";
    let gates = GatedTool::new();
    for name in ["s", "t", "w/one"] {
        gates.open(name);
    }
    let mut tools = ToolRegistry::new();
    tools
        .register(gates.tool("s", |_| ToolOutput::text("seeded")))
        .unwrap();
    tools
        .register(gates.tool("t", |_| ToolOutput::text(r#"{"a":"sibling"}"#)))
        .unwrap();
    tools
        .register(gates.tool_per_item("w", "x", |_| ToolOutput::text(r#"{"a":"item"}"#)))
        .unwrap();
    let io = SimpleGraphIo::new(Arc::new(SimpleIo::new(tools)), PersonaLoader::new(vec![]));
    let mut run = map_run(text, map_input(&["one"], None), &io).await;
    let out = bounded(
        "the merge run",
        || started(&gates),
        drive_plain(&mut run, &io),
    )
    .await;
    let GraphOutcome::Failed(e) = out else {
        panic!("a field written twice in one step fails the run: {out:?}");
    };
    assert!(e.contains("both write `a` in one step"), "{e}");
}

#[tokio::test]
async fn map_item_that_breaks_the_item_type_fails_the_run() {
    let gates = GatedTool::new();
    let io = map_io(&gates);
    let mut input = Map::new();
    input.insert("xs".into(), json!([1, 2]));
    let text = map_flow("", None).replace("xs = 'list'", "xs = 'list<int>'");
    let mut run = map_run(&text, input, &io).await;
    let GraphOutcome::Failed(e) = run.step(&io).await else {
        panic!("an item of the wrong type fails the run");
    };
    assert!(
        e.contains("node `m`") && e.contains("item 0") && e.contains("`x`"),
        "{e}"
    );
    assert!(gates.calls("w/a").is_empty(), "no item runs");
}

#[tokio::test]
async fn shared_budget_split_is_refused_until_it_is_run() {
    let gates = GatedTool::new();
    let io = map_io(&gates);
    let text = "name = 'b'\nstart = 't'\n[budget]\nsplit = 'shared'\n[state]\ny = 'string'\n[[nodes]]\nid = 't'\nkind = 'tool'\ntool = 'w'\nwrites = 'y'\n[[nodes]]\nid = 'e'\nkind = 'end'\n[[edges]]\nfrom = 't'\nto = 'e'\n";
    let flow = Flow::parse("test/b.toml", text).unwrap();
    let err = GraphRun::start(flow, Map::new(), Limits::default(), &io)
        .await
        .unwrap_err();
    assert!(
        matches!(&err, troupekit_graph::GraphError::Setup(m) if m.contains("shared budget split")),
        "{err}"
    );
}

fn chain_io(gates: &GatedTool, names: &[&str]) -> SimpleGraphIo {
    let mut tools = ToolRegistry::new();
    for name in names {
        let reply = format!("{name}!");
        tools
            .register(gates.tool(name, move |_| ToolOutput::text(reply.clone())))
            .unwrap();
    }
    SimpleGraphIo::new(Arc::new(SimpleIo::new(tools)), PersonaLoader::new(vec![]))
}

fn tool_node(id: &str, extra: &str) -> String {
    format!("[[nodes]]\nid = '{id}'\nkind = 'tool'\ntool = '{id}'\nwrites = '{id}'\n{extra}")
}

fn graph(state: &[&str], nodes: &[String], edges: &[(&str, &str)]) -> String {
    let mut text = "name = 'g'\nstart = 's'\n[state]\n".to_string();
    for key in state {
        text.push_str(&format!("{key} = 'string?'\n"));
    }
    for n in nodes {
        text.push_str(n);
    }
    for (from, to) in edges {
        text.push_str(&format!("[[edges]]\nfrom = '{from}'\nto = '{to}'\n"));
    }
    text
}

fn join_node(id: &str, mode: &str) -> String {
    format!("[[nodes]]\nid = '{id}'\nkind = 'join'\n{mode}")
}

const END_NODE: &str = "[[nodes]]\nid = 'e'\nkind = 'end'\n";

fn open_all(gates: &GatedTool, names: &[&str]) {
    for name in names {
        gates.open(name);
    }
}

async fn finish(
    text: &str,
    io: &SimpleGraphIo,
    gates: &GatedTool,
) -> (Map<String, Value>, GraphRun) {
    let flow = Flow::parse("test/g.toml", text).unwrap();
    let mut run = GraphRun::start(flow, Map::new(), Limits::default(), io)
        .await
        .unwrap();
    let out = bounded("the run", || started(gates), drive_plain(&mut run, io)).await;
    let GraphOutcome::Done(state) = out else {
        panic!("{out:?}");
    };
    (state, run)
}

#[tokio::test]
async fn nested_fan_out_joins_inner_then_outer() {
    let names = ["s", "a", "i", "c", "d", "k"];
    let gates = GatedTool::new();
    open_all(&gates, &names);
    let io = chain_io(&gates, &names);
    let nodes: Vec<String> = names
        .iter()
        .map(|n| tool_node(n, ""))
        .chain([
            join_node("j2", ""),
            join_node("j1", ""),
            END_NODE.to_string(),
        ])
        .collect();
    let text = graph(
        &names,
        &nodes,
        &[
            ("s", "a"),
            ("s", "i"),
            ("i", "c"),
            ("i", "d"),
            ("c", "j2"),
            ("d", "j2"),
            ("j2", "k"),
            ("k", "j1"),
            ("a", "j1"),
            ("j1", "e"),
        ],
    );
    let flow = Flow::parse("test/g.toml", &text).unwrap();
    assert!(!flow.is_v1(), "a nested fan-out is a 0.5 flow");
    let (state, run) = finish(&text, &io, &gates).await;
    let order = gates.finished();
    let at = |n: &str| order.iter().position(|o| o == n).unwrap();
    assert!(
        at("c") < at("k") && at("d") < at("k"),
        "inner join closes before k: {order:?}"
    );
    assert_eq!(order.len(), names.len(), "every tool ran once: {order:?}");
    for n in names {
        assert_eq!(state[n], json!(format!("{n}!")));
    }
    assert_eq!(run.checkpoint().state["format"], json!(GRAPH_FORMAT_FLOW));
}

#[tokio::test]
async fn any_join_takes_the_lowest_index_arrival() {
    let names = ["s", "a", "b", "after"];
    let gates = GatedTool::new();
    let io = chain_io(&gates, &names);
    let nodes: Vec<String> = names
        .iter()
        .map(|n| tool_node(n, ""))
        .chain([join_node("j", "mode = 'any'\n"), END_NODE.to_string()])
        .collect();
    let text = graph(
        &names,
        &nodes,
        &[
            ("s", "a"),
            ("s", "b"),
            ("a", "j"),
            ("b", "j"),
            ("j", "after"),
            ("after", "e"),
        ],
    );
    let flow = Flow::parse("test/g.toml", &text).unwrap();
    let mut run = GraphRun::start(flow, Map::new(), Limits::default(), &io)
        .await
        .unwrap();
    let opener = async {
        gates.open("s");
        while gates.in_flight("a") + gates.in_flight("b") < 2 {
            tokio::task::yield_now().await;
        }
        gates.open("b");
        while gates.finished().len() < 2 {
            tokio::task::yield_now().await;
        }
        gates.open("a");
        gates.open("after");
    };
    let (out, ()) = bounded("the any-join run", || started(&gates), async {
        tokio::join!(drive_plain(&mut run, &io), opener)
    })
    .await;
    let GraphOutcome::Done(state) = out else {
        panic!("{out:?}");
    };
    assert_eq!(gates.finished(), vec!["s", "b", "a", "after"]);
    assert_eq!(
        gates.calls("after").len(),
        1,
        "an any-join continues one branch"
    );
    assert_eq!(
        (&state["a"], &state["b"]),
        (&json!("a!"), &json!("b!")),
        "same-step arrivals keep their writes"
    );
}

#[tokio::test]
async fn any_join_drops_losing_branches_and_their_pending_writes() {
    let names = ["s", "a", "b1", "b2", "after"];
    let gates = GatedTool::new();
    open_all(&gates, &names);
    open_all(&gates, &["w/x", "w/y"]);
    let mut tools = ToolRegistry::new();
    for name in names {
        let reply = format!("{name}!");
        tools
            .register(gates.tool(name, move |_| ToolOutput::text(reply.clone())))
            .unwrap();
    }
    tools
        .register(gates.tool_per_item("w", "x", |args| {
            ToolOutput::text(args["x"].as_str().unwrap().to_string())
        }))
        .unwrap();
    let io = SimpleGraphIo::new(Arc::new(SimpleIo::new(tools)), PersonaLoader::new(vec![]));
    let mut nodes: Vec<String> = names.iter().map(|n| tool_node(n, "")).collect();
    nodes.push(
        "[[nodes]]\nid = 'm'\nkind = 'map'\nover = 'xs'\nitem = 'x'\nbody = 'w'\nwrites = 'out'\n"
            .into(),
    );
    nodes.push(
        "[[nodes]]\nid = 'w'\nkind = 'tool'\ntool = 'w'\nreads = ['x']\nwrites = 'y'\n".into(),
    );
    nodes.push(join_node("j", "mode = 'any'\n"));
    nodes.push(END_NODE.into());
    let mut text = graph(
        &names,
        &nodes,
        &[
            ("s", "a"),
            ("s", "b1"),
            ("b1", "m"),
            ("m", "b2"),
            ("a", "j"),
            ("b2", "j"),
            ("j", "after"),
            ("after", "e"),
        ],
    );
    text = text.replace(
        "[state]\n",
        "[state]\nxs = 'list'\nx = 'string'\ny = 'string'\nout = 'list'\n",
    );
    let flow = Flow::parse("test/g.toml", &text).unwrap();
    let mut input = Map::new();
    input.insert("xs".into(), json!(["x", "y"]));
    let mut run = GraphRun::start(flow, input, Limits::default(), &io)
        .await
        .unwrap();
    let out = bounded(
        "the any-join run",
        || started(&gates),
        drive_plain(&mut run, &io),
    )
    .await;
    let GraphOutcome::Done(state) = out else {
        panic!("{out:?}");
    };
    assert_eq!(state["a"], json!("a!"));
    assert_eq!(
        state["b1"],
        json!("b1!"),
        "a loser's write finished before the close stays"
    );
    assert!(
        gates.calls("w/x").is_empty() && gates.calls("w/y").is_empty(),
        "dropped map items never run"
    );
    assert!(
        gates.calls("b2").is_empty(),
        "the loser's next node never runs"
    );
    assert_eq!(state.get("out"), None, "the dropped map writes nothing");
    assert_eq!(gates.calls("after").len(), 1);
}

#[tokio::test]
async fn deferred_node_waits_for_all_other_work() {
    let names = ["s", "d", "a1", "a2", "after"];
    let gates = GatedTool::new();
    open_all(&gates, &names);
    let io = chain_io(&gates, &names);
    let nodes: Vec<String> = names
        .iter()
        .map(|n| tool_node(n, if *n == "d" { "defer = true\n" } else { "" }))
        .chain([join_node("j", ""), END_NODE.to_string()])
        .collect();
    let text = graph(
        &names,
        &nodes,
        &[
            ("s", "d"),
            ("s", "a1"),
            ("a1", "a2"),
            ("d", "j"),
            ("a2", "j"),
            ("j", "after"),
            ("after", "e"),
        ],
    );
    finish(&text, &io, &gates).await;
    assert_eq!(gates.finished(), vec!["s", "a1", "a2", "d", "after"]);
}

#[tokio::test]
async fn deferred_node_runs_once() {
    let names = ["s", "d1", "d2", "a"];
    let gates = GatedTool::new();
    let io = chain_io(&gates, &names);
    let nodes: Vec<String> = names
        .iter()
        .map(|n| {
            tool_node(
                n,
                if n.starts_with('d') {
                    "defer = true\n"
                } else {
                    ""
                },
            )
        })
        .chain([join_node("j", ""), END_NODE.to_string()])
        .collect();
    let text = graph(
        &names,
        &nodes,
        &[
            ("s", "d1"),
            ("s", "d2"),
            ("s", "a"),
            ("d1", "j"),
            ("d2", "j"),
            ("a", "j"),
            ("j", "e"),
        ],
    );
    let flow = Flow::parse("test/g.toml", &text).unwrap();
    let mut run = GraphRun::start(flow, Map::new(), Limits::default(), &io)
        .await
        .unwrap();
    let opener = async {
        open_all(&gates, &["s", "a"]);
        while gates.in_flight("d1") + gates.in_flight("d2") < 2 {
            tokio::task::yield_now().await;
        }
        assert_eq!(
            gates.finished(),
            vec!["s", "a"],
            "deferred nodes start after a"
        );
        open_all(&gates, &["d1", "d2"]);
    };
    let (out, ()) = bounded("the deferred run", || started(&gates), async {
        tokio::join!(drive_plain(&mut run, &io), opener)
    })
    .await;
    assert!(matches!(out, GraphOutcome::Done(_)), "{out:?}");
    assert_eq!(
        (gates.calls("d1").len(), gates.calls("d2").len()),
        (1, 1),
        "each deferred node runs once"
    );
    assert_eq!(
        gates.max_in_flight(),
        2,
        "parked deferred nodes start together"
    );
}

#[tokio::test]
async fn deferred_agent_with_a_model_override_runs_last_on_its_model() {
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
    let gates = GatedTool::new();
    open_all(&gates, &["s", "a"]);
    let mut tools = ToolRegistry::new();
    for name in ["s", "a"] {
        let reply = format!("{name}!");
        tools
            .register(gates.tool(name, move |_| ToolOutput::text(reply.clone())))
            .unwrap();
    }
    let base = Arc::new(FnModel::new("fake/m", |_| text_reply("base", 10)));
    let other = Arc::new(FnModel::new("fake/other", |_| text_reply("other", 10)));
    let agent = SimpleIo::new(tools)
        .with_model("fake/m", base.clone(), price)
        .with_model("fake/other", other.clone(), price);
    let personas = PersonaLoader::new(vec![Arc::new(MemorySource::new(
        "test",
        [(
            "writer.toml".to_string(),
            "name = 'writer'\ndisplay_name = 'W'\nrole = 'Writes.'\ngoal = 'Done.'\ninstructions = 'Do it.'\nmodel = 'fake/m'\n".to_string(),
        )],
    ))]);
    let io = SimpleGraphIo::new(Arc::new(agent), personas);
    let text = graph(
        &["s", "a", "w"],
        &[
            tool_node("s", ""),
            tool_node("a", ""),
            "[[nodes]]\nid = 'w'\nkind = 'agent'\npersona = 'writer'\nprompt = 'Write.'\nwrites = 'w'\nmodel = 'fake/other'\ndefer = true\n".into(),
            join_node("j", ""),
            END_NODE.into(),
        ],
        &[("s", "a"), ("s", "w"), ("a", "j"), ("w", "j"), ("j", "e")],
    );
    let (state, _) = finish(&text, &io, &gates).await;
    assert_eq!(state["w"], json!("other"));
    assert_eq!(state["a"], json!("a!"));
    assert_eq!((base.calls(), other.calls()), (0, 1));
}
