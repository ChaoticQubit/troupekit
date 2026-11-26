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
    let (out, ()) = tokio::join!(drive, opener);
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
            "list",
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
