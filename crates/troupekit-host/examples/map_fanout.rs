use serde_json::{json, Map, Value};
use std::fmt::Write as _;
use std::sync::Arc;
use std::time::{Duration, Instant};
use troupekit_agent::{Limits, PersonaLoader, SimpleIo};
use troupekit_core::inmem::InMemCheckpoints;
use troupekit_core::store::CheckpointStore;
use troupekit_graph::flow::Flow;
use troupekit_graph::testing::FnTool;
use troupekit_graph::{GraphOutcome, GraphRun, SimpleGraphIo};
use troupekit_tools::{ToolOutput, ToolRegistry};

const WIDTH_MAX: usize = 253;
const RUNS_DEFAULT: usize = 5;
const RUNS_MAX: usize = 1_000;

/// The two run shapes compared: one map node over `width` items, or a
/// static fan-out of `width` tool branches into one join. Both run the
/// same tool once per item or branch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Shape {
    Map,
    FanOut,
}

fn map_toml(width: usize) -> String {
    format!(
        r#"
name = "map"
start = "seed"
[state]
goal = "string"
items = "list"
item = "string"
out = "string"
results = "list"
[[nodes]]
id = "seed"
kind = "tool"
tool = "seed"
writes = "items"
[[nodes]]
id = "dig"
kind = "map"
over = "items"
item = "item"
body = "work"
writes = "results"
max_items = {width}
concurrency = {width}
[[nodes]]
id = "work"
kind = "tool"
tool = "work"
reads = ["item"]
writes = "out"
[[nodes]]
id = "done"
kind = "end"
[[edges]]
from = "seed"
to = "dig"
[[edges]]
from = "dig"
to = "done"
"#
    )
}

fn fan_out_toml(width: usize) -> String {
    let mut s = String::from("name = \"fan\"\nstart = \"seed\"\n[state]\ngoal = \"string\"\n");
    for i in 0..width {
        writeln!(s, "r{i} = \"string\"").expect("a string grows");
    }
    s.push_str("[[nodes]]\nid = \"seed\"\nkind = \"tool\"\ntool = \"seed\"\n");
    for i in 0..width {
        writeln!(
            s,
            "[[nodes]]\nid = \"w{i}\"\nkind = \"tool\"\ntool = \"work\"\nwrites = \"r{i}\""
        )
        .expect("a string grows");
    }
    s.push_str(
        "[[nodes]]\nid = \"join\"\nkind = \"join\"\n[[nodes]]\nid = \"done\"\nkind = \"end\"\n",
    );
    for i in 0..width {
        writeln!(
            s,
            "[[edges]]\nfrom = \"seed\"\nto = \"w{i}\"\n[[edges]]\nfrom = \"w{i}\"\nto = \"join\""
        )
        .expect("a string grows");
    }
    s.push_str("[[edges]]\nfrom = \"join\"\nto = \"done\"\n");
    s
}

fn io(width: usize) -> SimpleGraphIo {
    let items: Vec<String> = (0..width).map(|i| format!("item-{i}")).collect();
    let seed = serde_json::to_string(&items).expect("a list serializes");
    let mut tools = ToolRegistry::new();
    tools
        .register(FnTool::new("seed", move |_, _| {
            let seed = seed.clone();
            async move { Ok(ToolOutput::text(seed)) }
        }))
        .expect("valid tool");
    tools
        .register(FnTool::new("work", |_, _| async {
            Ok(ToolOutput::text("\"ok\""))
        }))
        .expect("valid tool");
    SimpleGraphIo::new(
        Arc::new(SimpleIo::new(tools)),
        PersonaLoader::new(Vec::new()),
    )
}

/// Times `runs` whole runs of `shape` at `width` items or branches, each
/// from start to done on a fresh history store with `step_durable`, so
/// every item or branch keeps its write. Input: width 1 to 253 (the node
/// cap minus seed, join and end), runs 1 to 1 000. Output: sorted times.
pub async fn measure(shape: Shape, width: usize, runs: usize) -> Vec<Duration> {
    assert!(
        (1..=WIDTH_MAX).contains(&width),
        "width must be 1 to {WIDTH_MAX}"
    );
    assert!(
        (1..=RUNS_MAX).contains(&runs),
        "runs must be 1 to {RUNS_MAX}"
    );
    let text = match shape {
        Shape::Map => map_toml(width),
        Shape::FanOut => fan_out_toml(width),
    };
    let flow = Flow::parse("perf.toml", &text).expect("valid flow");
    let io = io(width);
    let input: Map<String, Value> = [("goal".to_string(), json!("time"))].into_iter().collect();
    let mut times = Vec::with_capacity(runs);
    for _ in 0..runs {
        let store = InMemCheckpoints::with_history();
        let start = Instant::now();
        let mut run = GraphRun::start(flow.clone(), input.clone(), Limits::default(), &io)
            .await
            .expect("start");
        store.save(run.checkpoint()).await.expect("save");
        let state = loop {
            match run.step_durable(&io, &store).await.expect("step") {
                GraphOutcome::Continue => store.save(run.checkpoint()).await.expect("save"),
                GraphOutcome::Done(state) => break state,
                other => panic!("{shape:?} run stopped early: {other:?}"),
            }
        };
        store.save(run.checkpoint()).await.expect("save");
        times.push(start.elapsed());
        let filled = match shape {
            Shape::Map => state["results"].as_array().map_or(0, Vec::len),
            Shape::FanOut => (0..width)
                .filter(|i| state[&format!("r{i}")] == json!("ok"))
                .count(),
        };
        assert_eq!(filled, width, "every item or branch reached the state");
    }
    times.sort();
    assert_eq!(times.len(), runs);
    times
}

/// Returns the `pct` percentile of the sorted, non-empty `times`.
/// Input: 0 to 100. Output: the entry at that rank, rounded down.
pub fn percentile(times: &[Duration], pct: usize) -> Duration {
    assert!(
        !times.is_empty() && pct <= 100,
        "percentile of an empty list or above 100"
    );
    times[(times.len() - 1) * pct / 100]
}

/// Prints p50 and p99 of whole runs of a 253-item tool map next to a
/// 253-branch static fan-out, both stepped durably on a history store.
/// Input: the run count as the first argument, default 5.
#[tokio::main(flavor = "multi_thread")]
async fn main() {
    let runs = std::env::args()
        .nth(1)
        .map(|arg| arg.parse::<usize>().expect("the run count is a number"))
        .unwrap_or(RUNS_DEFAULT);
    println!("width {WIDTH_MAX} runs {runs}");
    for (label, shape) in [("map", Shape::Map), ("fan-out", Shape::FanOut)] {
        let times = measure(shape, WIDTH_MAX, runs).await;
        println!(
            "{label:>8}  run p50 {:>10} p99 {:>10}",
            ms(percentile(&times, 50)),
            ms(percentile(&times, 99)),
        );
    }
}

fn ms(d: Duration) -> String {
    format!("{:.3} ms", d.as_secs_f64() * 1000.0)
}
