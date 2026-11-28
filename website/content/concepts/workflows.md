---
title: Workflows and graphs
description: Chain agents and tool fns with the Graph builder, or write durable TOML workflows with routers, human pauses and joins.
---

# Workflows and graphs

There are two ways to run several steps. The `Graph` builder chains agents and tool fns in a line. A workflow TOML file adds routers, human pauses, fan-out and joins. Both run as a checkpointable graph run that can resume after a crash.

Source: `crates/troupekit-kit/src/graph.rs`, `crates/troupekit-graph/src/workflow.rs`

## The `Graph` builder

Each node reads the output of the node before it; the first reads the input, and the last node's output is the answer.

```rust
/// Trims the input.
#[tool]
fn clean_text(input: String) -> String {
    input.trim().to_string()
}

/// Counts characters of the cleaned text.
#[tool]
fn count_chars(clean: String) -> usize {
    clean.chars().count()
}
```

Source: `crates/troupekit-kit/tests/graph.rs`

```rust
let report = Graph::new("lab3")
    .step("clean", clean_text)
    .step("count", count_chars)
    .run_with("  abc ")
    .await
    .unwrap();
assert_eq!(report.output(), &json!("3"));
assert_eq!(report.state()["clean"], json!("abc"));
assert_eq!(report.state()["input"], json!("  abc "));
```

Source: `crates/troupekit-kit/tests/graph.rs`, test `no_edges_means_chain_in_order`

- `.agent(id, agent)` adds an agent node. The agent keeps its model; its tools and evals join the graph, and its storage, retries and events become the graph's. An agent with its own limits, laws or ledger fails `build`, so no spend cap is dropped.
- `.step(id, tool)` adds a fn node. Its argument names must equal the ids of the nodes that feed it, `input` for the first node.
- `.edge(from, to)` sets the order. Edges must form one chain; with no edges the nodes run in the order they were added.
- `.checkpoints(..)`, `.limits(..)`, `.ledger(..)`, `.max_steps(..)` and `.retries(..)` work as on an agent.

`run` returns the output value. `run_with` returns a `GraphReport` with `output()`, `state()`, `usage()`, `steps()`, `run_id()` and `trace_path()`. `BuiltGraph::resume(run_id)` continues a run.

Source: `crates/troupekit-kit/src/graph.rs`

## Workflow TOML

A workflow has a name, a start node, typed shared state, nodes and edges:

```toml
name = "plan-week"
start = "plan"
max_steps = 50
max_retries = 3

[state]
goal = "string"
plan = "{steps: list}"
approved = "bool"
note = "string"

[[nodes]]
id = "plan"
kind = "agent"
persona = "planner"
prompt = "Plan the week."
reads = ["goal"]
writes = "plan"

[[nodes]]
id = "approve"
kind = "human"
ask = "approval"
prompt = "Approve this plan?"
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
```

Source: `crates/troupekit-graph/tests/workflow.rs`, constant `FULL`

## Node kinds

| Kind | Keys | Does |
|---|---|---|
| `agent` | `persona`, `prompt`, `reads`, `writes` | runs a persona on the state keys it reads |
| `tool` | `tool`, `reads`, `writes` | calls one tool |
| `human` | `ask`, `prompt`, `reads`, `writes` | pauses the run until someone answers; `ask` is `approval` or `question` |
| `router` | `condition` | picks one out-edge by the label its condition returns |
| `join` | none | waits for every branch of a fan-out |
| `end` | none | finishes the run |

Source: `crates/troupekit-graph/src/workflow.rs`

A human node with `ask = "approval"` must write a `bool` key. The built-in condition `flag(key)` returns `"true"` or `"false"` for a `bool` state key; register more with `Conditions::register`. A condition must be deterministic, so a replayed step routes the same way.

Source: `crates/troupekit-graph/src/workflow.rs`, `crates/troupekit-graph/src/condition.rs`

State types use the persona output syntax: `string`, `int`, `number`, `bool`, `list`, `list<T>`, `{a, b: T}`, `enum(x, y)`, with `?` for nullable.

Source: `crates/troupekit-agent/src/schema.rs`

`Workflow::parse` checks fields, node kinds, state keys, edges, reachability, cycles without a router and fan-out joins, and returns the first problem.

| Key | Rule | Default |
|---|---|---|
| `name` | 1 to 64 of `[a-z0-9-]`, not starting with `-` | required |
| `start` | a node id | required |
| `max_steps` | 1 to 10 000 | 200 |
| `max_retries` | 0 to 50 | 5 |
| `trace`, `trace_content`, `trace_file` | see [Tracing and budgets](tracing.md) | off |

Source: `crates/troupekit-graph/src/workflow.rs`

## Hosting runs

`troupekit-host` runs workflows from a leased job queue. A `Runner` claims a job, steps the run from its latest checkpoint and saves it:

- worker mode, `runner.work(opts, shutdown)`, ticks in a loop;
- step mode, `runner.tick(opts)`, processes due jobs once, for one serverless request.

`runner.resume(run_id, node, answer)` answers a human node. `runner.raise_budget(run_id, limits)` raises the limits of a run paused on its budget.

Source: `crates/troupekit-host/src/runner.rs`, `crates/troupekit-host/src/lib.rs`

This is how the kill and resume example starts a run:

```rust
async fn start(troupe: &Troupe) -> RunId {
    let wf = Workflow::parse("three.toml", WORKFLOW).expect("valid workflow");
    let input: Map<String, Value> = [("goal".to_string(), json!("survive"))]
        .into_iter()
        .collect();
    let io = io(troupe, false);
    let run = GraphRun::new(wf, input, Limits::default(), io.as_ref())
        .await
        .expect("run");
    troupe.runner(io).enqueue(&run).await.expect("enqueue");
    run.run_id()
}
```

Source: `crates/troupekit/examples/kill_resume.rs`

See [Kill and resume a workflow](../guides/kill-resume.md) for the whole program.

Coming in 0.5: team, crew and handoff patterns in TOML, parallel branches with declared reducers, fan-out over lists, and forking a run from any checkpoint.
