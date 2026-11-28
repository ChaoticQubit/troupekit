---
title: Tracing and budgets
description: Trace every model call, tool call and graph step with one setting, and cap spend per call, run, scope and user.
---

# Tracing and budgets

## Turn tracing on

`.tracing(true)` traces every category for `build` and every run of an agent or graph. Spans go to `./traces/<name>-<trace id>.jsonl`, or to `$TROUPEKIT_TRACE_DIR` when it is set. Tracing is off by default.

Source: `crates/troupekit-kit/src/trace.rs`, `crates/troupekit-trace/src/sink.rs`

| Method | Meaning | Default |
|---|---|---|
| `tracing(on)` | every category, or off | off |
| `trace(categories)` | only the listed categories; an empty list turns tracing off | off |
| `trace_content(on)` | keep prompt, reply, tool argument, query and document text in spans | off: spans hold only byte counts |
| `trace_to(sink)` | send records to a `TraceSink` instead of a file | none |
| `trace_file(path)` | write records to the new file `path`; an existing file is refused at `build` | none |

Secrets are scrubbed whether content is kept or not.

Source: `crates/troupekit-kit/src/trace.rs`

The categories are `Agents`, `Models`, `Tools`, `Memory`, `Knowledge`, `Graphs`, `Laws`, `Evals`, `Budget`, `Store` and `Routing`.

Source: `crates/troupekit-trace/src/category.rs`

## Tracing from the macro

```rust
/// Echo the user.
#[agent(
    model = "fake/m",
    trace = [Agents, troupekit::prelude::Trace::Models],
    trace_content = false,
    private = false,
    route = office(),
    tier = "large",
)]
fn routed_echo() {}

/// Echo the user.
#[agent(model = "fake/m", trace = all)]
fn traced_echo() {}
```

Source: `crates/troupekit-kit/tests/macros_agent.rs`

Persona and workflow TOML files take the same three keys: `trace`, a list of lowercase category names such as `['graphs', 'tools']`, `trace_content` and `trace_file`.

Source: `crates/troupekit-agent/src/persona.rs`, `crates/troupekit-graph/tests/workflow.rs`

## Read a trace

```text
usage: troupekit trace <file> [--html [OUT]] [--only CATS] [--no-color]
```

Source: `crates/troupekit-cli/src/args.rs`

`troupekit trace` prints the span tree in the terminal, or writes one HTML page with `--html`. The reader skips and counts a bad, oversized or too-deep line instead of failing. Defaults: lines up to 1 MiB, at most 1 000 000 records, nesting at most 32 levels deep.

Source: `crates/troupekit-trace/src/view/mod.rs`

## Spend limits

`Limits` holds the layered limits a run checks before every model call. Amounts are millionths of a US dollar; `None` is no limit.

| Field | Limits |
|---|---|
| `call` | one model call |
| `run` | the whole run |
| `scope` | a `PeriodLimit` for a scope's period, such as a goal's month |
| `user` | a `PeriodLimit` for the user's period |

A `PeriodLimit` has a `key`, the ledger total it reads, such as `goal:42`, and `max_usd_micros`. The ledger decides what the period is.

Source: `crates/troupekit-agent/src/budget.rs`

Period limits read a usage ledger. This test records spend for `user-1` and caps the graph below it, so the run pauses or fails:

```rust
let capped = troupekit_agent::Limits {
    user: Some(troupekit_agent::PeriodLimit {
        key: "user-1".into(),
        max_usd_micros: 10,
    }),
    ..Default::default()
};
let result = Graph::new("g")
    .agent("draft", drafter())
    .limits(capped)
    .ledger(ledger.clone())
    .run("hi")
    .await;
assert!(
    matches!(
        result,
        Err(KitError::Paused { .. } | KitError::Failed { .. })
    ),
    "{result:?}"
);
```

Source: `crates/troupekit-kit/tests/graph.rs`, test `graph_ledger_carries_period_spend_and_agent_ledgers_are_refused`

- An agent's default ledger is in memory and lost on restart. Pass one durable ledger, such as the SQLite store's, to agents that share a cap.
- A persona's `budget = { call_usd = 0.05, run_usd = 0.5 }` is combined with the code limits: each of `call` and `run` becomes the stricter of the two.
- A route policy's `[budget] cloud_usd_per_day` caps remote spend per day; at the cap, remote targets are skipped.
- A hosted graph run paused on its budget continues after `Runner::raise_budget(run_id, limits)`.

Source: `crates/troupekit-kit/src/agent.rs`, `crates/troupekit-agent/src/persona.rs`, `crates/troupekit-agent/src/budget.rs`, `crates/troupekit-providers/src/route/policy.rs`, `crates/troupekit-host/src/runner.rs`
