---
title: Kill and resume a workflow
description: Kill a worker process with SIGKILL in the middle of a workflow step, resume in a new process, and get the same result.
---

# Kill and resume a workflow

The `kill_resume` example starts a three-tool workflow in a SQLite file, kills the worker process with SIGKILL in the middle of the second tool, resumes it in a new process, and compares the result with an uninterrupted run. It exits 0 only when they match.

Source: `crates/troupekit/examples/kill_resume.rs`

## Run it

```sh
cargo run -p troupekit --example kill_resume
```

The example needs the `sqlite` feature, on by default, and no model or key: every node is a tool. It prints five numbered lines, ending with `5. same as an uninterrupted run: true`.

Source: `crates/troupekit/Cargo.toml` `[[example]]`, `crates/troupekit/examples/kill_resume.rs`

## The workflow

```toml
name = "three"
start = "first"
[state]
goal = "string"
a = "string"
b = "string"
c = "string"
[[nodes]]
id = "first"
kind = "tool"
tool = "first"
reads = ["goal"]
writes = "a"
[[nodes]]
id = "second"
kind = "tool"
tool = "second"
reads = ["a"]
writes = "b"
[[nodes]]
id = "third"
kind = "tool"
tool = "third"
reads = ["b"]
writes = "c"
[[nodes]]
id = "done"
kind = "end"
[[edges]]
from = "first"
to = "second"
[[edges]]
from = "second"
to = "third"
[[edges]]
from = "third"
to = "done"
```

Source: `crates/troupekit/examples/kill_resume.rs`, constant `WORKFLOW`

## Start and enqueue

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

## Work until done

A worker opens the same file, builds a runner with a 200 ms lease and ticks until the job is done:

```rust
async fn work(path: &Path, id: RunId, gate: bool) {
    let troupe = Troupe::builder().sqlite(path).build().await.expect("open");
    let runner = troupe.runner(io(&troupe, gate)).with_lease_ms(200);
    let store = SqliteStore::open(path).await.expect("open");
    let start = Instant::now();
    while store.job(id).await.expect("job") != Some(JobState::Done) {
        assert!(start.elapsed() < DEADLINE, "worker ran past its deadline");
        runner.tick(TickOptions::default()).await.expect("tick");
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}
```

Source: `crates/troupekit/examples/kill_resume.rs`

## What it proves

1. The parent creates the run and spawns a worker.
2. The worker's second tool writes a marker blob and never answers. The parent sees the marker and kills the worker.
3. The killed worker's lease expires. A second worker claims the job, steps from the latest checkpoint and finishes.
4. The parent runs the same workflow uninterrupted in another file and compares state, steps and usage.

Each tool call gets an `idempotency_key` that stays the same when a step replays, so a tool with side effects can dedupe. See [Tools](../concepts/tools.md).
