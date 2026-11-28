use crate::common;

use common::{planner_model, saving_tools, stack_on, Stack, LINEAR};
use serde_json::json;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::Notify;
use troupekit_core::error::CoreError;
use troupekit_core::ids::RunId;
use troupekit_core::inmem::InMemCheckpoints;
use troupekit_core::store::{BranchWrite, Checkpoint, CheckpointStore, ForkLink};
use troupekit_graph::testing::FnTool;
use troupekit_agent::Limits;
use troupekit_graph::GraphStatus;
use troupekit_host::{HostError, JobQueue, JobState, Processed, Runner, TickOptions};
use troupekit_tools::{ToolOutput, ToolRegistry};

const FORK: &str = r#"
name = "fork"
start = "seed"
[state]
goal = "string"
a = "string"
b = "string"
[[nodes]]
id = "seed"
kind = "tool"
tool = "seed"
[[nodes]]
id = "left"
kind = "tool"
tool = "fast"
writes = "a"
[[nodes]]
id = "right"
kind = "tool"
tool = "slow"
writes = "b"
[[nodes]]
id = "join"
kind = "join"
[[nodes]]
id = "done"
kind = "end"
[[edges]]
from = "seed"
to = "left"
[[edges]]
from = "seed"
to = "right"
[[edges]]
from = "left"
to = "join"
[[edges]]
from = "right"
to = "join"
[[edges]]
from = "join"
to = "done"
"#;

const ONE: TickOptions = TickOptions {
    max_jobs: 1,
    budget_ms: 250_000,
};

const WAIT: Duration = Duration::from_secs(5);

type Keys = Arc<Mutex<Vec<String>>>;

/// Registers `seed`, `fast` (answers `A` at once) and `slow` (answers `B`,
/// first waiting on the notify `gate` returns, if any). `fast` and `slow`
/// log each call's idempotency key. Returns the registry and both logs.
fn fork_tools(gate: impl Fn() -> Option<Arc<Notify>> + Send + Sync + 'static) -> (ToolRegistry, Keys, Keys) {
    let (fast, slow) = (Keys::default(), Keys::default());
    let gate = Arc::new(gate);
    let mut tools = ToolRegistry::new();
    tools.register(FnTool::new("seed", |_, _| async { Ok(ToolOutput::text("go")) })).unwrap();
    let log = fast.clone();
    tools
        .register(FnTool::new("fast", move |_, ctx| {
            let log = log.clone();
            async move {
                log.lock().unwrap().push(ctx.idempotency_key.unwrap());
                Ok(ToolOutput::text("A"))
            }
        }))
        .unwrap();
    let log = slow.clone();
    tools
        .register(FnTool::new("slow", move |_, ctx| {
            let (log, gate) = (log.clone(), gate.clone());
            async move {
                log.lock().unwrap().push(ctx.idempotency_key.unwrap());
                if let Some(wait) = gate() {
                    wait.notified().await;
                }
                Ok(ToolOutput::text("B"))
            }
        }))
        .unwrap();
    (tools, fast, slow)
}

/// Returns a gate that holds the first `slow` call on `notify` and lets
/// every later call through.
fn first_call_waits_on(notify: Arc<Notify>) -> impl Fn() -> Option<Arc<Notify>> + Send + Sync + 'static {
    let first = AtomicBool::new(true);
    move || first.swap(false, Ordering::SeqCst).then(|| notify.clone())
}

/// Waits until the step in flight of `id` holds a branch write from node
/// `node`, failing after `WAIT`. Returns the branch writes of that step.
async fn wait_for_write(s: &Stack, id: RunId, node: &str) -> Vec<BranchWrite> {
    tokio::time::timeout(WAIT, async {
        loop {
            let step = s.checkpoints.latest(id).await.unwrap().unwrap().step + 1;
            let writes = s.checkpoints.branch_writes(id, step).await.unwrap();
            if writes.iter().any(|w| w.node() == node) {
                return writes;
            }
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    })
    .await
    .expect("the finished branch is written while its sibling runs")
}

/// Asserts that both branches of `FORK` reached the joined `state`.
fn assert_joined(state: &serde_json::Map<String, serde_json::Value>) {
    assert_eq!(state["a"], json!("A"));
    assert_eq!(state["b"], json!("B"));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn process_writes_each_finished_branch() {
    let release = Arc::new(Notify::new());
    let (tools, fast, slow) = fork_tools(first_call_waits_on(release.clone()));
    let s = stack_on(planner_model(), tools, InMemCheckpoints::with_history());
    let id = s.start(FORK, "g").await;
    let runner = Arc::new(s.runner());
    let worker = {
        let runner = runner.clone();
        tokio::spawn(async move { runner.tick(TickOptions::default()).await })
    };
    let writes = wait_for_write(&s, id, "left").await;
    assert_eq!(writes.len(), 1, "{writes:?}");
    assert_eq!(writes[0].branch(), 0);
    assert_eq!(slow.lock().unwrap().len(), 1);
    let held = s.checkpoints.latest(id).await.unwrap().unwrap().step;
    release.notify_one();
    let report = tokio::time::timeout(WAIT, worker).await.unwrap().unwrap().unwrap();
    assert!(report.errors.is_empty(), "{:?}", report.errors);
    assert_eq!(s.queue.job(id), Some(JobState::Done));
    let done = s.saved(id).await;
    assert_eq!(done.status(), &GraphStatus::Done);
    assert_joined(done.state());
    assert!(s.checkpoints.branch_writes(id, held + 1).await.unwrap().is_empty());
    assert_eq!((fast.lock().unwrap().len(), slow.lock().unwrap().len()), (1, 1));
}

#[tokio::test]
async fn process_after_crash_applies_pending_writes() {
    let (tools, fast, slow) = fork_tools(first_call_waits_on(Arc::new(Notify::new())));
    let s = stack_on(planner_model(), tools, InMemCheckpoints::with_history());
    let id = s.start(FORK, "g").await;
    let crashed = s.runner().with_lease_ms(1_000);
    let hung = tokio::time::timeout(Duration::from_millis(200), crashed.tick(TickOptions::default())).await;
    assert!(hung.is_err(), "the fan-out step hangs in the slow tool");
    drop(crashed);
    let writes = wait_for_write(&s, id, "left").await;
    assert_eq!(writes.len(), 1, "{writes:?}");
    let fresh = s.runner();
    assert_eq!(fresh.tick(TickOptions::default()).await.unwrap().processed, 0);
    s.clock.advance(1_000);
    let report = fresh.tick(TickOptions::default()).await.unwrap();
    assert!(report.errors.is_empty(), "{:?}", report.errors);
    assert_eq!(s.queue.job(id), Some(JobState::Done));
    let done = s.saved(id).await;
    assert_eq!(done.status(), &GraphStatus::Done);
    assert_joined(done.state());
    assert_eq!(fast.lock().unwrap().len(), 1, "the finished branch never runs again");
    let slow = slow.lock().unwrap().clone();
    assert_eq!(slow.len(), 2);
    assert_eq!(slow[0], slow[1], "the re-run keeps its idempotency key");
}

#[tokio::test]
async fn budget_raise_after_crash_keeps_the_finished_branch() {
    let (tools, fast, slow) = fork_tools(first_call_waits_on(Arc::new(Notify::new())));
    let s = stack_on(planner_model(), tools, InMemCheckpoints::with_history());
    let id = s.start(FORK, "g").await;
    let crashed = s.runner().with_lease_ms(1_000);
    let hung = tokio::time::timeout(Duration::from_millis(200), crashed.tick(TickOptions::default())).await;
    assert!(hung.is_err(), "the fan-out step hangs in the slow tool");
    drop(crashed);
    let writes = wait_for_write(&s, id, "left").await;
    assert_eq!(writes.len(), 1, "{writes:?}");
    let held = s.checkpoints.latest(id).await.unwrap().unwrap().step;
    let fresh = s.runner();
    fresh.raise_budget(id, Limits::default()).await.unwrap();
    assert_eq!(s.checkpoints.latest(id).await.unwrap().unwrap().step, held + 1);
    assert_eq!(s.checkpoints.branch_writes(id, held + 1).await.unwrap(), writes, "the bump keeps the carried write");
    s.clock.advance(1_000);
    let report = fresh.tick(TickOptions::default()).await.unwrap();
    assert!(report.errors.is_empty(), "{:?}", report.errors);
    assert_eq!(s.queue.job(id), Some(JobState::Done));
    let done = s.saved(id).await;
    assert_eq!(done.status(), &GraphStatus::Done);
    assert_joined(done.state());
    assert_eq!(fast.lock().unwrap().len(), 1, "the finished branch never runs again");
    assert_eq!(slow.lock().unwrap().len(), 2);
    for step in 0..=done.steps() + 1 {
        assert!(s.checkpoints.branch_writes(id, step).await.unwrap().is_empty(), "step {step}");
    }
}

#[tokio::test]
async fn lost_lease_leaves_one_write_per_branch_and_loser_is_lost() {
    let release = Arc::new(Notify::new());
    let (tools, fast, slow) = fork_tools(first_call_waits_on(release.clone()));
    let s = stack_on(planner_model(), tools, InMemCheckpoints::with_history());
    let id = s.start(FORK, "g").await;
    let stale = s.runner().with_lease_ms(1_000);
    let winner = s.runner().with_lease_ms(1_000);
    assert_eq!(stale.tick(ONE).await.unwrap().processed, 1);
    let lease = s.queue.claim_due(1, 1_000).await.unwrap().remove(0);
    let (lost, ()) = tokio::join!(stale.process(&lease), async {
        let writes = wait_for_write(&s, id, "left").await;
        assert_eq!(writes.len(), 1, "{writes:?}");
        s.clock.advance(1_000);
        let report = winner.tick(TickOptions::default()).await.unwrap();
        assert!(report.errors.is_empty(), "{:?}", report.errors);
        assert_eq!(s.queue.job(id), Some(JobState::Done));
        release.notify_one();
    });
    assert_eq!(lost.unwrap(), Processed::Lost);
    assert_eq!(s.queue.job(id), Some(JobState::Done));
    let done = s.saved(id).await;
    assert_eq!(done.status(), &GraphStatus::Done);
    assert_joined(done.state());
    assert_eq!(fast.lock().unwrap().len(), 1, "the winner applied the stale worker's write");
    let slow = slow.lock().unwrap().clone();
    assert_eq!(slow.len(), 2);
    assert_eq!(slow[0], slow[1], "both attempts are the same call of the same step");
    for step in 0..=done.steps() + 1 {
        assert!(s.checkpoints.branch_writes(id, step).await.unwrap().is_empty());
    }
}

#[tokio::test]
async fn fork_enqueues_a_new_run_from_any_step() {
    let (tools, keys) = saving_tools(false);
    let s = stack_on(planner_model(), tools, InMemCheckpoints::with_history());
    let id = s.start(LINEAR, "run 5k").await;
    let runner = s.runner();
    runner.tick(TickOptions::default()).await.unwrap();
    assert_eq!(s.queue.job(id), Some(JobState::Done));
    let before = s.checkpoints.history(id).await.unwrap();
    assert!(before.len() > 2, "{}", before.len());
    let child = runner.fork(id, 1).await.unwrap();
    assert_ne!(child, id);
    assert_eq!(s.checkpoints.latest(child).await.unwrap().unwrap().step, 0);
    assert_eq!(s.checkpoints.forks(id).await.unwrap(), vec![ForkLink::new(child, id, 1)]);
    assert!(matches!(s.queue.job(child), Some(JobState::Pending { .. })));
    runner.tick(TickOptions::default()).await.unwrap();
    assert_eq!(s.queue.job(child), Some(JobState::Done));
    let forked = s.saved(child).await;
    assert_eq!(forked.status(), &GraphStatus::Done);
    assert_eq!(forked.parent(), Some((id, 1)));
    assert_eq!(s.checkpoints.history(id).await.unwrap(), before, "the original never changes");
    let keys = keys.lock().unwrap().clone();
    assert_eq!(keys.len(), 2);
    assert_ne!(keys[0], keys[1], "the fork's tool call has its own idempotency key");
    let missing = runner.fork(id, 99).await.unwrap_err();
    assert_eq!(missing, HostError::NoCheckpoint(id));
    let unknown = RunId::new();
    assert_eq!(runner.fork(unknown, 0).await.unwrap_err(), HostError::NoCheckpoint(unknown));
}

/// Wraps an in-memory store and fails the `save_branch` of nodes `left`
/// and `right` with a store error, like a full disk. Every other call,
/// the engine's own seq reserve included, goes to the wrapped store.
/// Used to prove a failed branch write reschedules without a save.
struct FailingBranches(Arc<InMemCheckpoints>);

#[async_trait::async_trait]
impl CheckpointStore for FailingBranches {
    async fn save(&self, cp: Checkpoint) -> Result<(), CoreError> {
        self.0.save(cp).await
    }
    async fn latest(&self, run_id: RunId) -> Result<Option<Checkpoint>, CoreError> {
        self.0.latest(run_id).await
    }
    async fn history(&self, run_id: RunId) -> Result<Vec<Checkpoint>, CoreError> {
        self.0.history(run_id).await
    }
    async fn save_branch(&self, run_id: RunId, step: u64, write: BranchWrite) -> Result<bool, CoreError> {
        if matches!(write.node(), "left" | "right") {
            return Err(CoreError::Store("disk full".into()));
        }
        self.0.save_branch(run_id, step, write).await
    }
    async fn branch_writes(&self, run_id: RunId, step: u64) -> Result<Vec<BranchWrite>, CoreError> {
        self.0.branch_writes(run_id, step).await
    }
}

#[tokio::test]
async fn branch_write_error_reschedules_without_saving() {
    let (tools, fast, slow) = fork_tools(|| None);
    let s = stack_on(planner_model(), tools, InMemCheckpoints::with_history());
    let id = s.start(FORK, "g").await;
    let failing = Runner::new(s.queue.clone(), Arc::new(FailingBranches(s.checkpoints.clone())), s.io.clone());
    let report = failing.tick(TickOptions::default()).await.unwrap();
    assert_eq!(report.errors.len(), 1, "{:?}", report.errors);
    assert!(report.errors[0].contains("disk full"), "{:?}", report.errors);
    assert_eq!(s.queue.job(id), Some(JobState::Pending { due_ms: 1_001_000 }));
    let saved = s.saved(id).await;
    assert_ne!(saved.state().get("a"), Some(&json!("A")), "no checkpoint holds the failed step");
    assert_ne!(saved.state().get("b"), Some(&json!("B")));
    let ran = (fast.lock().unwrap().len(), slow.lock().unwrap().len());
    assert_eq!(ran.0, 1, "the fan-out step ran and its first branch write failed");
    assert!(ran.1 <= 1, "{ran:?}");
    s.clock.advance(1_000);
    let report = s.runner().tick(TickOptions::default()).await.unwrap();
    assert!(report.errors.is_empty(), "{:?}", report.errors);
    let done = s.saved(id).await;
    assert_eq!(done.status(), &GraphStatus::Done);
    assert_joined(done.state());
    assert_eq!(fast.lock().unwrap().len(), ran.0 + 1, "a branch whose write failed runs again");
}
