use crate::common;

use common::{planner_model, saving_tools, stack, Stack, LINEAR};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::Notify;
use troupekit_core::ids::RunId;
use troupekit_core::store::CheckpointStore;
use troupekit_graph::testing::FnTool;
use troupekit_graph::{GraphRun, GraphStatus};
use troupekit_host::{JobQueue, JobState, Processed, TickOptions};
use troupekit_tools::{ToolOutput, ToolRegistry};

async fn uninterrupted() -> (GraphRun, usize) {
    let (tools, _) = saving_tools(false);
    let s = stack(planner_model(), tools);
    let id = s.start(LINEAR, "run 5k").await;
    s.runner().tick(TickOptions::default()).await.unwrap();
    (s.saved(id).await, s.events.events(id).len())
}

fn gated_tool(
    keys: Arc<Mutex<Vec<String>>>,
    gate: impl Fn() -> Option<Arc<Notify>> + Send + Sync + 'static,
) -> ToolRegistry {
    let gate = Arc::new(gate);
    let mut tools = ToolRegistry::new();
    tools
        .register(FnTool::new("save_plan", move |_, ctx| {
            let (keys, gate) = (keys.clone(), gate.clone());
            async move {
                keys.lock().unwrap().push(ctx.idempotency_key.unwrap());
                if let Some(wait) = gate() {
                    wait.notified().await;
                }
                Ok(ToolOutput::text("saved"))
            }
        }))
        .unwrap();
    tools
}

fn contiguous(s: &Stack, id: RunId) -> usize {
    let events = s.events.events(id);
    assert_eq!(
        events.iter().map(|e| e.seq).collect::<Vec<_>>(),
        (0..events.len() as u64).collect::<Vec<_>>()
    );
    events.len()
}

#[tokio::test]
async fn crash_mid_step_resumes_from_last_checkpoint() {
    let keys = Arc::new(Mutex::new(Vec::new()));
    let first = Arc::new(AtomicBool::new(true));
    let never = Arc::new(Notify::new());
    let tools = gated_tool(keys.clone(), move || {
        first.swap(false, Ordering::SeqCst).then(|| never.clone())
    });
    let s = stack(planner_model(), tools);
    let id = s.start(LINEAR, "run 5k").await;
    let crashed = s.runner().with_lease_ms(1_000);
    let two = TickOptions {
        max_jobs: 2,
        ..TickOptions::default()
    };
    assert_eq!(crashed.tick(two).await.unwrap().processed, 2);
    let hung = tokio::time::timeout(
        Duration::from_millis(100),
        crashed.tick(TickOptions::default()),
    )
    .await;
    assert!(hung.is_err(), "the third step hangs in the tool");
    drop(crashed);
    assert_eq!(s.saved(id).await.steps(), 2);
    assert!(matches!(s.queue.job(id), Some(JobState::Leased { .. })));
    let fresh = s.runner();
    assert_eq!(
        fresh.tick(TickOptions::default()).await.unwrap().processed,
        0
    );
    s.clock.advance(1_000);
    let report = fresh.tick(TickOptions::default()).await.unwrap();
    assert_eq!(report.processed, 1);
    assert_eq!(s.queue.job(id), Some(JobState::Done));
    let done = s.saved(id).await;
    let (reference, reference_events) = uninterrupted().await;
    assert_eq!(done.status(), &GraphStatus::Done);
    assert_eq!(done.state(), reference.state());
    assert_eq!(done.steps(), reference.steps());
    assert_eq!(done.usage(), reference.usage());
    let keys = keys.lock().unwrap().clone();
    assert_eq!(keys, vec![format!("{id}:3:0"), format!("{id}:3:0")]);
    assert_eq!(contiguous(&s, id), reference_events);
}

#[tokio::test]
async fn crash_between_steps_resumes() {
    let (tools, keys) = saving_tools(false);
    let s = stack(planner_model(), tools);
    let id = s.start(LINEAR, "run 5k").await;
    let crashed = s.queue.claim_due(1, 1_000).await.unwrap();
    assert_eq!(crashed.len(), 1);
    let cp = s.checkpoints.latest(id).await.unwrap().unwrap();
    let mut run = GraphRun::from_checkpoint(&cp).unwrap();
    run.step(s.io.as_ref()).await;
    s.checkpoints.save(run.checkpoint()).await.unwrap();
    let runner = s.runner();
    assert_eq!(
        runner.tick(TickOptions::default()).await.unwrap().processed,
        0
    );
    s.clock.advance(1_000);
    assert_eq!(
        runner.tick(TickOptions::default()).await.unwrap().processed,
        2
    );
    assert_eq!(s.queue.job(id), Some(JobState::Done));
    let (reference, reference_events) = uninterrupted().await;
    assert_eq!(s.saved(id).await.state(), reference.state());
    assert_eq!(s.model.calls(), 1);
    assert_eq!(keys.lock().unwrap().len(), 1);
    assert_eq!(contiguous(&s, id), reference_events);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn two_runners_never_advance_a_run_twice() {
    let (tools, keys) = saving_tools(false);
    let s = stack(planner_model(), tools);
    let mut ids = Vec::new();
    for i in 0..8 {
        ids.push(s.start(LINEAR, &format!("goal {i}")).await);
    }
    let (a, b) = (Arc::new(s.runner()), Arc::new(s.runner()));
    let opts = TickOptions {
        max_jobs: 3,
        ..TickOptions::default()
    };
    for _ in 0..50 {
        if ids
            .iter()
            .all(|id| s.queue.job(*id) == Some(JobState::Done))
        {
            break;
        }
        let (ra, rb) = (a.clone(), b.clone());
        let (x, y) = tokio::join!(
            tokio::spawn(async move { ra.tick(opts).await }),
            tokio::spawn(async move { rb.tick(opts).await })
        );
        for report in [x.unwrap().unwrap(), y.unwrap().unwrap()] {
            assert!(report.errors.is_empty(), "{:?}", report.errors);
            assert_eq!(report.lost, 0);
        }
    }
    for id in &ids {
        assert_eq!(s.queue.job(*id), Some(JobState::Done));
        assert_eq!(s.saved(*id).await.status(), &GraphStatus::Done);
    }
    assert_eq!(s.model.calls(), ids.len());
    assert_eq!(keys.lock().unwrap().len(), ids.len());
}

#[tokio::test]
async fn stale_runner_loses_the_save() {
    let keys = Arc::new(Mutex::new(Vec::new()));
    let first = Arc::new(AtomicBool::new(true));
    let release = Arc::new(Notify::new());
    let gate = release.clone();
    let tools = gated_tool(keys.clone(), move || {
        first.swap(false, Ordering::SeqCst).then(|| gate.clone())
    });
    let s = stack(planner_model(), tools);
    let id = s.start(LINEAR, "run 5k").await;
    let stale = s.runner().with_lease_ms(1_000);
    let winner = s.runner().with_lease_ms(1_000);
    let two = TickOptions {
        max_jobs: 2,
        ..TickOptions::default()
    };
    assert_eq!(stale.tick(two).await.unwrap().processed, 2);
    let lease = s.queue.claim_due(1, 1_000).await.unwrap().remove(0);
    let (lost, ()) = tokio::join!(stale.process(&lease), async {
        while keys.lock().unwrap().is_empty() {
            tokio::task::yield_now().await;
        }
        s.clock.advance(1_000);
        let report = winner.tick(TickOptions::default()).await.unwrap();
        assert_eq!(report.processed, 1);
        assert_eq!(s.queue.job(id), Some(JobState::Done));
        release.notify_one();
    });
    assert_eq!(lost.unwrap(), Processed::Lost);
    assert_eq!(s.queue.job(id), Some(JobState::Done));
    let done = s.saved(id).await;
    assert_eq!(done.status(), &GraphStatus::Done);
    assert_eq!(done.steps(), 3);
    assert_eq!(keys.lock().unwrap().len(), 2);
}
