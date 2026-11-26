use crate::common;

use common::{planner_model, saving_tools, stack, APPROVAL, LINEAR};
use serde_json::json;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::watch;
use troupekit_agent::EventSink;
use troupekit_core::event::{Event, EventBody, PauseReason};
use troupekit_core::ids::RunId;
use troupekit_graph::{GraphError, GraphOutcome, GraphStatus};
use troupekit_host::{EventLog, HostError, JobState, TickOptions, WorkOptions};

#[tokio::test]
async fn tick_runs_a_workflow_to_done() {
    let (tools, keys) = saving_tools(false);
    let s = stack(planner_model(), tools);
    let id = s.start(LINEAR, "run 5k").await;
    assert_eq!(s.saved(id).await.steps(), 0);
    let report = s.runner().tick(TickOptions::default()).await.unwrap();
    assert_eq!(report.processed, 3);
    assert_eq!((report.lost, report.errors.len()), (0, 0));
    assert_eq!(s.queue.job(id), Some(JobState::Done));
    let run = s.saved(id).await;
    assert_eq!(run.status(), &GraphStatus::Done);
    assert_eq!(run.steps(), 3);
    assert_eq!(run.state()["note"], json!("saved"));
    assert_eq!(keys.lock().unwrap().clone(), vec![format!("{id}:3:0")]);
    assert_eq!(
        s.runner()
            .tick(TickOptions::default())
            .await
            .unwrap()
            .processed,
        0
    );
}

#[tokio::test]
async fn tick_respects_max_jobs_and_budget() {
    let (tools, _) = saving_tools(false);
    let s = stack(planner_model(), tools);
    let ids = [
        s.start(LINEAR, "a").await,
        s.start(LINEAR, "b").await,
        s.start(LINEAR, "c").await,
    ];
    let runner = s.runner();
    let none = TickOptions {
        budget_ms: 0,
        ..TickOptions::default()
    };
    assert_eq!(runner.tick(none).await.unwrap().processed, 0);
    let two = TickOptions {
        max_jobs: 2,
        ..TickOptions::default()
    };
    assert_eq!(runner.tick(two).await.unwrap().processed, 2);
    let stepped = {
        let mut n = 0;
        for id in ids {
            n += s.saved(id).await.steps();
        }
        n
    };
    assert_eq!(stepped, 2);
    let all = runner.tick(TickOptions::default()).await.unwrap();
    assert_eq!(all.processed, 7);
    for id in ids {
        assert_eq!(s.queue.job(id), Some(JobState::Done));
    }
}

#[tokio::test]
async fn human_pause_completes_job_and_resume_requeues() {
    let (tools, _) = saving_tools(false);
    let s = stack(planner_model(), tools);
    let id = s.start(APPROVAL, "g").await;
    let runner = s.runner();
    assert_eq!(
        runner.tick(TickOptions::default()).await.unwrap().processed,
        2
    );
    assert_eq!(s.queue.job(id), Some(JobState::Done));
    let run = s.saved(id).await;
    assert_eq!(
        run.status(),
        &GraphStatus::WaitingHuman {
            reason: PauseReason::Approval {
                node: "approve".into()
            }
        }
    );
    assert_eq!(run.pending_human()[0].node, "approve");
    assert_eq!(
        runner.tick(TickOptions::default()).await.unwrap().processed,
        0
    );
    let wrong = runner.resume(id, "approve", json!("yes")).await;
    assert!(matches!(
        wrong,
        Err(HostError::Graph(GraphError::Resume(_)))
    ));
    runner.resume(id, "approve", json!(true)).await.unwrap();
    assert!(matches!(s.queue.job(id), Some(JobState::Pending { .. })));
    let again = runner.resume(id, "approve", json!(true)).await;
    assert!(matches!(
        again,
        Err(HostError::Graph(GraphError::Resume(_)))
    ));
    assert_eq!(
        runner.tick(TickOptions::default()).await.unwrap().processed,
        1
    );
    assert_eq!(s.saved(id).await.status(), &GraphStatus::Done);
    assert_eq!(s.queue.job(id), Some(JobState::Done));
    let missing = runner.resume(RunId::new(), "approve", json!(true)).await;
    assert!(matches!(missing, Err(HostError::NoCheckpoint(_))));
}

#[tokio::test]
async fn retry_reschedules_with_not_before() {
    let (tools, _) = saving_tools(false);
    let s = stack(planner_model(), tools);
    let id = s.start(LINEAR, "blocked").await;
    let runner = s.runner();
    assert_eq!(
        runner.tick(TickOptions::default()).await.unwrap().processed,
        1
    );
    assert_eq!(
        s.queue.job(id),
        Some(JobState::Pending { due_ms: 1_001_000 })
    );
    assert_eq!(
        runner.tick(TickOptions::default()).await.unwrap().processed,
        0
    );
    s.clock.advance(1_000);
    assert_eq!(
        runner.tick(TickOptions::default()).await.unwrap().processed,
        1
    );
    assert_eq!(
        s.queue.job(id),
        Some(JobState::Pending { due_ms: 1_003_000 })
    );
    assert_eq!(s.saved(id).await.steps(), 2);
}

#[tokio::test]
async fn failed_run_fails_the_job() {
    let (tools, _) = saving_tools(true);
    let s = stack(planner_model(), tools);
    let id = s.start(LINEAR, "g").await;
    let report = s.runner().tick(TickOptions::default()).await.unwrap();
    assert_eq!(report.processed, 3);
    let Some(JobState::Failed { error }) = s.queue.job(id) else {
        panic!("the job fails");
    };
    assert!(error.contains("boom"), "{error}");
    assert!(matches!(
        s.saved(id).await.status(),
        GraphStatus::Failed { .. }
    ));
    let out = s.saved(id).await.step(s.io.as_ref()).await;
    assert!(matches!(out, GraphOutcome::Failed(_)));
}

#[tokio::test]
async fn events_dedupe_on_run_and_seq() {
    let log = EventLog::default();
    let (a, b) = (RunId::new(), RunId::new());
    let event = |run_id, seq, text: &str| Event {
        run_id,
        seq,
        at_ms: 0,
        body: EventBody::TextDelta { text: text.into() },
    };
    log.emit(event(a, 1, "one"));
    log.emit(event(a, 0, "zero"));
    log.emit(event(b, 0, "other"));
    log.emit(event(a, 1, "one again"));
    let got = log.events(a);
    assert_eq!(got.iter().map(|e| e.seq).collect::<Vec<_>>(), vec![0, 1]);
    assert_eq!(got[1], event(a, 1, "one again"));
    assert_eq!(log.events(b).len(), 1);
    let (tools, _) = saving_tools(false);
    let s = stack(planner_model(), tools);
    let id = s.start(LINEAR, "g").await;
    s.runner().tick(TickOptions::default()).await.unwrap();
    let events = s.events.events(id);
    assert_eq!(
        events.iter().map(|e| e.seq).collect::<Vec<_>>(),
        (0..events.len() as u64).collect::<Vec<_>>()
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn worker_mode_stops_on_shutdown() {
    let (tools, _) = saving_tools(false);
    let s = stack(planner_model(), tools);
    let id = s.start(LINEAR, "g").await;
    let runner = Arc::new(s.runner());
    let (tx, rx) = watch::channel(false);
    let worker = {
        let runner = runner.clone();
        let opts = WorkOptions {
            idle_ms: 10,
            ..WorkOptions::default()
        };
        tokio::spawn(async move { runner.work(opts, rx).await })
    };
    tokio::time::timeout(Duration::from_secs(5), async {
        while s.queue.job(id) != Some(JobState::Done) {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .expect("the worker finishes the run");
    tx.send(true).unwrap();
    tokio::time::timeout(Duration::from_secs(2), worker)
        .await
        .expect("the worker stops on shutdown")
        .unwrap();
    assert_eq!(s.saved(id).await.status(), &GraphStatus::Done);
}

#[tokio::test]
async fn runner_process_emits_job_span() {
    let (tools, _) = saving_tools(false);
    let s = stack(planner_model(), tools);
    let id = s.start(LINEAR, "run 5k").await;
    let sink = Arc::new(troupekit_trace::MemorySink::default());
    let tracer = troupekit_trace::Tracer::builder()
        .sink(sink.clone())
        .build()
        .unwrap();
    let report = tracer
        .in_scope(s.runner().tick(TickOptions::default()))
        .await
        .unwrap();
    drop(tracer);
    assert_eq!(report.processed, 3);
    let records: Vec<serde_json::Value> = sink
        .lines()
        .iter()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    let starts = |name: &str| -> Vec<&serde_json::Value> {
        records
            .iter()
            .filter(|r| r["t"] == "start" && r["name"] == name)
            .collect()
    };
    let jobs = starts("graph.job");
    assert_eq!(jobs.len(), 3);
    assert!(jobs
        .iter()
        .all(|r| r["cat"] == "graphs" && r["fields"]["run"] == id.to_string().as_str()));
    let job_ids: Vec<&serde_json::Value> = jobs.iter().map(|r| &r["span"]).collect();
    let steps = starts("graph.step");
    assert_eq!(steps.len(), 3);
    assert!(steps.iter().all(|r| job_ids.contains(&&r["parent"])));
    let ends: Vec<&serde_json::Value> = records
        .iter()
        .filter(|r| r["t"] == "end" && job_ids.contains(&&r["span"]))
        .collect();
    assert!(ends.iter().all(|r| r["fields"]["outcome"] == "stepped"));
}
