use serde_json::json;
use std::collections::HashSet;
use std::path::Path;
use troupekit_core::error::CoreError;
use troupekit_core::ids::RunId;
use troupekit_core::message::{Message, Role};
use troupekit_core::store::{BlobStore, Checkpoint, CheckpointStore, JobQueue, MemoryLog};
use troupekit_store_sqlite::SqliteStore;

const HANDLES: usize = 4;
const ROUNDS: usize = 100;

async fn handles(path: &Path) -> Vec<SqliteStore> {
    let mut out = Vec::with_capacity(HANDLES);
    for _ in 0..HANDLES {
        out.push(SqliteStore::open(path).await.unwrap());
    }
    out
}

async fn race<T, F, Fut>(stores: &[SqliteStore], f: F) -> Vec<T>
where
    T: Send + 'static,
    F: Fn(SqliteStore, usize) -> Fut,
    Fut: std::future::Future<Output = T> + Send + 'static,
{
    let barrier = std::sync::Arc::new(tokio::sync::Barrier::new(stores.len()));
    let tasks: Vec<_> = stores
        .iter()
        .enumerate()
        .map(|(i, s)| {
            let fut = f(s.clone(), i);
            let barrier = barrier.clone();
            tokio::spawn(async move {
                barrier.wait().await;
                fut.await
            })
        })
        .collect();
    let mut out = Vec::with_capacity(tasks.len());
    for task in tasks {
        out.push(task.await.unwrap());
    }
    out
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn second_handle_sees_committed_writes() {
    let dir = tempfile::tempdir().unwrap();
    let stores = handles(&dir.path().join("shared.db")).await;
    let run = RunId::new();
    stores[0]
        .save(Checkpoint {
            run_id: run,
            step: 0,
            state: json!({"x": 1}),
        })
        .await
        .unwrap();
    stores[0].put("p", vec![1, 2]).await.unwrap();
    assert_eq!(
        stores[1].latest(run).await.unwrap().unwrap().state,
        json!({"x": 1})
    );
    assert_eq!(stores[1].get("p").await.unwrap(), vec![1, 2]);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn two_handles_cannot_save_the_same_step() {
    let dir = tempfile::tempdir().unwrap();
    let stores = handles(&dir.path().join("shared.db")).await;
    for _ in 0..ROUNDS {
        let run = RunId::new();
        let results = race(&stores, move |s, i| async move {
            s.save(Checkpoint {
                run_id: run,
                step: 0,
                state: json!(i),
            })
            .await
        })
        .await;
        assert_eq!(
            results.iter().filter(|r| r.is_ok()).count(),
            1,
            "{results:?}"
        );
        for err in results.into_iter().filter_map(Result::err) {
            assert_eq!(
                err,
                CoreError::Conflict {
                    expected: 1,
                    actual: 0
                }
            );
        }
        assert_eq!(stores[0].history(run).await.unwrap().len(), 1);
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn two_handles_never_claim_the_same_job() {
    let dir = tempfile::tempdir().unwrap();
    let stores = handles(&dir.path().join("shared.db")).await;
    let runs: Vec<RunId> = (0..ROUNDS * 2).map(|_| RunId::new()).collect();
    for run in &runs {
        stores[0].enqueue(*run, 0).await.unwrap();
    }
    let mut claimed = Vec::new();
    for _ in 0..ROUNDS {
        for leases in race(&stores, |s, _| async move { s.claim_due(1, 60_000).await }).await {
            claimed.extend(leases.unwrap());
        }
    }
    let unique: HashSet<RunId> = claimed.iter().map(|l| l.run_id).collect();
    assert_eq!(unique.len(), claimed.len());
    assert_eq!(unique, runs.into_iter().collect());
    let tokens: HashSet<u64> = claimed.iter().map(|l| l.token).collect();
    assert_eq!(tokens.len(), claimed.len());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn two_handles_cannot_append_at_the_same_length() {
    let dir = tempfile::tempdir().unwrap();
    let stores = handles(&dir.path().join("shared.db")).await;
    for round in 0..ROUNDS {
        let log = format!("log-{round}");
        let results = race(&stores, |s, i| {
            let log = log.clone();
            async move {
                s.append(&log, 0, vec![Message::text(Role::User, i.to_string())])
                    .await
            }
        })
        .await;
        assert_eq!(
            results.iter().filter(|r| r.is_ok()).count(),
            1,
            "{results:?}"
        );
        for err in results.into_iter().filter_map(Result::err) {
            assert_eq!(
                err,
                CoreError::Conflict {
                    expected: 0,
                    actual: 1
                }
            );
        }
        assert_eq!(stores[1].read(&log).await.unwrap().len(), 1);
    }
}
