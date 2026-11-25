use std::path::PathBuf;
use std::sync::Arc;
use troupekit_core::clock::FixedClock;
use troupekit_core::error::CoreError;
use troupekit_core::ids::RunId;
use troupekit_core::store::{JobQueue, JobState};
use troupekit_store_sqlite::SqliteStore;

struct Temp {
    _dir: tempfile::TempDir,
    path: PathBuf,
    clock: Arc<FixedClock>,
}

impl Temp {
    fn new() -> Self {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("queue.db");
        Self {
            _dir: dir,
            path,
            clock: Arc::new(FixedClock::new(1_000)),
        }
    }

    async fn open(&self) -> SqliteStore {
        SqliteStore::builder(&self.path)
            .clock(self.clock.clone())
            .open()
            .await
            .unwrap()
    }
}

async fn job(q: &SqliteStore, run: RunId) -> Option<JobState> {
    q.job(run).await.unwrap()
}

#[tokio::test]
async fn one_job_per_run() {
    let t = Temp::new();
    let q = t.open().await;
    let run = RunId::new();
    q.enqueue(run, 900).await.unwrap();
    q.enqueue(run, 950).await.unwrap();
    assert_eq!(job(&q, run).await, Some(JobState::Pending { due_ms: 900 }));
    q.enqueue(run, 100).await.unwrap();
    assert_eq!(job(&q, run).await, Some(JobState::Pending { due_ms: 100 }));
    let leases = q.claim_due(10, 500).await.unwrap();
    assert_eq!(leases.len(), 1);
    assert!(q.claim_due(10, 500).await.unwrap().is_empty());
    assert_eq!(job(&q, RunId::new()).await, None);
}

#[tokio::test]
async fn claims_due_in_order_with_leases() {
    let t = Temp::new();
    let q = t.open().await;
    let (a, b, c) = (RunId::new(), RunId::new(), RunId::new());
    q.enqueue(a, 800).await.unwrap();
    q.enqueue(b, 700).await.unwrap();
    q.enqueue(c, 2_000).await.unwrap();
    let first = q.claim_due(1, 500).await.unwrap();
    assert_eq!(first.len(), 1);
    assert_eq!(first[0].run_id, b);
    assert_eq!(first[0].attempts, 1);
    let rest = q.claim_due(10, 500).await.unwrap();
    assert_eq!(rest.iter().map(|l| l.run_id).collect::<Vec<_>>(), vec![a]);
    assert_ne!(rest[0].token, first[0].token);
    assert_eq!(
        job(&q, a).await,
        Some(JobState::Leased {
            token: rest[0].token,
            until_ms: 1_500,
            next_due_ms: None
        })
    );
    q.complete(&first[0]).await.unwrap();
    assert_eq!(job(&q, b).await, Some(JobState::Done));
    q.fail(&rest[0], "boom").await.unwrap();
    assert_eq!(
        job(&q, a).await,
        Some(JobState::Failed {
            error: "boom".into()
        })
    );
    t.clock.advance(1_000);
    let late = q.claim_due(10, 500).await.unwrap();
    assert_eq!(late.iter().map(|l| l.run_id).collect::<Vec<_>>(), vec![c]);
    q.reschedule(&late[0], 2_500).await.unwrap();
    assert_eq!(job(&q, c).await, Some(JobState::Pending { due_ms: 2_500 }));
}

#[tokio::test]
async fn expired_lease_is_reclaimed() {
    let t = Temp::new();
    let q = t.open().await;
    let run = RunId::new();
    q.enqueue(run, 1_000).await.unwrap();
    let first = q.claim_due(10, 500).await.unwrap().remove(0);
    t.clock.advance(499);
    assert!(q.claim_due(10, 500).await.unwrap().is_empty());
    t.clock.advance(1);
    let second = q.claim_due(10, 500).await.unwrap().remove(0);
    assert_eq!(second.run_id, run);
    assert_eq!(second.attempts, 2);
    assert_ne!(second.token, first.token);
    q.reschedule(&second, 0).await.unwrap();
    assert_eq!(q.claim_due(10, 500).await.unwrap()[0].attempts, 1);
}

#[tokio::test]
async fn stale_token_changes_nothing() {
    let t = Temp::new();
    let q = t.open().await;
    let run = RunId::new();
    q.enqueue(run, 1_000).await.unwrap();
    let stale = q.claim_due(10, 500).await.unwrap().remove(0);
    t.clock.advance(500);
    let fresh = q.claim_due(10, 500).await.unwrap().remove(0);
    let held = job(&q, run).await;
    for result in [
        q.complete(&stale).await,
        q.fail(&stale, "x").await,
        q.reschedule(&stale, 0).await,
    ] {
        assert_eq!(
            result,
            Err(CoreError::Conflict {
                expected: fresh.token,
                actual: stale.token
            })
        );
    }
    assert_eq!(job(&q, run).await, held);
    q.complete(&fresh).await.unwrap();
    assert_eq!(
        q.complete(&fresh).await,
        Err(CoreError::Conflict {
            expected: 0,
            actual: fresh.token
        })
    );
}

#[tokio::test]
async fn enqueue_while_leased_applies_on_complete() {
    let t = Temp::new();
    let q = t.open().await;
    let run = RunId::new();
    q.enqueue(run, 1_000).await.unwrap();
    let lease = q.claim_due(10, 500).await.unwrap().remove(0);
    q.enqueue(run, 1_200).await.unwrap();
    q.enqueue(run, 1_100).await.unwrap();
    assert!(q.claim_due(10, 500).await.unwrap().is_empty());
    q.complete(&lease).await.unwrap();
    assert_eq!(
        job(&q, run).await,
        Some(JobState::Pending { due_ms: 1_100 })
    );
    assert!(
        q.claim_due(10, 500).await.unwrap().is_empty(),
        "not due until 1100"
    );
    q.enqueue(run, 1_000).await.unwrap();
    let lease2 = q.claim_due(10, 500).await.unwrap().remove(0);
    q.enqueue(run, 5_000).await.unwrap();
    q.reschedule(&lease2, 3_000).await.unwrap();
    assert_eq!(
        job(&q, run).await,
        Some(JobState::Pending { due_ms: 3_000 })
    );
}

#[tokio::test]
async fn leased_job_survives_reopen_and_expires() {
    let t = Temp::new();
    let run = RunId::new();
    let first = {
        let q = t.open().await;
        q.enqueue(run, 1_000).await.unwrap();
        q.claim_due(10, 500).await.unwrap().remove(0)
    };
    let q = t.open().await;
    assert_eq!(
        job(&q, run).await,
        Some(JobState::Leased {
            token: first.token,
            until_ms: 1_500,
            next_due_ms: None
        })
    );
    assert!(q.claim_due(10, 500).await.unwrap().is_empty());
    t.clock.advance(500);
    let second = q.claim_due(10, 500).await.unwrap().remove(0);
    assert_eq!(second.run_id, run);
    assert_eq!(second.attempts, first.attempts + 1);
}

#[tokio::test]
async fn token_never_repeats_after_reopen() {
    let t = Temp::new();
    let mut seen = Vec::new();
    for _ in 0..3 {
        let q = t.open().await;
        for _ in 0..3 {
            let run = RunId::new();
            q.enqueue(run, 0).await.unwrap();
            let lease = q.claim_due(1, 500).await.unwrap().remove(0);
            seen.push(lease.token);
            q.complete(&lease).await.unwrap();
        }
    }
    let mut unique = seen.clone();
    unique.sort();
    unique.dedup();
    assert_eq!(unique.len(), seen.len(), "{seen:?}");
}

#[tokio::test]
async fn ending_a_lease_of_an_unknown_job_is_not_found() {
    let t = Temp::new();
    let q = t.open().await;
    let run = RunId::new();
    let lease = troupekit_core::store::Lease {
        run_id: run,
        token: 1,
        attempts: 1,
    };
    let expected = CoreError::NotFound(format!("job for run {run}"));
    assert_eq!(q.complete(&lease).await, Err(expected.clone()));
    assert_eq!(q.fail(&lease, "x").await, Err(expected.clone()));
    assert_eq!(q.reschedule(&lease, 0).await, Err(expected));
    assert_eq!(job(&q, run).await, None);
}

#[tokio::test]
#[should_panic(expected = "a lease lasts some time")]
async fn zero_lease_panics_like_the_memory_queue() {
    let t = Temp::new();
    let q = t.open().await;
    let _ = q.claim_due(1, 0).await;
}

#[tokio::test]
async fn corrupt_job_row_is_a_store_error() {
    let t = Temp::new();
    let q = t.open().await;
    let (bad_attempts, bad_shape) = (RunId::new(), RunId::new());
    {
        let conn = rusqlite::Connection::open(&t.path).unwrap();
        conn.execute(
            "INSERT INTO job (run_id, state, claim_at_ms, attempts) VALUES (?1, 'pending', 0, ?2)",
            rusqlite::params![bad_attempts.to_string(), 5_000_000_000_i64],
        )
        .unwrap();
        conn.pragma_update(None, "ignore_check_constraints", true)
            .unwrap();
        conn.execute(
            "INSERT INTO job (run_id, state, claim_at_ms) VALUES (?1, 'leased', 0)",
            [bad_shape.to_string()],
        )
        .unwrap();
    }
    for run in [bad_attempts, bad_shape] {
        let err = q.job(run).await.unwrap_err();
        assert!(
            matches!(&err, CoreError::Store(m) if m.starts_with("corrupt store")),
            "{err:?}"
        );
    }
    let err = q.claim_due(10, 500).await.unwrap_err();
    assert!(
        matches!(&err, CoreError::Store(m) if m.starts_with("corrupt store")),
        "{err:?}"
    );
}
