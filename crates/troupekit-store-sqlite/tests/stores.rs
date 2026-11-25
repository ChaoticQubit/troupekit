use serde_json::json;
use std::path::PathBuf;
use troupekit_core::error::CoreError;
use troupekit_core::ids::RunId;
use troupekit_core::store::*;
use troupekit_store_sqlite::{SqliteStore, SqliteStoreBuilder};

struct Temp {
    _dir: tempfile::TempDir,
    path: PathBuf,
}

fn temp() -> Temp {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("store.db");
    Temp { _dir: dir, path }
}

impl Temp {
    async fn open(&self) -> SqliteStore {
        SqliteStore::open(&self.path).await.unwrap()
    }

    async fn with(&self, f: impl FnOnce(SqliteStoreBuilder) -> SqliteStoreBuilder) -> SqliteStore {
        f(SqliteStore::builder(&self.path)).open().await.unwrap()
    }
}

fn cp(run_id: RunId, step: u64) -> Checkpoint {
    Checkpoint {
        run_id,
        step,
        state: json!({ "step": step }),
    }
}

async fn save_steps(s: &SqliteStore, run: RunId, steps: std::ops::RangeInclusive<u64>) {
    for step in steps {
        s.save(cp(run, step)).await.unwrap();
    }
}

#[tokio::test]
async fn checkpoints_must_advance_by_one() {
    let t = temp();
    let s = t.open().await;
    let run = RunId::new();
    s.save(Checkpoint {
        run_id: run,
        step: 0,
        state: json!({"a":1}),
    })
    .await
    .unwrap();
    s.save(Checkpoint {
        run_id: run,
        step: 1,
        state: json!({"a":2}),
    })
    .await
    .unwrap();
    let err = s
        .save(Checkpoint {
            run_id: run,
            step: 1,
            state: json!({}),
        })
        .await
        .unwrap_err();
    assert_eq!(
        err,
        CoreError::Conflict {
            expected: 2,
            actual: 1
        }
    );
    assert_eq!(s.latest(run).await.unwrap().unwrap().state, json!({"a":2}));
    assert!(s.latest(RunId::new()).await.unwrap().is_none());
    let err = s.save(cp(RunId::new(), 3)).await.unwrap_err();
    assert_eq!(
        err,
        CoreError::Conflict {
            expected: 0,
            actual: 3
        }
    );
}

#[tokio::test]
async fn checkpoint_survives_reopen() {
    let t = temp();
    let run = RunId::new();
    save_steps(&t.open().await, run, 0..=2).await;
    assert_eq!(t.open().await.latest(run).await.unwrap(), Some(cp(run, 2)));
}

#[tokio::test]
async fn latest_of_unknown_run_is_none() {
    let t = temp();
    assert_eq!(t.open().await.latest(RunId::new()).await.unwrap(), None);
}

#[tokio::test]
async fn history_lists_every_step_in_order() {
    let t = temp();
    let s = t.open().await;
    let run = RunId::new();
    let other = RunId::new();
    save_steps(&s, run, 0..=4).await;
    save_steps(&s, other, 0..=1).await;
    let expected: Vec<Checkpoint> = (0..=4).map(|n| cp(run, n)).collect();
    assert_eq!(s.history(run).await.unwrap(), expected);
    assert!(s.history(RunId::new()).await.unwrap().is_empty());
}

#[tokio::test]
async fn prune_keeps_the_last_n_steps() {
    let t = temp();
    let s = t.open().await;
    let run = RunId::new();
    save_steps(&s, run, 0..=5).await;
    assert_eq!(s.prune(run, 2).await.unwrap(), 4);
    assert_eq!(s.history(run).await.unwrap(), vec![cp(run, 4), cp(run, 5)]);
    assert_eq!(s.prune(run, 10).await.unwrap(), 0);
    assert_eq!(s.prune(RunId::new(), 1).await.unwrap(), 0);
}

#[tokio::test]
async fn save_after_prune_takes_the_next_step() {
    let t = temp();
    let s = t.open().await;
    let run = RunId::new();
    save_steps(&s, run, 0..=5).await;
    s.prune(run, 1).await.unwrap();
    s.save(cp(run, 6)).await.unwrap();
    let err = s.save(cp(run, 0)).await.unwrap_err();
    assert_eq!(
        err,
        CoreError::Conflict {
            expected: 7,
            actual: 0
        }
    );
}

#[tokio::test]
async fn prune_rejects_keep_last_zero() {
    let t = temp();
    let s = t.open().await;
    let run = RunId::new();
    save_steps(&s, run, 0..=2).await;
    assert!(matches!(s.prune(run, 0).await, Err(CoreError::Store(_))));
    assert_eq!(s.history(run).await.unwrap().len(), 3);
}

#[tokio::test]
async fn step_beyond_i64_is_rejected() {
    let t = temp();
    let s = t.open().await;
    let err = s.save(cp(RunId::new(), u64::MAX)).await.unwrap_err();
    assert!(matches!(err, CoreError::Store(_)), "{err:?}");
}

#[tokio::test]
async fn state_above_the_setting_is_rejected() {
    let t = temp();
    let s = t.with(|b| b.max_state_bytes(16)).await;
    let run = RunId::new();
    let big = Checkpoint {
        run_id: run,
        step: 0,
        state: json!("this text is longer than 16"),
    };
    let err = s.save(big).await.unwrap_err();
    assert!(
        matches!(&err, CoreError::Store(m) if m.contains("max_state_bytes")),
        "{err:?}"
    );
    assert!(s.latest(run).await.unwrap().is_none());
    s.save(Checkpoint {
        run_id: run,
        step: 0,
        state: json!("short"),
    })
    .await
    .unwrap();
}

#[tokio::test]
async fn history_above_the_setting_is_rejected() {
    let t = temp();
    let s = t.with(|b| b.max_history_steps(2)).await;
    let run = RunId::new();
    save_steps(&s, run, 0..=2).await;
    let err = s.history(run).await.unwrap_err();
    assert!(
        matches!(&err, CoreError::Store(m) if m.contains("max_history_steps")),
        "{err:?}"
    );
    s.prune(run, 2).await.unwrap();
    assert_eq!(s.history(run).await.unwrap().len(), 2);
}
