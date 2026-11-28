use serde_json::json;
use std::path::{Path, PathBuf};
use std::process::{Child, Command};
use std::time::{Duration, Instant};
use troupekit_core::error::CoreError;
use troupekit_core::ids::RunId;
use troupekit_core::store::*;
use troupekit_store_sqlite::SqliteStore;

const CHILD_DB: &str = "TROUPEKIT_BRANCH_WRITE_CHILD_DB";
const CHILD_RUN: &str = "TROUPEKIT_BRANCH_WRITE_CHILD_RUN";
const DEADLINE: Duration = Duration::from_secs(60);

struct Temp {
    _dir: tempfile::TempDir,
    path: PathBuf,
}

fn temp() -> Temp {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("store.db");
    Temp { _dir: dir, path }
}

fn cp(run_id: RunId, step: u64) -> Checkpoint {
    Checkpoint {
        run_id,
        step,
        state: json!({ "step": step }),
    }
}

async fn save_steps(s: &SqliteStore, run: RunId, steps: u64) {
    for step in 0..steps {
        s.save(cp(run, step)).await.unwrap();
        s.drop_branch_writes(run, step + 1).await.unwrap();
    }
}

fn rows(path: &Path, sql: &str) -> i64 {
    let conn = rusqlite::Connection::open(path).unwrap();
    conn.query_row(sql, [], |row| row.get(0)).unwrap()
}

#[tokio::test]
async fn branch_write_round_trips() {
    let t = temp();
    let run = RunId::new();
    let writes = vec![
        BranchWrite::new(0, "left", json!({"at": "leave", "out": {"text": "a"}})),
        BranchWrite::new(2, "right", json!({"at": "leave", "out": [1, 2, 3]})),
    ];
    {
        let s = SqliteStore::open(&t.path).await.unwrap();
        save_steps(&s, run, 1).await;
        assert!(s.save_branch(run, 1, writes[1].clone()).await.unwrap());
        assert!(s.save_branch(run, 1, writes[0].clone()).await.unwrap());
        assert!(s.save_branch(RunId::new(), 1, writes[0].clone()).await.unwrap());
        assert_eq!(s.branch_writes(run, 1).await.unwrap(), writes);
    }
    let s = SqliteStore::open(&t.path).await.unwrap();
    assert_eq!(s.branch_writes(run, 1).await.unwrap(), writes);
    assert!(s.branch_writes(run, 2).await.unwrap().is_empty());
    assert!(s.branch_writes(RunId::new(), 1).await.unwrap().is_empty());
}

#[tokio::test]
async fn save_keeps_branch_writes_until_dropped() {
    let t = temp();
    let s = SqliteStore::open(&t.path).await.unwrap();
    let run = RunId::new();
    save_steps(&s, run, 1).await;
    assert!(s.save_branch(run, 1, BranchWrite::new(0, "a", json!(1))).await.unwrap());
    assert!(s.save_branch(run, 1, BranchWrite::new(1, "b", json!(2))).await.unwrap());
    assert!(s.save_branch(run, 2, BranchWrite::new(0, "a", json!(3))).await.unwrap());
    s.save(cp(run, 1)).await.unwrap();
    s.save(cp(run, 2)).await.unwrap();
    assert_eq!(s.branch_writes(run, 1).await.unwrap().len(), 2);
    assert_eq!(s.prune(run, 1).await.unwrap(), 2);
    assert_eq!(rows(&t.path, "SELECT COUNT(*) FROM branch_write"), 3);
    assert_eq!(s.drop_branch_writes(run, 1).await.unwrap(), 0);
    assert_eq!(s.drop_branch_writes(run, 2).await.unwrap(), 2);
    assert_eq!(s.branch_writes(run, 2).await.unwrap(), vec![BranchWrite::new(0, "a", json!(3))]);
    assert_eq!(s.drop_branch_writes(run, 3).await.unwrap(), 1);
    assert_eq!(rows(&t.path, "SELECT COUNT(*) FROM branch_write"), 0);
    assert_eq!(s.drop_branch_writes(RunId::new(), 9).await.unwrap(), 0);
}

#[tokio::test]
async fn branch_write_for_a_saved_step_is_refused() {
    let t = temp();
    let s = SqliteStore::open(&t.path).await.unwrap();
    let run = RunId::new();
    save_steps(&s, run, 3).await;
    for step in [0, 2] {
        let err = s.save_branch(run, step, BranchWrite::new(0, "a", json!(1))).await.unwrap_err();
        assert_eq!(err, CoreError::Conflict { expected: 3, actual: step });
    }
    assert_eq!(rows(&t.path, "SELECT COUNT(*) FROM branch_write"), 0);
    let err = s.save_branch(run, u64::MAX, BranchWrite::new(0, "a", json!(1))).await.unwrap_err();
    assert!(matches!(err, CoreError::Store(_)), "{err:?}");
}

#[tokio::test]
async fn second_different_write_for_a_branch_conflicts() {
    let t = temp();
    let s = SqliteStore::open(&t.path).await.unwrap();
    let run = RunId::new();
    save_steps(&s, run, 1).await;
    let write = BranchWrite::new(0, "a", json!({"out": 1}));
    assert!(s.save_branch(run, 1, write.clone()).await.unwrap());
    assert!(s.save_branch(run, 1, write.clone()).await.unwrap());
    for other in [BranchWrite::new(0, "a", json!({"out": 2})), BranchWrite::new(0, "b", json!({"out": 1}))] {
        let err = s.save_branch(run, 1, other).await.unwrap_err();
        assert_eq!(err, CoreError::Conflict { expected: 1, actual: 1 });
    }
    assert_eq!(s.branch_writes(run, 1).await.unwrap(), vec![write]);
}

#[tokio::test]
async fn oversized_branch_write_is_refused() {
    let t = temp();
    let run = RunId::new();
    let big = BranchWrite::new(0, "a", json!("x".repeat(200)));
    {
        let s = SqliteStore::builder(&t.path).max_branch_write_bytes(100).open().await.unwrap();
        save_steps(&s, run, 1).await;
        let err = s.save_branch(run, 1, big.clone()).await.unwrap_err();
        assert!(matches!(&err, CoreError::Store(m) if m.contains("max_branch_write_bytes")), "{err:?}");
        assert!(!format!("{err}").contains("xxxx"), "the error never holds the write");
        assert!(s.save_branch(run, 1, BranchWrite::new(0, "a", json!("small"))).await.unwrap());
    }
    {
        let s = SqliteStore::builder(&t.path).max_state_bytes(100).open().await.unwrap();
        let err = s.save_branch(run, 1, BranchWrite::new(1, "b", json!("x".repeat(200)))).await.unwrap_err();
        assert!(matches!(&err, CoreError::Store(m) if m.contains("max_branch_write_bytes")), "{err:?}");
        let mut deep = json!(1);
        for _ in 0..200 {
            deep = json!([deep]);
        }
        let err = s.save_branch(run, 1, BranchWrite::new(1, "b", deep)).await.unwrap_err();
        assert!(matches!(&err, CoreError::Store(m) if m.contains("nests deeper")), "{err:?}");
    }
    assert_eq!(rows(&t.path, "SELECT COUNT(*) FROM branch_write"), 1);
    for bytes in [0, 1_000_000_001] {
        let err = SqliteStore::builder(&t.path).max_branch_write_bytes(bytes).open().await.unwrap_err();
        assert!(matches!(&err, CoreError::Store(m) if m.contains("max_branch_write_bytes")), "{err:?}");
    }
}

#[tokio::test]
async fn fork_links_survive_reopen() {
    let t = temp();
    let (parent, other) = (RunId::new(), RunId::new());
    let (first, second, third) = (RunId::new(), RunId::new(), RunId::new());
    {
        let s = SqliteStore::open(&t.path).await.unwrap();
        assert!(s.link_fork(ForkLink::new(second, parent, 1)).await.unwrap());
        assert!(s.link_fork(ForkLink::new(third, other, 0)).await.unwrap());
        assert!(s.link_fork(ForkLink::new(first, parent, 4)).await.unwrap());
        assert!(s.link_fork(ForkLink::new(first, parent, 4)).await.unwrap());
    }
    let s = SqliteStore::open(&t.path).await.unwrap();
    assert_eq!(
        s.forks(parent).await.unwrap(),
        vec![ForkLink::new(first, parent, 4), ForkLink::new(second, parent, 1)]
    );
    assert_eq!(s.forks(other).await.unwrap(), vec![ForkLink::new(third, other, 0)]);
    assert!(s.forks(first).await.unwrap().is_empty());
    let err = s.link_fork(ForkLink::new(first, other, 4)).await.unwrap_err();
    assert!(matches!(err, CoreError::Conflict { .. }), "{err:?}");
    let err = s.link_fork(ForkLink::new(parent, parent, 0)).await.unwrap_err();
    assert!(matches!(err, CoreError::Store(_)), "{err:?}");
    assert_eq!(rows(&t.path, "SELECT COUNT(*) FROM run_fork"), 3);
}

struct Worker(Child);

impl Drop for Worker {
    fn drop(&mut self) {
        if matches!(self.0.try_wait(), Ok(None)) {
            let _ = self.0.kill();
        }
        let _ = self.0.wait();
    }
}

#[cfg(unix)]
#[tokio::test]
async fn killed_between_branch_write_and_save_keeps_both_consistent() {
    let t = temp();
    let run = RunId::new();
    drop(SqliteStore::open(&t.path).await.unwrap());
    let mut worker = Worker(
        Command::new(std::env::current_exe().unwrap())
            .args(["--ignored", "--exact", "branch_writes::child_writes_branches_until_killed", "--test-threads=1", "--quiet"])
            .env(CHILD_DB, &t.path)
            .env(CHILD_RUN, run.to_string())
            .stdout(std::process::Stdio::null())
            .spawn()
            .unwrap(),
    );
    let start = Instant::now();
    while rows(&t.path, "SELECT COUNT(*) FROM checkpoint") < 50 {
        assert!(start.elapsed() < DEADLINE, "child saved too few steps");
        assert!(matches!(worker.0.try_wait(), Ok(None)), "child exited early");
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    worker.0.kill().unwrap();
    let status = worker.0.wait().unwrap();
    assert!(!status.success(), "a killed child does not exit cleanly");
    let s = SqliteStore::open(&t.path).await.unwrap();
    let latest = s.latest(run).await.unwrap().unwrap().step;
    let stale = format!("SELECT COUNT(*) FROM branch_write WHERE run_id = '{run}' AND step < {latest}");
    assert_eq!(rows(&t.path, &stale), 0, "every write below the saved step was dropped");
    let undropped = format!("SELECT COUNT(*) FROM branch_write WHERE run_id = '{run}' AND step = {latest}");
    assert!(rows(&t.path, &undropped) <= 2, "a kill between save and drop leaves the saved step's writes");
    let ahead = format!("SELECT COUNT(*) FROM branch_write WHERE run_id = '{run}' AND step > {}", latest + 1);
    assert_eq!(rows(&t.path, &ahead), 0, "writes exist only for the step in flight");
    let pending = s.branch_writes(run, latest + 1).await.unwrap();
    assert!(pending.len() <= 2, "{pending:?}");
    for (index, write) in pending.iter().enumerate() {
        assert_eq!(write.branch() as usize, index);
        assert_eq!(write.write(), &json!({ "step": latest + 1, "branch": index }));
    }
    s.save(cp(run, latest + 1)).await.unwrap();
    assert_eq!(s.branch_writes(run, latest + 1).await.unwrap(), pending, "a save keeps the writes");
    s.drop_branch_writes(run, latest + 2).await.unwrap();
    assert_eq!(rows(&t.path, &format!("SELECT COUNT(*) FROM branch_write WHERE run_id = '{run}'")), 0);
}

#[tokio::test]
#[ignore = "child process of killed_between_branch_write_and_save_keeps_both_consistent"]
async fn child_writes_branches_until_killed() {
    let (Ok(path), Ok(run)) = (std::env::var(CHILD_DB), std::env::var(CHILD_RUN)) else {
        return;
    };
    let run: RunId = serde_json::from_value(serde_json::Value::String(run)).unwrap();
    let s = SqliteStore::open(&path).await.unwrap();
    s.save(cp(run, 0)).await.unwrap();
    for step in 1..1_000_000u64 {
        for branch in 0..2u32 {
            let write = BranchWrite::new(branch, format!("n{branch}"), json!({ "step": step, "branch": branch }));
            assert!(s.save_branch(run, step, write).await.unwrap());
        }
        s.save(cp(run, step)).await.unwrap();
        s.drop_branch_writes(run, step + 1).await.unwrap();
    }
}
