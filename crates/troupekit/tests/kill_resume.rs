#![cfg(unix)]

use crate::common;

use common::{graph_io, saved, start, tools, Hook};
use std::path::{Path, PathBuf};
use std::process::{Child, Command};
use std::sync::Arc;
use std::time::{Duration, Instant};
use troupekit::core::clock::{Clock, SystemClock};
use troupekit::core::ids::RunId;
use troupekit::core::store::JobState;
use troupekit::graph::{GraphRun, GraphStatus};
use troupekit::host::TickOptions;
use troupekit::{Durability, SqliteStore, Troupe};

const ROLE: &str = "TROUPEKIT_E2E_ROLE";
const DB: &str = "TROUPEKIT_E2E_DB";
const GATE: &str = "TROUPEKIT_E2E_GATE";
const SYNC: &str = "TROUPEKIT_E2E_SYNC";
const RUN: &str = "TROUPEKIT_E2E_RUN";
const MARKER: &str = "e2e/marker";
const DEADLINE: Duration = Duration::from_secs(10);
const LEASE_MS: i64 = 200;
const RESTARTS_MAX: usize = 5;

fn runs() -> usize {
    std::env::var("TROUPEKIT_E2E_RUNS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3)
}

fn clock() -> Arc<dyn Clock> {
    Arc::new(SystemClock)
}

fn durability_name(d: Durability) -> &'static str {
    match d {
        Durability::Normal => "normal",
        _ => "full",
    }
}

async fn troupe_on(path: &Path, durability: Durability) -> Troupe {
    let store = SqliteStore::builder(path).durability(durability);
    Troupe::builder().sqlite_with(store).build().await.unwrap()
}

/// Kills and reaps the worker when dropped, so a failed assertion never
/// leaves a child process alive.
struct Worker(Child);

impl Drop for Worker {
    fn drop(&mut self) {
        if matches!(self.0.try_wait(), Ok(None)) {
            let _ = self.0.kill();
        }
        let _ = self.0.wait();
    }
}

impl Worker {
    fn spawn(path: &Path, id: RunId, durability: Durability, gate: bool) -> Worker {
        let mut cmd = Command::new(std::env::current_exe().unwrap());
        cmd.args([
            "--ignored",
            "--exact",
            "kill_resume::kill_resume_worker",
            "--test-threads=1",
            "--quiet",
        ])
        .env(ROLE, "worker")
        .env(DB, path)
        .env(RUN, id.to_string())
        .env(SYNC, durability_name(durability))
        .stdout(std::process::Stdio::null());
        if gate {
            cmd.env(GATE, "1");
        }
        Worker(cmd.spawn().unwrap())
    }

    fn kill(mut self) {
        self.0.kill().unwrap();
        let status = self.0.wait().unwrap();
        assert!(!status.success(), "a killed worker does not exit cleanly");
    }

    fn kill_any(mut self) {
        self.0.kill().unwrap();
        self.0.wait().unwrap();
    }

    async fn finish(mut self) {
        let start = Instant::now();
        loop {
            if let Some(status) = self.0.try_wait().unwrap() {
                assert!(status.success(), "worker failed: {status}");
                return;
            }
            assert!(
                start.elapsed() < DEADLINE,
                "worker did not finish within {DEADLINE:?}"
            );
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }

    fn running(&mut self) -> bool {
        matches!(self.0.try_wait(), Ok(None))
    }
}

async fn wait_for_marker(store: &SqliteStore) {
    use troupekit::core::store::BlobStore;
    let start = Instant::now();
    while store.get(MARKER).await.is_err() {
        assert!(
            start.elapsed() < DEADLINE,
            "worker never reached the second tool"
        );
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}

async fn reference(dir: &Path) -> GraphRun {
    let troupe = troupe_on(&dir.join("reference.db"), Durability::Full).await;
    let io = graph_io(&troupe, tools(None), clock());
    let id = start(&troupe, io.clone(), "kill me").await;
    common::finish(&troupe, io, id).await
}

async fn assert_identical(path: &Path, id: RunId, reference: &GraphRun) {
    let store = SqliteStore::open(path).await.unwrap();
    assert_eq!(store.job(id).await.unwrap(), Some(JobState::Done));
    let troupe = troupe_on(path, Durability::Full).await;
    let done = saved(&troupe, id).await;
    assert_eq!(done.status(), &GraphStatus::Done);
    assert_eq!(done.state(), reference.state());
    assert_eq!(done.steps(), reference.steps());
    assert_eq!(done.usage(), reference.usage());
    let steps: Vec<u64> = troupe
        .checkpoints()
        .history(id)
        .await
        .unwrap()
        .iter()
        .map(|c| c.step)
        .collect();
    assert_eq!(steps, (0..=done.steps()).collect::<Vec<_>>());
}

async fn new_run(path: &Path, durability: Durability) -> RunId {
    let troupe = troupe_on(path, durability).await;
    let io = graph_io(&troupe, tools(None), clock());
    start(&troupe, io, "kill me").await
}

async fn killed_mid_step_once(
    dir: &Path,
    round: usize,
    durability: Durability,
    reference: &GraphRun,
) {
    let path = dir.join(format!("mid-{round}-{}.db", durability_name(durability)));
    let id = new_run(&path, durability).await;
    let watcher = SqliteStore::open(&path).await.unwrap();
    let first = Worker::spawn(&path, id, durability, true);
    wait_for_marker(&watcher).await;
    first.kill();
    let troupe = troupe_on(&path, durability).await;
    assert_ne!(saved(&troupe, id).await.status(), &GraphStatus::Done);
    Worker::spawn(&path, id, durability, false).finish().await;
    assert_identical(&path, id, reference).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn killed_mid_step_resumes_to_identical_output() {
    let dir = tempfile::tempdir().unwrap();
    let reference = reference(dir.path()).await;
    for round in 0..runs() {
        for durability in [Durability::Full, Durability::Normal] {
            killed_mid_step_once(dir.path(), round, durability, &reference).await;
        }
    }
}

struct XorShift(u64);

impl XorShift {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

async fn killed_at_random_once(dir: &Path, round: usize, rng: &mut XorShift, reference: &GraphRun) {
    let path = dir.join(format!("random-{round}.db"));
    let id = new_run(&path, Durability::Full).await;
    for _ in 0..RESTARTS_MAX {
        let mut worker = Worker::spawn(&path, id, Durability::Full, false);
        tokio::time::sleep(Duration::from_millis(rng.next() % 51)).await;
        if !worker.running() {
            worker.finish().await;
            break;
        }
        worker.kill_any();
    }
    let store = SqliteStore::open(&path).await.unwrap();
    if store.job(id).await.unwrap() != Some(JobState::Done) {
        Worker::spawn(&path, id, Durability::Full, false)
            .finish()
            .await;
    }
    assert_identical(&path, id, reference).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn killed_at_random_point_resumes_to_identical_output() {
    let seed = std::env::var("TROUPEKIT_E2E_SEED")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64
                | 1
        });
    eprintln!("kill seed {seed} (set TROUPEKIT_E2E_SEED to replay)");
    let mut rng = XorShift(seed);
    let dir = tempfile::tempdir().unwrap();
    let reference = reference(dir.path()).await;
    for round in 0..runs() {
        killed_at_random_once(dir.path(), round, &mut rng, &reference).await;
    }
}

fn gate_hook(troupe: &Troupe) -> Hook {
    let blobs = troupe.blobs();
    Arc::new(move || {
        let blobs = blobs.clone();
        Box::pin(async move {
            blobs.put(MARKER, vec![1]).await.unwrap();
            std::future::pending::<()>().await;
        })
    })
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "worker process of the kill tests, started by them"]
async fn kill_resume_worker() {
    if std::env::var(ROLE).as_deref() != Ok("worker") {
        return;
    }
    let path = PathBuf::from(std::env::var(DB).unwrap());
    let durability = match std::env::var(SYNC).as_deref() {
        Ok("normal") => Durability::Normal,
        _ => Durability::Full,
    };
    let troupe = troupe_on(&path, durability).await;
    let hook = std::env::var(GATE).is_ok().then(|| gate_hook(&troupe));
    let io = graph_io(&troupe, tools(hook), clock());
    let runner = troupe.runner(io).with_lease_ms(LEASE_MS);
    let id: RunId = serde_json::from_value(serde_json::json!(std::env::var(RUN).unwrap())).unwrap();
    let store = SqliteStore::open(&path).await.unwrap();
    let start = Instant::now();
    loop {
        runner.tick(TickOptions::default()).await.unwrap();
        if store.job(id).await.unwrap() == Some(JobState::Done) {
            return;
        }
        assert!(start.elapsed() < DEADLINE, "worker ran past its deadline");
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}
