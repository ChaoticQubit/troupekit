use serde_json::{json, Value};
use std::time::{Duration, Instant};
use troupekit_core::ids::RunId;
use troupekit_core::store::{BranchWrite, Checkpoint, CheckpointStore};
use troupekit_store_sqlite::{Durability, SqliteStore};

const WRITES_DEFAULT: usize = 1_000;
const WRITES_MAX: usize = 1_000_000;
const STATE_BYTES: usize = 2 * 1024;

/// Latencies of one measured run: per step one branch write, then the
/// checkpoint save that settles it, both of a 2 KiB state.
/// Each list is sorted, shortest first, and holds one entry per step.
pub struct Measured {
    pub branch_writes: Vec<Duration>,
    pub checkpoint_saves: Vec<Duration>,
}

/// Times `writes` steps at `durability` on a fresh file in a temp dir:
/// each step does one `save_branch` and then the `save` of that step.
/// Input: 1 to 1 000 000 steps and the level. Output: the sorted timings.
pub async fn measure(writes: usize, durability: Durability) -> Measured {
    assert!((1..=WRITES_MAX).contains(&writes), "writes must be 1 to {WRITES_MAX}");
    let dir = tempfile::tempdir().expect("temp dir");
    let store = SqliteStore::builder(dir.path().join("branch_writes.db"))
        .durability(durability)
        .open()
        .await
        .expect("open");
    let state = Value::String("x".repeat(STATE_BYTES - 2));
    let run = RunId::new();
    store.save(Checkpoint { run_id: run, step: 0, state: state.clone() }).await.expect("save step 0");
    let mut measured = Measured {
        branch_writes: Vec::with_capacity(writes),
        checkpoint_saves: Vec::with_capacity(writes),
    };
    for step in 1..=writes as u64 {
        let write = BranchWrite::new(0, "work", json!({ "out": state }));
        let start = Instant::now();
        assert!(store.save_branch(run, step, write).await.expect("save_branch"));
        measured.branch_writes.push(start.elapsed());
        let start = Instant::now();
        store.save(Checkpoint { run_id: run, step, state: state.clone() }).await.expect("save");
        measured.checkpoint_saves.push(start.elapsed());
    }
    measured.branch_writes.sort();
    measured.checkpoint_saves.sort();
    assert_eq!(measured.branch_writes.len(), writes);
    assert_eq!(measured.checkpoint_saves.len(), writes);
    measured
}

/// Returns the `pct` percentile of the sorted, non-empty `times`.
/// Input: 0 to 100. Output: the entry at that rank, rounded down.
pub fn percentile(times: &[Duration], pct: usize) -> Duration {
    assert!(!times.is_empty() && pct <= 100, "percentile of an empty list or above 100");
    times[(times.len() - 1) * pct / 100]
}

/// Prints p50 and p99 of branch writes and checkpoint saves under
/// `Durability::Full` and `Normal`, side by side, so the cost of one
/// durable branch write is visible next to a checkpoint save.
/// Input: the step count as the first argument, default 1 000.
#[tokio::main(flavor = "multi_thread")]
async fn main() {
    let writes = std::env::args()
        .nth(1)
        .map(|arg| arg.parse::<usize>().expect("the step count is a number"))
        .unwrap_or(WRITES_DEFAULT);
    println!("sqlite {} steps {writes} state {STATE_BYTES} bytes", rusqlite::version());
    for (label, durability) in [("full", Durability::Full), ("normal", Durability::Normal)] {
        let m = measure(writes, durability).await;
        println!(
            "{label:>6}  branch write p50 {:>9} p99 {:>9}  checkpoint save p50 {:>9} p99 {:>9}",
            ms(percentile(&m.branch_writes, 50)),
            ms(percentile(&m.branch_writes, 99)),
            ms(percentile(&m.checkpoint_saves, 50)),
            ms(percentile(&m.checkpoint_saves, 99)),
        );
    }
}

fn ms(d: Duration) -> String {
    format!("{:.3} ms", d.as_secs_f64() * 1000.0)
}
