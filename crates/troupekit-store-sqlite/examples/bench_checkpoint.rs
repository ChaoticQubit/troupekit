use serde_json::Value;
use std::time::{Duration, Instant};
use troupekit_core::ids::RunId;
use troupekit_core::store::{Checkpoint, CheckpointStore};
use troupekit_store_sqlite::{Durability, SqliteStore};

const SAVES: usize = 2_000;
const SIZES: [(&str, usize); 3] = [
    ("4 KiB", 4 * 1024),
    ("64 KiB", 64 * 1024),
    ("1 MiB", 1024 * 1024),
];

/// Times `SAVES` checkpoint writes per state size at one durability level
/// and prints p50, p99 and max per size plus the file size.
/// Input: `full` or `normal` as the first argument, default `full`.
/// Output: one line per size on stdout.
#[tokio::main(flavor = "multi_thread")]
async fn main() {
    let level = std::env::args().nth(1).unwrap_or_else(|| "full".into());
    let durability = match level.as_str() {
        "full" => Durability::Full,
        "normal" => Durability::Normal,
        other => panic!("durability must be full or normal, got {other}"),
    };
    println!(
        "sqlite {} durability {level} saves {SAVES}",
        rusqlite_version()
    );
    for (label, bytes) in SIZES {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("bench.db");
        let store = SqliteStore::builder(&path)
            .durability(durability)
            .open()
            .await
            .expect("open");
        let state = Value::String("x".repeat(bytes - 2));
        let run = RunId::new();
        let mut times = Vec::with_capacity(SAVES);
        for step in 0..SAVES as u64 {
            let cp = Checkpoint {
                run_id: run,
                step,
                state: state.clone(),
            };
            let start = Instant::now();
            store.save(cp).await.expect("save");
            times.push(start.elapsed());
        }
        times.sort();
        assert_eq!(times.len(), SAVES);
        let file = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0)
            + std::fs::metadata(dir.path().join("bench.db-wal"))
                .map(|m| m.len())
                .unwrap_or(0);
        println!(
            "{label:>7}  p50 {:>9}  p99 {:>9}  max {:>9}  file {:.1} MiB",
            ms(times[SAVES / 2]),
            ms(times[SAVES * 99 / 100]),
            ms(times[SAVES - 1]),
            file as f64 / (1024.0 * 1024.0)
        );
    }
}

fn ms(d: Duration) -> String {
    format!("{:.3} ms", d.as_secs_f64() * 1000.0)
}

fn rusqlite_version() -> &'static str {
    rusqlite::version()
}
