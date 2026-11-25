//! Index-only query latency and memory at 50 000 vectors, 384 dims.
//! Run: `cargo run --release -p troupekit-index --example latency`.
//! Prints p50, p99 and heap bytes per kind and quantization.

mod common;

use std::time::Instant;

use troupekit_index::{new_index, HnswParams, IndexKind, Quantization};

const DIMS: usize = 384;
const QUERIES: usize = 2_000;
const K: usize = 10;

fn percentile(sorted_us: &[f64], p: f64) -> f64 {
    assert!(!sorted_us.is_empty(), "no samples");
    let i = ((sorted_us.len() as f64 - 1.0) * p).round() as usize;
    sorted_us[i]
}

fn main() {
    let n = common::count_arg(50_000);
    let data = common::clustered(0x4C41_5459, n + QUERIES, DIMS, 100, 0.05);
    let (items, queries) = data.split_at(n);
    println!("vectors={n} dims={DIMS} queries={QUERIES} k={K}");
    println!("| kind | quant | p50 us | p99 us | heap MB | build s |");
    println!("|---|---|---|---|---|---|");
    for (name, kind) in [("brute", IndexKind::BruteForce), ("hnsw", IndexKind::Hnsw(HnswParams::default()))] {
        for quant in [Quantization::F32, Quantization::F16, Quantization::I8] {
            let started = Instant::now();
            let mut index = new_index(&kind, DIMS, quant).expect("index");
            for (i, v) in items.iter().enumerate() {
                index.insert(i as u64, v).expect("insert");
            }
            let build_s = started.elapsed().as_secs_f64();
            let mut us: Vec<f64> = queries
                .iter()
                .map(|q| {
                    let t = Instant::now();
                    let hits = index.search(q, K, None).expect("search");
                    assert_eq!(hits.len(), K);
                    t.elapsed().as_secs_f64() * 1e6
                })
                .collect();
            us.sort_by(f64::total_cmp);
            let mb = index.heap_bytes() as f64 / (1024.0 * 1024.0);
            println!("| {name} | {quant:?} | {:.0} | {:.0} | {mb:.1} | {build_s:.1} |", percentile(&us, 0.5), percentile(&us, 0.99));
        }
    }
}
