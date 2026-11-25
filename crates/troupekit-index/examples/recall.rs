//! Recall@10 of HNSW against brute force on clustered 384-dim data.
//! Run: `cargo run --release -p troupekit-index --example recall -- 100000`.
//! Asserts the default `ef_search` reaches 0.95.

mod common;

use std::time::Instant;

use troupekit_index::{new_index, HnswParams, IndexKind, Quantization};

const DIMS: usize = 384;
const QUERIES: usize = 1_000;
const CLUSTERS: usize = 100;
const SPREAD: f32 = 0.05;
const EF: [usize; 5] = [16, 32, 64, 128, 256];

fn main() {
    let n = common::count_arg(100_000);
    let data = common::clustered(0x5245_4341, n + QUERIES, DIMS, CLUSTERS, SPREAD);
    let (items, queries) = data.split_at(n);

    let mut oracle = new_index(&IndexKind::BruteForce, DIMS, Quantization::F32).expect("brute force index");
    for (i, v) in items.iter().enumerate() {
        oracle.insert(i as u64, v).expect("insert into brute force");
    }
    let truth: Vec<Vec<u64>> = queries.iter().map(|q| oracle.search(q, 10, None).expect("brute force search").iter().map(|h| h.id).collect()).collect();
    drop(oracle);

    let default_ef = HnswParams::default().ef_search();
    let mut default_recall = None;
    println!("vectors={n} dims={DIMS} queries={QUERIES} clusters={CLUSTERS}");
    for ef in EF {
        let params = HnswParams::default().with_ef_search(ef);
        let started = Instant::now();
        let mut index = new_index(&IndexKind::Hnsw(params), DIMS, Quantization::F32).expect("hnsw index");
        for (i, v) in items.iter().enumerate() {
            index.insert(i as u64, v).expect("insert into hnsw");
        }
        let build_s = started.elapsed().as_secs_f64();
        let started = Instant::now();
        let mut found = 0usize;
        for (q, want) in queries.iter().zip(&truth) {
            let got = index.search(q, 10, None).expect("hnsw search");
            found += got.iter().filter(|h| want.contains(&h.id)).count();
        }
        let query_ms = started.elapsed().as_secs_f64() * 1_000.0 / QUERIES as f64;
        let recall = found as f64 / (QUERIES * 10) as f64;
        println!("ef_search={ef:<4} recall@10={recall:.4} build_s={build_s:.1} mean_query_ms={query_ms:.3}");
        if ef == default_ef {
            default_recall = Some(recall);
        }
    }
    let recall = default_recall.expect("default ef_search is one of the measured values");
    assert!(recall >= 0.95, "recall_at_10_is_at_least_0_95_on_100k: default ef_search {default_ef} gave {recall:.4}");
    println!("recall_at_10_is_at_least_0_95_on_100k: ok, default ef_search {default_ef} gave {recall:.4}");
}
