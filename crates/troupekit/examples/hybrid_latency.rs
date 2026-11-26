//! Hybrid query latency over one SQLite file: seeded chunks with the
//! clustered 384-dimension vectors of the index benchmarks (100 clusters,
//! spread 0.05), then timed `hybrid_search` calls with `top_k` 8,
//! for brute force and HNSW in f32 and i8. Run with `cargo run --release
//! -p troupekit --features knowledge --example hybrid_latency -- [chunks] [queries]`.

mod common;

use std::collections::{BTreeMap, HashMap};
use std::process::ExitCode;
use std::time::{Duration, Instant};
use troupekit::context::embed::Embedder;
use troupekit::context::hybrid::{hybrid_search, HybridConfig};
use troupekit::context::ingest::{META_ORIGIN, META_SCOPE, META_SOURCE};
use troupekit::context::ContextError;
use troupekit::core::store::{Filter, VectorRecord, VectorStore};
use troupekit::{HnswParams, IndexKind, Quantization, SqliteStore, VectorSettings};

const DIMS: usize = 384;
const CHUNKS_DEFAULT: usize = 50_000;
const QUERIES_DEFAULT: usize = 2_000;
const TOP_K: usize = 8;
const BATCH: usize = 5_000;
const SCOPE: &str = "docs";
const SEED: u64 = 0x4C41_5459;
const CLUSTERS: usize = 100;
const SPREAD: f32 = 0.05;
const WORDS: [&str; 32] = [
    "refund", "parcel", "invoice", "account", "password", "delivery", "return", "label", "card", "order", "size", "colour", "warranty", "gift", "voucher", "shipping",
    "tracking", "address", "email", "login", "basket", "checkout", "payment", "bank", "express", "furniture", "carrier", "damaged", "exchange", "credit", "app", "status",
];

/// Splitmix64 step: advances `state` and returns the next value.
/// Input: the state. Output: a well-mixed 64-bit value.
/// Seeded, so every run generates the same data.
fn next(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Seeded text of 24 words for chunk or query `index` under `salt`.
/// Input: index and salt. Output: the text.
/// Words come from a fixed support vocabulary so BM25 has matches.
fn seeded_text(index: usize, salt: u64) -> String {
    let mut state = (index as u64) ^ salt;
    (0..24).map(|_| WORDS[(next(&mut state) % WORDS.len() as u64) as usize]).collect::<Vec<_>>().join(" ")
}

/// Embedder that returns the clustered vector drawn for each query text,
/// so queries come from the same clusters as the chunks. Input: texts.
/// Output: their vectors, or an error for a text it was not given.
struct Lookup(HashMap<String, Vec<f32>>);

#[async_trait::async_trait]
impl Embedder for Lookup {
    fn dims(&self) -> usize {
        DIMS
    }

    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, ContextError> {
        texts.iter().map(|t| self.0.get(t).cloned().ok_or_else(|| ContextError::Embed(format!("no vector for {t:?}")))).collect()
    }
}

/// Value at percentile `p` of sorted `samples`, nearest rank.
/// Input: sorted samples and p in (0, 100]. Output: the sample.
/// Panics on an empty slice.
fn percentile(samples: &[Duration], p: usize) -> Duration {
    assert!(!samples.is_empty() && (1..=100).contains(&p), "percentile of nothing");
    samples[(samples.len() * p).div_ceil(100) - 1]
}

/// Fills a fresh file with `chunks` seeded chunks under `settings`, then
/// times `queries` hybrid searches. Input: counts and settings. Output:
/// p50 and p99, or the error of an unsupported kind.
async fn measure(chunks: usize, queries: usize, settings: VectorSettings) -> Result<(Duration, Duration), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let store = SqliteStore::open(dir.path().join("latency.db")).await?;
    let vectors = store.vectors("docs", settings.with_max_batch(BATCH)).await?;
    let mut drawn = common::clustered(SEED, chunks + queries, DIMS, CLUSTERS, SPREAD);
    let query_vectors = drawn.split_off(chunks);
    for start in (0..chunks).step_by(BATCH) {
        let records = (start..chunks.min(start + BATCH))
            .map(|i| {
                let text = seeded_text(i, 1);
                VectorRecord {
                    id: format!("c{i}"),
                    vector: std::mem::take(&mut drawn[i]),
                    text,
                    meta: BTreeMap::from([
                        (META_SCOPE.to_string(), SCOPE.to_string()),
                        (META_SOURCE.to_string(), format!("doc{}.md", i / 10)),
                        (META_ORIGIN.to_string(), "external".to_string()),
                    ]),
                }
            })
            .collect();
        vectors.upsert(records).await?;
    }
    let texts: Vec<String> = (0..queries).map(|q| format!("query{q} {}", seeded_text(q, 2).split(' ').take(4).collect::<Vec<_>>().join(" "))).collect();
    let embedder = Lookup(texts.iter().cloned().zip(query_vectors).collect());
    let cfg = HybridConfig::default().with_top_k(TOP_K);
    let mut samples = Vec::with_capacity(queries);
    for query in &texts {
        let started = Instant::now();
        let hits = hybrid_search(&vectors, &vectors, &embedder, query, &[SCOPE], &Filter::default(), &cfg).await?;
        samples.push(started.elapsed());
        assert!(hits.len() <= TOP_K, "more hits than top_k");
    }
    samples.sort_unstable();
    Ok((percentile(&samples, 50), percentile(&samples, 99)))
}

#[tokio::main]
async fn main() -> ExitCode {
    let mut args = std::env::args().skip(1).map(|a| a.parse::<usize>());
    let (chunks, queries) = match (args.next().unwrap_or(Ok(CHUNKS_DEFAULT)), args.next().unwrap_or(Ok(QUERIES_DEFAULT))) {
        (Ok(c), Ok(q)) if c > 0 && q > 0 => (c, q),
        _ => {
            eprintln!("usage: hybrid_latency [chunks] [queries]");
            return ExitCode::from(2);
        }
    };
    println!("chunks={chunks} dims={DIMS} queries={queries} top_k={TOP_K}");
    println!("| index | quant | p50 us | p99 us |");
    println!("|---|---|---|---|");
    let kinds = [("brute_force", IndexKind::BruteForce), ("hnsw", IndexKind::Hnsw(HnswParams::default()))];
    let quants = [("f32", Quantization::F32), ("i8", Quantization::I8)];
    let mut failed = false;
    for (kind_name, kind) in &kinds {
        for (quant_name, quant) in &quants {
            let settings = VectorSettings::new(DIMS).with_kind(kind.clone()).with_quantization(*quant);
            match measure(chunks, queries, settings).await {
                Ok((p50, p99)) => println!("| {kind_name} | {quant_name} | {} | {} |", p50.as_micros(), p99.as_micros()),
                Err(e) => {
                    failed = true;
                    println!("| {kind_name} | {quant_name} | not measured: {e} | |");
                }
            }
        }
    }
    if failed {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}
