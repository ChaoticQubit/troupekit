//! Hybrid query latency over one SQLite file: seeded chunks with seeded
//! 384-dimension vectors, then timed `hybrid_search` calls with `top_k` 8,
//! for brute force and HNSW in f32 and i8. Run with `cargo run --release
//! -p troupekit --features knowledge --example hybrid_latency -- [chunks] [queries]`.

use std::collections::BTreeMap;
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

/// Seeded vector of `DIMS` values in [-1, 1) for `text`, seeded by the
/// FNV-1a hash of the text. Input: the text. Output: the vector, the same
/// for the same text on every run.
fn seeded_vector(text: &str) -> Vec<f32> {
    let mut state = text.bytes().fold(0xcbf2_9ce4_8422_2325_u64, |h, b| (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3));
    (0..DIMS).map(|_| (next(&mut state) >> 40) as f32 / (1u64 << 23) as f32 - 1.0).collect()
}

/// Seeded text of 24 words for chunk or query `index` under `salt`.
/// Input: index and salt. Output: the text.
/// Words come from a fixed support vocabulary so BM25 has matches.
fn seeded_text(index: usize, salt: u64) -> String {
    let mut state = (index as u64) ^ salt;
    (0..24).map(|_| WORDS[(next(&mut state) % WORDS.len() as u64) as usize]).collect::<Vec<_>>().join(" ")
}

/// Embedder returning `seeded_vector` of each text, so queries and
/// chunks share one generator. Input: texts. Output: their vectors.
struct Seeded;

#[async_trait::async_trait]
impl Embedder for Seeded {
    fn dims(&self) -> usize {
        DIMS
    }

    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, ContextError> {
        Ok(texts.iter().map(|t| seeded_vector(t)).collect())
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
    for start in (0..chunks).step_by(BATCH) {
        let records = (start..chunks.min(start + BATCH))
            .map(|i| {
                let text = seeded_text(i, 1);
                VectorRecord {
                    id: format!("c{i}"),
                    vector: seeded_vector(&text),
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
    let cfg = HybridConfig::default().with_top_k(TOP_K);
    let mut samples = Vec::with_capacity(queries);
    for q in 0..queries {
        let query: String = seeded_text(q, 2).split(' ').take(4).collect::<Vec<_>>().join(" ");
        let started = Instant::now();
        let hits = hybrid_search(&vectors, &vectors, &Seeded, &query, &[SCOPE], &Filter::default(), &cfg).await?;
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
    println!("chunks {chunks}, queries {queries}, top_k {TOP_K}, dims {DIMS}");
    let kinds = [("brute_force", IndexKind::BruteForce), ("hnsw", IndexKind::Hnsw(HnswParams::default()))];
    let quants = [("f32", Quantization::F32), ("i8", Quantization::I8)];
    let mut failed = false;
    for (kind_name, kind) in &kinds {
        for (quant_name, quant) in &quants {
            let settings = VectorSettings::new(DIMS).with_kind(kind.clone()).with_quantization(*quant);
            match measure(chunks, queries, settings).await {
                Ok((p50, p99)) => println!("{kind_name} {quant_name}: p50 {} us, p99 {} us", p50.as_micros(), p99.as_micros()),
                Err(e) => {
                    failed = true;
                    println!("{kind_name} {quant_name}: not measured: {e}");
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
