use serde_json::Value;
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use troupekit_context::chunk::{chunk, ChunkConfig};
use troupekit_context::embed::Embedder;
use troupekit_context::hybrid::{hybrid_search, HybridConfig, HybridMode};
use troupekit_context::ingest::{document_records, Document, Origin};
use troupekit_context::load::{walk, LoadSettings};
use troupekit_context::ContextError;
use troupekit_core::store::Filter;
use troupekit_store_sqlite::{SqliteStore, VectorSettings};

const SCOPE: &str = "docs";
const CHUNK: ChunkConfig = ChunkConfig { max_tokens: 200, overlap_tokens: 20 };
const CUTOFF: usize = 10;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn corpus() -> PathBuf {
    root().join("../troupekit/examples/support/docs")
}

fn read_json(relative: &str) -> Value {
    let path = root().join(relative);
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!("{}: {e}; record it with `cargo run -p troupekit-context --features onnx --example embed_fixture`", path.display())
    });
    serde_json::from_str(&text).unwrap()
}

fn labels() -> Value {
    read_json("tests/fixtures/ndcg/labels.json")
}

fn fixture() -> Value {
    read_json("tests/fixtures/ndcg/embeddings.json")
}

fn vector(v: &Value) -> Vec<f32> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect()
}

/// Test double for the network-free case: returns the recorded vector of
/// a known text and fails on any other, so a corpus or chunker change
/// can never fall back to a made-up vector.
struct Recorded {
    dims: usize,
    vectors: HashMap<String, Vec<f32>>,
}

#[async_trait::async_trait]
impl Embedder for Recorded {
    fn dims(&self) -> usize {
        self.dims
    }

    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, ContextError> {
        texts.iter().map(|t| self.vectors.get(t).cloned().ok_or_else(|| ContextError::Embed(format!("no recorded vector for {t:?}")))).collect()
    }
}

fn recorded(fixture: &Value) -> Recorded {
    let mut vectors = HashMap::new();
    for item in fixture["chunks"].as_array().unwrap().iter().chain(fixture["queries"].as_array().unwrap()) {
        vectors.insert(item["text"].as_str().unwrap().to_string(), vector(&item["vector"]));
    }
    Recorded { dims: fixture["dims"].as_u64().unwrap() as usize, vectors }
}

fn ndcg(ranked: &[&str], relevance: &BTreeMap<String, u64>, cutoff: usize) -> f64 {
    let gain = |grade: u64| 2f64.powi(grade as i32) - 1.0;
    let dcg: f64 = ranked.iter().take(cutoff).enumerate().map(|(i, s)| gain(*relevance.get(*s).unwrap_or(&0)) / (i as f64 + 2.0).log2()).sum();
    let mut ideal: Vec<u64> = relevance.values().copied().collect();
    ideal.sort_unstable_by(|a, b| b.cmp(a));
    let idcg: f64 = ideal.iter().take(cutoff).enumerate().map(|(i, g)| gain(*g) / (i as f64 + 2.0).log2()).sum();
    if idcg == 0.0 {
        0.0
    } else {
        dcg / idcg
    }
}

#[test]
fn ndcg_of_a_perfect_ranking_is_one() {
    let rel: BTreeMap<String, u64> = [("a".to_string(), 2), ("b".to_string(), 1)].into();
    assert!((ndcg(&["a", "b", "c"], &rel, CUTOFF) - 1.0).abs() < 1e-12);
    assert!(ndcg(&["b", "a"], &rel, CUTOFF) < 1.0);
    assert_eq!(ndcg(&["c", "d"], &rel, CUTOFF), 0.0);
    assert_eq!(ndcg(&["a"], &BTreeMap::new(), CUTOFF), 0.0);
}

#[test]
fn fixture_matches_corpus_and_chunk_config() {
    let fixture = fixture();
    assert_eq!(fixture["model"], "intfloat/multilingual-e5-small");
    assert!(fixture["fastembed"].as_str().is_some_and(|v| !v.is_empty()), "fixture names its fastembed version");
    assert_eq!(fixture["chunk"]["max_tokens"].as_u64(), Some(CHUNK.max_tokens as u64));
    assert_eq!(fixture["chunk"]["overlap_tokens"].as_u64(), Some(CHUNK.overlap_tokens as u64));
    let walked = walk(&corpus(), &LoadSettings::default()).unwrap();
    let mut want = Vec::new();
    for file in &walked.files {
        for (index, text) in chunk(&file.text, &CHUNK).unwrap().into_iter().enumerate() {
            want.push((file.source.clone(), index as u64, text));
        }
    }
    let got: Vec<(String, u64, String)> = fixture["chunks"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| (c["source"].as_str().unwrap().to_string(), c["index"].as_u64().unwrap(), c["text"].as_str().unwrap().to_string()))
        .collect();
    assert_eq!(got, want, "corpus or chunker changed; re-record the fixture");
    let labels = labels();
    let labelled: Vec<&str> = labels["queries"].as_array().unwrap().iter().map(|q| q["query"].as_str().unwrap()).collect();
    let recorded_queries: Vec<String> = fixture["queries"].as_array().unwrap().iter().map(|q| q["text"].as_str().unwrap().to_string()).collect();
    assert_eq!(recorded_queries, labelled);
    let dims = fixture["dims"].as_u64().unwrap() as usize;
    assert!(fixture["chunks"].as_array().unwrap().iter().all(|c| c["vector"].as_array().unwrap().len() == dims));
}

#[tokio::test]
async fn hybrid_beats_keyword_and_vector_on_ndcg_at_10() {
    let fixture = fixture();
    let labels = labels();
    let embedder = recorded(&fixture);
    let dir = tempfile::tempdir().unwrap();
    let store = SqliteStore::open(dir.path().join("ndcg.db")).await.unwrap();
    let vectors = store.vectors("docs", VectorSettings::new(embedder.dims())).await.unwrap();
    for file in walk(&corpus(), &LoadSettings::default()).unwrap().files {
        let doc = Document { scope: SCOPE.into(), source: file.source.clone(), text: file.text, at_ms: 0, origin: Origin::External, meta: BTreeMap::new() };
        let records = document_records(&embedder, &doc, &CHUNK).await.unwrap();
        vectors.replace_source(SCOPE, &file.source, records, &file.content_hash).await.unwrap();
    }
    let mut means = BTreeMap::new();
    for (name, mode) in [("hybrid", HybridMode::Hybrid), ("keyword", HybridMode::Keyword), ("vector", HybridMode::Vector)] {
        let cfg = HybridConfig::default().with_mode(mode).with_top_k(30).with_candidates(50);
        let mut total = 0.0;
        let queries = labels["queries"].as_array().unwrap();
        for q in queries {
            let relevance: BTreeMap<String, u64> = q["relevance"].as_object().unwrap().iter().map(|(k, v)| (k.clone(), v.as_u64().unwrap())).collect();
            let hits = hybrid_search(&vectors, &vectors, &embedder, q["query"].as_str().unwrap(), &[SCOPE], &Filter::default(), &cfg).await.unwrap();
            let mut ranked: Vec<&str> = Vec::new();
            for hit in &hits {
                let source = hit.record.meta["source"].as_str();
                if !ranked.contains(&source) {
                    ranked.push(source);
                }
            }
            total += ndcg(&ranked, &relevance, CUTOFF);
        }
        means.insert(name, total / queries.len() as f64);
    }
    eprintln!("mean nDCG@10 over {} queries: {means:?}", labels["queries"].as_array().unwrap().len());
    assert!(means["hybrid"] > means["keyword"], "{means:?}");
    assert!(means["hybrid"] > means["vector"], "{means:?}");
    assert!(Path::new(&corpus()).is_dir());
}
