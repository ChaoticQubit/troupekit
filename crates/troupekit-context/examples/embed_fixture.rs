//! Records the nDCG fixture: every chunk of the support corpus and every
//! labelled query, embedded once with `OnnxEmbedder`, so the nDCG test
//! runs offline. Run from the repo root with
//! `cargo run -p troupekit-context --features onnx --example embed_fixture`.

use serde_json::{json, Value};
use std::path::Path;
use troupekit_context::chunk::{chunk, ChunkConfig};
use troupekit_context::embed::embed_checked;
use troupekit_context::load::{walk, LoadSettings};
use troupekit_context::onnx::{OnnxEmbedder, ONNX_DIMS};

const CORPUS: &str = "../troupekit/examples/support/docs";
const LABELS: &str = "tests/fixtures/ndcg/labels.json";
const OUT: &str = "tests/fixtures/ndcg/embeddings.json";
const CHUNK: ChunkConfig = ChunkConfig { max_tokens: 200, overlap_tokens: 20 };
const MODEL: &str = "intfloat/multilingual-e5-small";
const LOCK: &str = "../../Cargo.lock";

/// Reads the locked `fastembed` version from the workspace `Cargo.lock`,
/// so the fixture names the exact library that made its vectors.
/// Input: the lock file text. Output: the version string.
fn fastembed_version(lock: &str) -> String {
    let mut lines = lock.lines();
    while let Some(line) = lines.next() {
        if line == "name = \"fastembed\"" {
            let version = lines.next().and_then(|v| v.strip_prefix("version = \"")).and_then(|v| v.strip_suffix('"'));
            return version.expect("fastembed entry has a version line").to_string();
        }
    }
    panic!("fastembed is not in Cargo.lock");
}

#[tokio::main]
async fn main() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let walked = walk(&root.join(CORPUS), &LoadSettings::default()).expect("corpus loads");
    let mut sources = Vec::new();
    let mut texts = Vec::new();
    for file in &walked.files {
        for (index, text) in chunk(&file.text, &CHUNK).expect("corpus chunks").into_iter().enumerate() {
            sources.push((file.source.clone(), index));
            texts.push(text);
        }
    }
    let labels: Value = serde_json::from_str(&std::fs::read_to_string(root.join(LABELS)).expect("labels read")).expect("labels parse");
    let queries: Vec<String> = labels["queries"].as_array().expect("queries list").iter().map(|q| q["query"].as_str().expect("query text").to_string()).collect();
    let fastembed = fastembed_version(&std::fs::read_to_string(root.join(LOCK)).expect("Cargo.lock reads"));
    let embedder = OnnxEmbedder::new(None).expect("model loads");
    let chunk_vectors = embed_checked(&embedder, &texts).await.expect("chunks embed");
    let query_vectors = embed_checked(&embedder, &queries).await.expect("queries embed");
    let chunks: Vec<Value> = sources
        .iter()
        .zip(&texts)
        .zip(&chunk_vectors)
        .map(|(((source, index), text), vector)| json!({ "source": source, "index": index, "text": text, "vector": vector }))
        .collect();
    let queries: Vec<Value> = queries.iter().zip(&query_vectors).map(|(text, vector)| json!({ "text": text, "vector": vector })).collect();
    let fixture = json!({
        "model": MODEL,
        "fastembed": fastembed,
        "dims": ONNX_DIMS,
        "chunk": { "max_tokens": CHUNK.max_tokens, "overlap_tokens": CHUNK.overlap_tokens },
        "chunks": chunks,
        "queries": queries,
    });
    std::fs::write(root.join(OUT), serde_json::to_string(&fixture).expect("fixture serializes") + "\n").expect("fixture writes");
    println!("wrote {} chunks and {} queries to {OUT}", texts.len(), query_vectors.len());
}
