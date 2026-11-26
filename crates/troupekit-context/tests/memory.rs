use std::collections::BTreeMap;
use std::sync::Arc;
use troupekit_context::block::ContextBuilder;
use troupekit_context::chunk::ChunkConfig;
use troupekit_context::embed::HashingEmbedder;
use troupekit_context::facts::Fact;
use troupekit_context::ingest::{Document, Origin};
use troupekit_context::memory::Memory;
use troupekit_context::retrieve::RetrievalConfig;
use troupekit_context::ContextError;
use troupekit_core::clock::FixedClock;
use troupekit_core::ids::RunId;
use troupekit_core::inmem::{InMemMemory, InMemVectors};
use troupekit_core::message::{Message, Role};
use troupekit_core::store::Filter;

const NOW: i64 = 1_700_000_000_000;

fn memory(store_dims: usize) -> Result<Memory, ContextError> {
    Memory::new(Arc::new(InMemMemory::default()), Arc::new(InMemVectors::new(store_dims)), Arc::new(HashingEmbedder::new(64)), Arc::new(FixedClock::new(NOW)))
}

fn doc(scope: &str, source: &str, text: &str, origin: Origin) -> Document {
    Document { scope: scope.into(), source: source.into(), text: text.into(), at_ms: NOW, origin, meta: BTreeMap::from([("user".into(), "7".into())]) }
}

#[tokio::test]
async fn memory_scopes_end_to_end() {
    let m = memory(64).unwrap();
    let run = RunId::new();
    assert_eq!(m.append_history(run, 0, vec![Message::text(Role::User, "plan my week")]).await.unwrap(), 1);
    assert_eq!(m.history(run).await.unwrap()[0].text_content(), "plan my week");
    assert!(m.history(RunId::new()).await.unwrap().is_empty());

    m.facts().put("user:7", Fact { key: "diet".into(), value: "vegetarian".into(), source: "chat".into(), at_ms: NOW, confidence: 1.0 }).await.unwrap();
    assert_eq!(m.facts().list("user:7").await.unwrap()[0].value, "vegetarian");

    let cfg = ChunkConfig { max_tokens: 64, overlap_tokens: 8 };
    m.ingest(&doc("goal:1", "notes.md", "Eat a protein breakfast every morning.", Origin::Internal), &cfg).await.unwrap();
    m.ingest(&doc("goal:2", "https://example.com/abs", "Core training builds visible abs with protein.", Origin::External), &cfg).await.unwrap();
    m.ingest(&doc("goal:3", "other.md", "Protein for an unrelated goal.", Origin::Internal), &cfg).await.unwrap();

    let builder = ContextBuilder { budget_tokens: 500, retrieval: RetrievalConfig::default() };
    let user = Filter(BTreeMap::from([("user".into(), "7".into())]));
    let block = builder.build(&m, "protein abs", &["goal:1", "goal:2"], &user).await.unwrap();
    let mut sources = block.sources.clone();
    sources.sort();
    assert_eq!(sources, vec!["https://example.com/abs", "notes.md"]);
    assert_eq!(block.text.matches("<untrusted-content>").count(), 1);
    assert!(block.tokens <= 500);

    let other = Filter(BTreeMap::from([("user".into(), "8".into())]));
    assert!(builder.build(&m, "protein abs", &["goal:1", "goal:2"], &other).await.unwrap().text.is_empty());
}

#[test]
fn mismatched_dims_are_rejected() {
    assert!(matches!(memory(32), Err(ContextError::Invalid(_))));
}

struct LeakyEmbedder;

#[async_trait::async_trait]
impl troupekit_context::embed::Embedder for LeakyEmbedder {
    fn dims(&self) -> usize {
        64
    }

    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, ContextError> {
        Err(ContextError::Embed(format!("model refused: {}", texts.join(" "))))
    }
}

#[tokio::test]
async fn failed_ingest_keeps_error_text_out_of_the_trace() {
    use troupekit_context::ingest::ingest;
    let vectors = InMemVectors::new(64);
    let d = doc("s", "notes.md", "patient zebra_record_77 history", Origin::Internal);
    for keep in [false, true] {
        let sink = Arc::new(troupekit_trace::MemorySink::default());
        let tracer = troupekit_trace::Tracer::builder().content(keep).sink(sink.clone()).build().unwrap();
        let result = tracer.in_scope(ingest(&vectors, &LeakyEmbedder, &d, &ChunkConfig { max_tokens: 64, overlap_tokens: 8 })).await;
        drop(tracer);
        assert!(matches!(result, Err(ContextError::Embed(_))), "{result:?}");
        let records: Vec<serde_json::Value> = sink.lines().iter().map(|l| serde_json::from_str(l).unwrap()).collect();
        let end = records.iter().find(|r| r["t"] == "end").unwrap();
        assert_eq!(end["ok"], false);
        assert_eq!(end["err"], "knowledge ingest failed");
        let leaked = sink.lines().iter().any(|l| l.contains("zebra_record_77"));
        assert_eq!(leaked, keep, "error text appears only with content on");
    }
}
