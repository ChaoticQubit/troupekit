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
