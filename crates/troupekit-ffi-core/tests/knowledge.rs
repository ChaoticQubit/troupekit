#![cfg(feature = "knowledge")]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use troupekit_ffi_core::{
    AgentBuilder, ErrorCode, HostEmbedder, HostError, HostModel, HostModelCaps, HostReply,
    HostRequest, KnowledgeConfig, MemoryConfig, TokenSink,
};

struct Quiet;

impl HostModel for Quiet {
    fn id(&self) -> String {
        "host/quiet".into()
    }

    fn capabilities(&self) -> HostModelCaps {
        HostModelCaps::default()
    }

    fn generate(
        &self,
        _request: HostRequest,
        _sink: Arc<TokenSink>,
    ) -> Result<HostReply, HostError> {
        let mut reply = HostReply::default();
        reply.text = r#"{"answer": "ok"}"#.into();
        Ok(reply)
    }
}

#[derive(Default)]
struct Counting {
    texts: AtomicUsize,
}

impl HostEmbedder for Counting {
    fn dims(&self) -> u32 {
        4
    }

    fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, HostError> {
        self.texts.fetch_add(texts.len(), Ordering::Relaxed);
        Ok(texts
            .iter()
            .map(|t| vec![t.len() as f32, 1.0, 0.0, 0.0])
            .collect())
    }
}

fn helper() -> Arc<AgentBuilder> {
    AgentBuilder::new("helper".into())
        .instructions("Answer from the docs.".into())
        .host_model(Arc::new(Quiet))
}

#[tokio::test]
async fn knowledge_ingests_with_the_host_embedder() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(
        dir.path().join("faq.md"),
        "# Returns\n\nReturns take 30 days.\n",
    )
    .unwrap();
    let embedder = Arc::new(Counting::default());
    let mut config = KnowledgeConfig::new(dir.path().to_string_lossy());
    config.store = Some(dir.path().join("index.db").to_string_lossy().into_owned());
    helper()
        .knowledge(config, Some(embedder.clone()))
        .build()
        .await
        .unwrap();
    assert!(embedder.texts.load(Ordering::Relaxed) > 0);
}

#[tokio::test]
async fn knowledge_on_a_missing_dir_fails_the_build() {
    let config = KnowledgeConfig::new("/no/such/troupekit/dir");
    let err = helper()
        .knowledge(config, Some(Arc::new(Counting::default())))
        .build()
        .await
        .unwrap_err();
    assert_ne!(err.code, ErrorCode::Unsupported);
}

#[tokio::test]
async fn memory_builds_with_a_sqlite_store() {
    let dir = tempfile::tempdir().unwrap();
    let mut config = MemoryConfig::new("ada", "s1");
    config.sqlite = Some(dir.path().join("memory.db").to_string_lossy().into_owned());
    config.preamble = false;
    helper().memory(config).build().await.unwrap();
}

#[tokio::test]
async fn memory_with_empty_user_is_refused() {
    let err = helper()
        .memory(MemoryConfig::new("", "s1"))
        .build()
        .await
        .unwrap_err();
    assert_ne!(err.code, ErrorCode::Unsupported);
}
