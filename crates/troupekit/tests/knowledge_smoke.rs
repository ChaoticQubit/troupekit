use troupekit::context::embed::HashingEmbedder;
use troupekit::context::hybrid::{hybrid_search, HybridConfig, HybridMode};
use troupekit::core::store::Filter;
use troupekit::knowledge::{ingest_dir, IngestSettings};
use troupekit::{SqliteStore, VectorSettings};

const DIMS: usize = 64;

#[tokio::test]
async fn knowledge_smoke_open_ingest_search() {
    let dir = tempfile::tempdir().unwrap();
    let docs = dir.path().join("docs");
    std::fs::create_dir(&docs).unwrap();
    std::fs::write(docs.join("returns.md"), "# Returns\n\nUnused items can be returned within 30 days of delivery.").unwrap();
    let store = SqliteStore::open(dir.path().join("smoke.db")).await.unwrap();
    let vectors = store.vectors("docs", VectorSettings::new(DIMS)).await.unwrap();
    let embedder = HashingEmbedder::new(DIMS);
    let report = ingest_dir(&vectors, &embedder, &docs, &IngestSettings::default()).await.unwrap();
    assert_eq!((report.added, report.failed.len()), (1, 0));
    assert!(report.chunks >= 1);
    for mode in [HybridMode::Keyword, HybridMode::Vector, HybridMode::Hybrid] {
        let cfg = HybridConfig::default().with_mode(mode);
        let hits = hybrid_search(&vectors, &vectors, &embedder, "returned within 30 days", &["docs"], &Filter::default(), &cfg).await.unwrap();
        assert_eq!(hits.first().map(|h| h.record.meta["source"].as_str()), Some("returns.md"), "{mode:?}");
    }
}
