use troupekit::knowledge::{ingest_dir, IngestSettings};
use troupekit_context::embed::HashingEmbedder;
use troupekit_store_sqlite::{SqliteStore, VectorSettings};

#[tokio::test]
async fn ingest_dir_lives_in_the_kit_and_skips_unchanged_files() {
    let dir = tempfile::tempdir().unwrap();
    let docs = dir.path().join("docs");
    std::fs::create_dir(&docs).unwrap();
    std::fs::write(docs.join("refunds.md"), "Refunds take five days.").unwrap();
    let store = SqliteStore::open(dir.path().join("kb.db")).await.unwrap();
    let vectors = store
        .vectors("docs", VectorSettings::new(32))
        .await
        .unwrap();
    let embedder = HashingEmbedder::new(32);
    let settings = IngestSettings::default();
    let first = ingest_dir(&vectors, &embedder, &docs, &settings)
        .await
        .unwrap();
    assert_eq!((first.added, first.skipped), (1, 0));
    let again = ingest_dir(&vectors, &embedder, &docs, &settings)
        .await
        .unwrap();
    assert_eq!((again.added, again.skipped), (0, 1));
}
