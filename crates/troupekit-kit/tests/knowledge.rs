use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use troupekit::knowledge::{ingest_dir, Embedder, IngestSettings, LoadSettings};
use troupekit_context::embed::HashingEmbedder;
use troupekit_context::ContextError;
use troupekit_store_sqlite::{SqliteStore, SqliteVectors, VectorSettings};

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

struct LateWriter {
    inner: HashingEmbedder,
    vectors: SqliteVectors,
    docs: PathBuf,
    fired: AtomicBool,
}

#[async_trait::async_trait]
impl Embedder for LateWriter {
    fn dims(&self) -> usize {
        self.inner.dims()
    }

    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, ContextError> {
        if !self.fired.swap(true, Ordering::SeqCst) {
            std::fs::write(self.docs.join("late.md"), "Written by a newer ingest.").unwrap();
            let newer = ingest_dir(
                &self.vectors,
                &self.inner,
                &self.docs,
                &IngestSettings::default(),
            )
            .await
            .unwrap();
            assert_eq!(newer.added, 1);
        }
        self.inner.embed(texts).await
    }
}

#[tokio::test]
async fn ingest_keeps_sources_a_newer_ingest_wrote_after_its_walk() {
    let dir = tempfile::tempdir().unwrap();
    let docs = dir.path().join("docs");
    std::fs::create_dir(&docs).unwrap();
    for name in ["a.md", "c.md", "late.md"] {
        std::fs::write(docs.join(name), format!("{name} v1")).unwrap();
    }
    let store = SqliteStore::open(dir.path().join("kb.db")).await.unwrap();
    let vectors = store
        .vectors("docs", VectorSettings::new(32))
        .await
        .unwrap();
    let plain = HashingEmbedder::new(32);
    let settings = IngestSettings::default();
    ingest_dir(&vectors, &plain, &docs, &settings)
        .await
        .unwrap();
    std::fs::remove_file(docs.join("late.md")).unwrap();
    std::fs::write(docs.join("a.md"), "a.md v2").unwrap();
    let older = LateWriter {
        inner: HashingEmbedder::new(32),
        vectors: vectors.clone(),
        docs: docs.clone(),
        fired: AtomicBool::new(false),
    };
    let report = ingest_dir(&vectors, &older, &docs, &settings)
        .await
        .unwrap();
    assert_eq!(report.removed, 0, "late.md is back on disk");
    let mut sources = vectors.sources("docs").await.unwrap();
    sources.sort();
    assert_eq!(sources, ["a.md", "c.md", "late.md"]);
}

#[tokio::test]
async fn files_excluded_by_the_filter_keep_their_sources() {
    let dir = tempfile::tempdir().unwrap();
    let docs = dir.path().join("docs");
    std::fs::create_dir(&docs).unwrap();
    std::fs::write(docs.join("a.txt"), "Plain notes.").unwrap();
    std::fs::write(docs.join("b.md"), "Markdown notes.").unwrap();
    std::fs::write(docs.join("c.md"), "More markdown.").unwrap();
    let store = SqliteStore::open(dir.path().join("kb.db")).await.unwrap();
    let vectors = store
        .vectors("docs", VectorSettings::new(32))
        .await
        .unwrap();
    let embedder = HashingEmbedder::new(32);
    let all = ingest_dir(&vectors, &embedder, &docs, &IngestSettings::default())
        .await
        .unwrap();
    assert_eq!(all.added, 3);
    let md_only =
        IngestSettings::default().with_load(LoadSettings::default().with_extensions(["md"]));
    let report = ingest_dir(&vectors, &embedder, &docs, &md_only)
        .await
        .unwrap();
    assert_eq!((report.removed, report.skipped), (0, 2));
    std::fs::remove_file(docs.join("a.txt")).unwrap();
    let report = ingest_dir(&vectors, &embedder, &docs, &md_only)
        .await
        .unwrap();
    assert_eq!(
        report.removed, 1,
        "a file gone from disk still leaves the index"
    );
    let mut sources = vectors.sources("docs").await.unwrap();
    sources.sort();
    assert_eq!(sources, ["b.md", "c.md"]);
}
