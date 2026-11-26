use crate::chunk::{chunk, ChunkConfig};
use crate::embed::{embed_checked, Embedder};
use crate::error::ContextError;
use crate::text::stable_hash;
use std::collections::BTreeMap;
use troupekit_core::store::{VectorRecord, VectorStore};
use troupekit_trace::{content, trace_span, Trace};

pub const META_SCOPE: &str = "scope";
pub const META_SOURCE: &str = "source";
pub const META_CHUNK: &str = "chunk";
pub const META_AT: &str = "at_ms";
pub const META_ORIGIN: &str = "origin";

const RESERVED: [&str; 5] = [META_SCOPE, META_SOURCE, META_CHUNK, META_AT, META_ORIGIN];

/// Whether text was written by the user or the app, or came from outside.
/// External text, such as a fetched web page, is fenced as untrusted
/// when it is put into a prompt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Origin {
    Internal,
    External,
}

impl Origin {
    /// Returns the meta value for this origin: `internal` or `external`.
    /// Stored under `META_ORIGIN` on every ingested chunk.
    /// The value is what `Origin::of` reads back.
    pub fn as_str(self) -> &'static str {
        match self {
            Origin::Internal => "internal",
            Origin::External => "external",
        }
    }

    /// Reads the origin from record `meta`.
    /// Only the exact value `internal` is internal; a missing or unknown
    /// value is external, so unlabelled text is never trusted.
    pub fn of(meta: &BTreeMap<String, String>) -> Origin {
        match meta.get(META_ORIGIN).map(String::as_str) {
            Some("internal") => Origin::Internal,
            _ => Origin::External,
        }
    }
}

/// A text to ingest into one scope, such as `goal:42`, from `source`.
/// `at_ms` is when the source was written or fetched, in Unix milliseconds.
/// `meta` holds extra pairs, such as a tenant id, copied onto every chunk;
/// it may not use a reserved `META_*` key.
#[derive(Debug, Clone, PartialEq)]
pub struct Document {
    pub scope: String,
    pub source: String,
    pub text: String,
    pub at_ms: i64,
    pub origin: Origin,
    pub meta: BTreeMap<String, String>,
}

/// Builds the stable id of chunk `index` of `doc` from its scope, source and
/// extra meta, so the same source under two tenants never shares an id.
/// Re-ingesting an unchanged document reuses the ids, so an upsert replaces
/// rather than duplicates. Returns 16 hex chars.
pub fn chunk_id(doc: &Document, index: usize) -> String {
    let index = index.to_string();
    let mut parts: Vec<&str> = vec![&doc.scope, &doc.source, &index];
    for (k, v) in &doc.meta {
        parts.push(k);
        parts.push(v);
    }
    format!("{:016x}", stable_hash(&parts))
}

/// Chunks `doc`, embeds the chunks in one call and upserts them into `vectors`.
/// Each record carries the reserved meta keys plus `doc.meta`.
/// Returns the chunk count, or `ContextError::Invalid` for a blank scope or
/// source or a reserved key in `doc.meta`; blank text writes nothing.
pub async fn ingest(vectors: &dyn VectorStore, embedder: &dyn Embedder, doc: &Document, cfg: &ChunkConfig) -> Result<usize, ContextError> {
    let mut span = trace_span!(Trace::Knowledge, "knowledge.ingest", text_bytes = doc.text.len(), source = content(doc.source.as_str()));
    let result = span.instrument(ingest_records(vectors, embedder, doc, cfg)).await;
    match &result {
        Ok(chunks) => span.record("chunks", *chunks),
        Err(e) => span.fail(e.to_string()),
    }
    result
}

async fn ingest_records(vectors: &dyn VectorStore, embedder: &dyn Embedder, doc: &Document, cfg: &ChunkConfig) -> Result<usize, ContextError> {
    let records = document_records(embedder, doc, cfg).await?;
    if records.is_empty() {
        return Ok(0);
    }
    let count = records.len();
    vectors.upsert(records).await?;
    Ok(count)
}

/// Chunks and embeds `doc` into the records `ingest` would write, with the
/// same ids and meta keys, without writing them. Returns the records, none
/// for blank text, or `ContextError::Invalid` for a blank scope or source
/// or a reserved key in `doc.meta`.
pub async fn document_records(embedder: &dyn Embedder, doc: &Document, cfg: &ChunkConfig) -> Result<Vec<VectorRecord>, ContextError> {
    if doc.scope.trim().is_empty() {
        return Err(ContextError::Invalid("scope must not be empty".into()));
    }
    if doc.source.trim().is_empty() {
        return Err(ContextError::Invalid("source must not be empty".into()));
    }
    if let Some(key) = doc.meta.keys().find(|k| RESERVED.contains(&k.as_str())) {
        return Err(ContextError::Invalid(format!("meta key {key} is reserved")));
    }
    let chunks = chunk(&doc.text, cfg)?;
    if chunks.is_empty() {
        return Ok(Vec::new());
    }
    let embeddings = embed_checked(embedder, &chunks).await?;
    let count = chunks.len();
    let records: Vec<VectorRecord> = chunks
        .into_iter()
        .zip(embeddings)
        .enumerate()
        .map(|(index, (text, vector))| {
            let mut meta = doc.meta.clone();
            meta.insert(META_SCOPE.into(), doc.scope.clone());
            meta.insert(META_SOURCE.into(), doc.source.clone());
            meta.insert(META_CHUNK.into(), index.to_string());
            meta.insert(META_AT.into(), doc.at_ms.to_string());
            meta.insert(META_ORIGIN.into(), doc.origin.as_str().into());
            VectorRecord { id: chunk_id(doc, index), vector, text, meta }
        })
        .collect();
    assert_eq!(records.len(), count, "one record per chunk");
    Ok(records)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embed::HashingEmbedder;
    use troupekit_core::inmem::InMemVectors;
    use troupekit_core::store::{Filter, Scored};

    const DIMS: usize = 32;

    fn doc(scope: &str, source: &str, text: &str) -> Document {
        Document {
            scope: scope.into(),
            source: source.into(),
            text: text.into(),
            at_ms: 1_000,
            origin: Origin::Internal,
            meta: BTreeMap::from([("user".to_string(), "7".to_string())]),
        }
    }

    fn small() -> ChunkConfig {
        ChunkConfig { max_tokens: 8, overlap_tokens: 2 }
    }

    async fn all(store: &InMemVectors, filter: &[(&str, &str)]) -> Vec<Scored> {
        let f = Filter(filter.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect());
        store.search(&[1.0; DIMS], 1000, &f).await.unwrap()
    }

    const TEXT: &str = "Eat protein at breakfast.\n\nWalk ten thousand steps.\n\nSleep eight hours each night.";

    #[tokio::test]
    async fn ingest_writes_chunks_with_meta() {
        let store = InMemVectors::new(DIMS);
        let n = ingest(&store, &HashingEmbedder::new(DIMS), &doc("goal:1", "notes.md", TEXT), &small()).await.unwrap();
        assert!(n >= 3);
        let hits = all(&store, &[(META_SCOPE, "goal:1")]).await;
        assert_eq!(hits.len(), n);
        let first = hits.iter().find(|h| h.meta[META_CHUNK] == "0").unwrap();
        assert_eq!(first.id, chunk_id(&doc("goal:1", "notes.md", TEXT), 0));
        assert_eq!(first.meta[META_SOURCE], "notes.md");
        assert_eq!(first.meta[META_AT], "1000");
        assert_eq!(first.meta[META_ORIGIN], "internal");
        assert_eq!(first.meta["user"], "7");
    }

    #[tokio::test]
    async fn reingest_is_idempotent() {
        let store = InMemVectors::new(DIMS);
        let e = HashingEmbedder::new(DIMS);
        let n = ingest(&store, &e, &doc("goal:1", "notes.md", TEXT), &small()).await.unwrap();
        ingest(&store, &e, &doc("goal:1", "notes.md", TEXT), &small()).await.unwrap();
        assert_eq!(all(&store, &[]).await.len(), n);
    }

    #[tokio::test]
    async fn same_source_two_scopes_keeps_both() {
        let store = InMemVectors::new(DIMS);
        let e = HashingEmbedder::new(DIMS);
        let n = ingest(&store, &e, &doc("goal:1", "notes.md", TEXT), &small()).await.unwrap();
        ingest(&store, &e, &doc("goal:2", "notes.md", TEXT), &small()).await.unwrap();
        assert_eq!(all(&store, &[(META_SCOPE, "goal:1")]).await.len(), n);
        assert_eq!(all(&store, &[(META_SCOPE, "goal:2")]).await.len(), n);
    }

    #[tokio::test]
    async fn same_source_two_tenants_keeps_both() {
        let store = InMemVectors::new(DIMS);
        let e = HashingEmbedder::new(DIMS);
        let mut theirs = doc("goal:1", "notes.md", TEXT);
        theirs.meta.insert("user".into(), "8".into());
        let n = ingest(&store, &e, &doc("goal:1", "notes.md", TEXT), &small()).await.unwrap();
        ingest(&store, &e, &theirs, &small()).await.unwrap();
        assert_eq!(all(&store, &[("user", "7")]).await.len(), n);
        assert_eq!(all(&store, &[("user", "8")]).await.len(), n);
    }

    #[tokio::test]
    async fn rejects_bad_document() {
        let store = InMemVectors::new(DIMS);
        let e = HashingEmbedder::new(DIMS);
        for d in [doc(" ", "notes.md", TEXT), doc("goal:1", "", TEXT)] {
            assert!(matches!(ingest(&store, &e, &d, &small()).await, Err(ContextError::Invalid(_))));
        }
        let mut d = doc("goal:1", "notes.md", TEXT);
        d.meta.insert(META_SCOPE.into(), "goal:9".into());
        assert!(matches!(ingest(&store, &e, &d, &small()).await, Err(ContextError::Invalid(_))));
        assert!(all(&store, &[]).await.is_empty());
    }

    #[tokio::test]
    async fn empty_text_writes_nothing() {
        let store = InMemVectors::new(DIMS);
        assert_eq!(ingest(&store, &HashingEmbedder::new(DIMS), &doc("goal:1", "a", "  "), &small()).await.unwrap(), 0);
    }

    #[test]
    fn origin_defaults_to_external() {
        let mut meta = BTreeMap::new();
        assert_eq!(Origin::of(&meta), Origin::External);
        meta.insert(META_ORIGIN.to_string(), "internal".to_string());
        assert_eq!(Origin::of(&meta), Origin::Internal);
        meta.insert(META_ORIGIN.to_string(), "weird".to_string());
        assert_eq!(Origin::of(&meta), Origin::External);
    }
}
