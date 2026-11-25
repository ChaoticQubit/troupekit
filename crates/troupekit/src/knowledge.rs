//! Ingest a folder into a named vector index in the store file. Unchanged
//! files are skipped by content hash, changed files are replaced whole and
//! a file that fails is recorded and the walk goes on.

use std::path::Path;
use troupekit_context::chunk::ChunkConfig;
use troupekit_context::embed::Embedder;
use troupekit_context::ingest::{document_records, Document, Origin};
use troupekit_context::load::{walk, LoadSettings};
use troupekit_context::text::stable_hash;
use troupekit_context::ContextError;
use troupekit_core::clock::{Clock, SystemClock};
use troupekit_store_sqlite::SqliteVectors;

const SCOPE_DEFAULT: &str = "docs";
const CHUNK_MAX_TOKENS_DEFAULT: usize = 200;
const CHUNK_OVERLAP_TOKENS_DEFAULT: usize = 20;

/// Settings for `ingest_dir`. Defaults: scope `docs`, chunks of 200
/// tokens with 20 of overlap, the safe folder walk of `LoadSettings` and
/// `Origin::External`, so folder text is fenced as untrusted in prompts.
/// A changed origin or chunk config re-ingests every file on the next run.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IngestSettings {
    scope: String,
    chunk: ChunkConfig,
    load: LoadSettings,
    origin: Origin,
}

impl Default for IngestSettings {
    fn default() -> Self {
        Self {
            scope: SCOPE_DEFAULT.to_string(),
            chunk: ChunkConfig { max_tokens: CHUNK_MAX_TOKENS_DEFAULT, overlap_tokens: CHUNK_OVERLAP_TOKENS_DEFAULT },
            load: LoadSettings::default(),
            origin: Origin::External,
        }
    }
}

impl IngestSettings {
    /// Sets the scope every chunk is stored under.
    /// Must not be blank; `ingest_dir` refuses a blank scope.
    /// Default `docs`.
    pub fn with_scope(mut self, scope: impl Into<String>) -> Self {
        self.scope = scope.into();
        self
    }

    /// Sets the chunk size and overlap, in estimated tokens.
    /// Checked by the chunker on the first file.
    /// Default 200 tokens with 20 of overlap.
    pub fn with_chunk(mut self, chunk: ChunkConfig) -> Self {
        self.chunk = chunk;
        self
    }

    /// Sets the folder walk bounds and filters.
    /// A walk cap error stops the ingest before any write.
    /// Default `LoadSettings::default()`.
    pub fn with_load(mut self, load: LoadSettings) -> Self {
        self.load = load;
        self
    }

    /// Sets the origin stored on every chunk.
    /// `Internal` text is put in prompts unfenced; use it only for text
    /// the app wrote itself. Default `Origin::External`.
    pub fn with_origin(mut self, origin: Origin) -> Self {
        self.origin = origin;
        self
    }

    /// Returns the scope every chunk is stored under.
    /// It is the `scope` meta value of each chunk.
    /// Default `docs`.
    pub fn scope(&self) -> &str {
        &self.scope
    }

    /// Returns the chunk size and overlap.
    /// In estimated tokens of four chars.
    /// Default 200 and 20.
    pub fn chunk(&self) -> ChunkConfig {
        self.chunk
    }

    /// Returns the folder walk settings.
    /// Their caps bound one ingest.
    /// Default `LoadSettings::default()`.
    pub fn load(&self) -> &LoadSettings {
        &self.load
    }

    /// Returns the origin stored on every chunk.
    /// External text is fenced as untrusted.
    /// Default `Origin::External`.
    pub fn origin(&self) -> Origin {
        self.origin
    }
}

/// What one `ingest_dir` call did: files added, replaced and skipped as
/// unchanged, files the walk passed over (`unsupported`, `symlinks`),
/// failed files with their error, and the chunks written.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IngestReport {
    pub added: usize,
    pub replaced: usize,
    pub skipped: usize,
    pub unsupported: usize,
    pub symlinks: usize,
    pub failed: Vec<(String, String)>,
    pub chunks: usize,
}

/// Ingests every loadable file under `dir` into `vectors`. A file whose
/// hash matches its last successful ingest is skipped without embedding;
/// any other is chunked, embedded and swapped in with `replace_source`.
/// Output: the report, or an error for a walk cap, a blank scope or a
/// store that cannot record a failed file.
pub async fn ingest_dir(vectors: &SqliteVectors, embedder: &dyn Embedder, dir: &Path, settings: &IngestSettings) -> Result<IngestReport, ContextError> {
    if settings.scope.trim().is_empty() {
        return Err(ContextError::Invalid("scope must not be empty".into()));
    }
    let walked = walk(dir, &settings.load)?;
    let mut report = IngestReport {
        added: 0,
        replaced: 0,
        skipped: 0,
        unsupported: walked.unsupported,
        symlinks: walked.symlinks,
        failed: Vec::new(),
        chunks: 0,
    };
    let at_ms = SystemClock.now_ms();
    for file in walked.files {
        let ingest_hash = ingest_hash(&file.content_hash, settings);
        let previous = vectors.source_hash(&settings.scope, &file.source).await?;
        if previous.as_deref() == Some(ingest_hash.as_str()) {
            report.skipped += 1;
            continue;
        }
        let doc = Document {
            scope: settings.scope.clone(),
            source: file.source.clone(),
            text: file.text,
            at_ms,
            origin: settings.origin,
            meta: Default::default(),
        };
        let outcome = match document_records(embedder, &doc, &settings.chunk).await {
            Ok(records) => vectors
                .replace_source(&settings.scope, &file.source, records, &ingest_hash)
                .await
                .map_err(ContextError::from),
            Err(e) => Err(e),
        };
        match outcome {
            Ok(written) => {
                report.chunks += written;
                if previous.is_some() {
                    report.replaced += 1;
                } else {
                    report.added += 1;
                }
            }
            Err(e) => {
                let error = e.to_string();
                vectors.record_failed_source(&settings.scope, &file.source, &error).await?;
                report.failed.push((file.source, error));
            }
        }
    }
    let handled = report.added + report.replaced + report.skipped + report.failed.len();
    assert!(handled <= settings.load.max_files(), "more files handled than the walk allows");
    Ok(report)
}

/// Hash stored per source: the file's content hash together with the
/// origin and chunk config its chunks were made with, so a settings
/// change no longer matches and the file is ingested again.
/// Input: the content hash and the settings. Output: 16 hex chars.
fn ingest_hash(content_hash: &str, settings: &IngestSettings) -> String {
    let max = settings.chunk.max_tokens.to_string();
    let overlap = settings.chunk.overlap_tokens.to_string();
    let hash = format!("{:016x}", stable_hash(&[content_hash, settings.origin.as_str(), &max, &overlap]));
    assert_eq!(hash.len(), 16, "ingest hash is 16 hex chars");
    hash
}
