//! Ingest a folder into a named vector index in the store file. Unchanged
//! files are skipped by content hash, changed files are replaced whole and
//! a file that fails is recorded and the walk goes on.

use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;
pub use troupekit_context::chunk::ChunkConfig;
pub use troupekit_context::embed::Embedder;
use troupekit_context::ingest::{document_records, Document, Origin};
use troupekit_context::load::walk;
pub use troupekit_context::load::LoadSettings;
use troupekit_context::ContextError;
use troupekit_core::clock::{Clock, SystemClock};
use troupekit_store_sqlite::SqliteVectors;

const SCOPE_DEFAULT: &str = "docs";
const CHUNK_MAX_TOKENS_DEFAULT: usize = 200;
const CHUNK_OVERLAP_TOKENS_DEFAULT: usize = 20;

/// Settings for `ingest_dir`. Defaults: scope `docs`, chunks of 200
/// tokens with 20 of overlap, the safe folder walk of `LoadSettings` and
/// `Origin::External`, so folder text is fenced as untrusted in prompts.
/// Built with `Default` and the `with_*` setters.
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
            chunk: ChunkConfig {
                max_tokens: CHUNK_MAX_TOKENS_DEFAULT,
                overlap_tokens: CHUNK_OVERLAP_TOKENS_DEFAULT,
            },
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
pub async fn ingest_dir(
    vectors: &SqliteVectors,
    embedder: &dyn Embedder,
    dir: &Path,
    settings: &IngestSettings,
) -> Result<IngestReport, ContextError> {
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
        let previous = vectors.source_hash(&settings.scope, &file.source).await?;
        if previous.as_deref() == Some(file.content_hash.as_str()) {
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
                .replace_source(&settings.scope, &file.source, records, &file.content_hash)
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
                vectors
                    .record_failed_source(&settings.scope, &file.source, &error)
                    .await?;
                report.failed.push((file.source, error));
            }
        }
    }
    let handled = report.added + report.replaced + report.skipped + report.failed.len();
    assert!(
        handled <= settings.load.max_files(),
        "more files handled than the walk allows"
    );
    Ok(report)
}

/// A folder an agent can search: `Knowledge::dir("docs")` plus settings.
/// At `build` the folder is ingested into index `index` of the store file
/// (unchanged files skipped) and the agent gets the `search_knowledge`
/// tool. `embedder` is required in 0.2.
#[derive(Clone)]
pub struct Knowledge {
    pub(crate) dir: PathBuf,
    pub(crate) index: String,
    pub(crate) store: Option<PathBuf>,
    pub(crate) embedder: Option<Arc<dyn Embedder>>,
    pub(crate) load: LoadSettings,
    pub(crate) chunk: ChunkConfig,
}

impl Knowledge {
    /// Starts settings for folder `dir`, read at `build`.
    /// Index `docs`, the agent's SQLite checkpoint file or
    /// `./troupekit.db`, and the `ingest_dir` walk and chunk defaults.
    pub fn dir(dir: impl AsRef<Path>) -> Self {
        let defaults = IngestSettings::default();
        Knowledge {
            dir: dir.as_ref().to_path_buf(),
            index: defaults.scope().to_string(),
            store: None,
            embedder: None,
            load: defaults.load().clone(),
            chunk: defaults.chunk(),
        }
    }

    /// Sets the vector index and scope the folder is stored under.
    /// 1 to 64 of `[a-z0-9_]`, checked at `build`.
    /// Default `docs`.
    pub fn index(mut self, index: impl Into<String>) -> Self {
        self.index = index.into();
        self
    }

    /// Sets the SQLite file the index lives in.
    /// Default: the agent's `Storage::sqlite` file, else `./troupekit.db`.
    /// Returns the settings.
    pub fn store(mut self, path: impl AsRef<Path>) -> Self {
        self.store = Some(path.as_ref().to_path_buf());
        self
    }

    /// Sets the embedder for the folder and for searches. Required in 0.2;
    /// the dimensions must match an existing index of the same name.
    /// Returns the settings.
    pub fn embedder(mut self, embedder: Arc<dyn Embedder>) -> Self {
        self.embedder = Some(embedder);
        self
    }

    /// Sets the folder walk bounds and filters.
    /// Default `LoadSettings::default()`.
    /// Returns the settings.
    pub fn load(mut self, load: LoadSettings) -> Self {
        self.load = load;
        self
    }

    /// Sets the chunk size and overlap, in estimated tokens.
    /// Default 200 tokens with 20 of overlap.
    /// Returns the settings.
    pub fn chunk(mut self, chunk: ChunkConfig) -> Self {
        self.chunk = chunk;
        self
    }
}

impl From<&str> for Knowledge {
    fn from(dir: &str) -> Self {
        Knowledge::dir(dir)
    }
}

impl From<&Path> for Knowledge {
    fn from(dir: &Path) -> Self {
        Knowledge::dir(dir)
    }
}

impl From<PathBuf> for Knowledge {
    fn from(dir: PathBuf) -> Self {
        Knowledge::dir(dir)
    }
}

impl fmt::Debug for Knowledge {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Knowledge")
            .field("dir", &self.dir)
            .field("index", &self.index)
            .field("store", &self.store)
            .field("embedder", &self.embedder.is_some())
            .finish_non_exhaustive()
    }
}
