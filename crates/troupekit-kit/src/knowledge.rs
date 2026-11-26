//! Ingest a folder into a named vector index in the store file. Unchanged
//! files are skipped by content hash, changed files are replaced whole, a
//! file that fails is recorded and the walk goes on, and the sources of
//! files gone from the folder are removed.

use std::collections::HashSet;
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;
pub use troupekit_context::chunk::ChunkConfig;
pub use troupekit_context::embed::Embedder;
use troupekit_context::ingest::{document_records, Document, Origin};
use troupekit_context::load::walk;
pub use troupekit_context::load::LoadSettings;
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
/// unchanged, sources removed because their file is gone, files the walk
/// passed over (`unsupported`, `symlinks`), failed files with their error,
/// and the chunks written.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IngestReport {
    pub added: usize,
    pub replaced: usize,
    pub skipped: usize,
    pub removed: usize,
    pub unsupported: usize,
    pub symlinks: usize,
    pub failed: Vec<(String, String)>,
    pub chunks: usize,
}

/// Ingests every loadable file under `dir` into `vectors`: unchanged
/// files are skipped, others swapped in with `replace_source`, and after
/// a complete walk the sources of the scope with no file are removed.
/// Output: the report, or an error for a walk cap, a blank scope or a
/// store write that fails.
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
        removed: 0,
        unsupported: walked.unsupported,
        symlinks: walked.symlinks,
        failed: Vec::new(),
        chunks: 0,
    };
    let at_ms = SystemClock.now_ms();
    let present: HashSet<String> = walked.files.iter().map(|f| f.source.clone()).collect();
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
                vectors
                    .record_failed_source(&settings.scope, &file.source, &error)
                    .await?;
                report.failed.push((file.source, error));
            }
        }
    }
    for source in vectors.sources(&settings.scope).await? {
        if !present.contains(&source) {
            vectors.remove_source(&settings.scope, &source).await?;
            report.removed += 1;
        }
    }
    let handled = report.added + report.replaced + report.skipped + report.failed.len();
    assert!(
        handled <= settings.load.max_files(),
        "more files handled than the walk allows"
    );
    Ok(report)
}

/// Hash stored per source: the file's content hash together with the
/// origin and chunk config its chunks were made with, so a settings
/// change no longer matches and the file is ingested again.
/// Input: the content hash and the settings. Output: 16 hex chars.
fn ingest_hash(content_hash: &str, settings: &IngestSettings) -> String {
    let max = settings.chunk.max_tokens.to_string();
    let overlap = settings.chunk.overlap_tokens.to_string();
    let hash = format!(
        "{:016x}",
        stable_hash(&[content_hash, settings.origin.as_str(), &max, &overlap])
    );
    assert_eq!(hash.len(), 16, "ingest hash is 16 hex chars");
    hash
}

/// A folder an agent can search: `Knowledge::dir("docs")` plus settings.
/// At `build` the folder is ingested into index `index` of the store file
/// (unchanged files skipped) and the agent gets the `search_knowledge`
/// tool. The embedder defaults to the catalog's for the agent's vendor.
#[derive(Clone)]
pub struct Knowledge {
    pub(crate) dir: PathBuf,
    pub(crate) index: String,
    pub(crate) store: Option<PathBuf>,
    pub(crate) embedder: Option<Arc<dyn Embedder>>,
    pub(crate) load: LoadSettings,
    pub(crate) chunk: ChunkConfig,
    pub(crate) allow_failed: bool,
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
            allow_failed: false,
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

    /// Sets the embedder for the folder and for searches, overriding the
    /// catalog embedder of the agent's vendor; dimensions must match an
    /// existing index of the same name.
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

impl Knowledge {
    /// Lets `build` succeed when some files fail to ingest; the rest is
    /// searchable and failed files are retried on the next build. Default
    /// `false`: `build` fails and lists the failed files.
    pub fn allow_failed(mut self, allow: bool) -> Self {
        self.allow_failed = allow;
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
            .field("allow_failed", &self.allow_failed)
            .finish_non_exhaustive()
    }
}
