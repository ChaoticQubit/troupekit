//! Ingest a folder into a named vector index in the store file. Unchanged
//! files are skipped by content hash, changed files are replaced whole, a
//! file that fails is recorded and the walk goes on, and the sources of
//! files gone from the folder are removed.

use std::collections::HashSet;
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
const MAX_REMOVE_SHARE_DEFAULT: u8 = 50;

/// Settings for `ingest_dir`. Defaults: scope `docs`, chunks of 200
/// tokens with 20 of overlap, the safe folder walk of `LoadSettings` and
/// `Origin::External`, so folder text is fenced as untrusted in prompts.
/// A run never removes more than half the scope, or all of it from an
/// empty folder, unless allowed. Changed origin or chunk config re-ingests.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IngestSettings {
    scope: String,
    chunk: ChunkConfig,
    load: LoadSettings,
    origin: Origin,
    allow_remove_all: bool,
    max_remove_share: u8,
}

impl Default for IngestSettings {
    fn default() -> Self {
        Self {
            scope: SCOPE_DEFAULT.to_string(),
            chunk: ChunkConfig { max_tokens: CHUNK_MAX_TOKENS_DEFAULT, overlap_tokens: CHUNK_OVERLAP_TOKENS_DEFAULT },
            load: LoadSettings::default(),
            origin: Origin::External,
            allow_remove_all: false,
            max_remove_share: MAX_REMOVE_SHARE_DEFAULT,
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

    /// Allows a run to remove any number of sources, all of them included,
    /// for a folder that was emptied on purpose. Off, a walk that finds no
    /// supported file while the scope has sources is refused. Default off.
    pub fn with_allow_remove_all(mut self, allow: bool) -> Self {
        self.allow_remove_all = allow;
        self
    }

    /// Sets the largest share of the scope's sources, in percent 0 to 100,
    /// one run may remove; a run above it is refused with the planned
    /// count before any write. Ignored with `allow_remove_all`. Default 50.
    pub fn with_max_remove_share(mut self, percent: u8) -> Self {
        self.max_remove_share = percent;
        self
    }

    /// Returns whether a run may remove every source of the scope.
    /// Guards against an unmounted or mistyped folder.
    /// Default false.
    pub fn allow_remove_all(&self) -> bool {
        self.allow_remove_all
    }

    /// Returns the largest share of sources, in percent, one run removes.
    /// Above it the run is refused before any write.
    /// Default 50.
    pub fn max_remove_share(&self) -> u8 {
        self.max_remove_share
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
pub async fn ingest_dir(vectors: &SqliteVectors, embedder: &dyn Embedder, dir: &Path, settings: &IngestSettings) -> Result<IngestReport, ContextError> {
    if settings.scope.trim().is_empty() {
        return Err(ContextError::Invalid("scope must not be empty".into()));
    }
    if settings.max_remove_share > 100 {
        return Err(ContextError::Invalid(format!("max_remove_share is {}%, above 100%", settings.max_remove_share)));
    }
    let walked = walk(dir, &settings.load)?;
    let present: HashSet<String> = walked.files.iter().map(|f| f.source.clone()).collect();
    let stored = vectors.sources(&settings.scope).await?;
    let gone: Vec<String> = stored.iter().filter(|s| !present.contains(*s)).cloned().collect();
    check_removals(gone.len(), stored.len(), present.is_empty(), settings)?;
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
    for source in &gone {
        vectors.remove_source(&settings.scope, source).await?;
        report.removed += 1;
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

/// Refuses a run that would wipe the scope from an empty folder, or
/// remove more than `max_remove_share` percent of its sources, unless
/// `allow_remove_all`. Input: planned and stored counts, whether the walk
/// found no file, and settings. Output: `Ok`, or `Invalid` with the count.
fn check_removals(planned: usize, stored: usize, no_files: bool, settings: &IngestSettings) -> Result<(), ContextError> {
    assert!(planned <= stored, "more removals planned than sources stored");
    if settings.allow_remove_all || planned == 0 {
        return Ok(());
    }
    let scope = &settings.scope;
    if no_files {
        return Err(ContextError::Invalid(format!(
            "walk found no supported files but scope {scope} has {stored} sources; refusing to remove them all \
             (check the folder, or set with_allow_remove_all(true))"
        )));
    }
    let share = usize::from(settings.max_remove_share);
    if planned * 100 > stored * share {
        return Err(ContextError::Invalid(format!(
            "refusing to remove {planned} of {stored} sources of scope {scope}, above max_remove_share {share}% \
             (check the folder, or raise with_max_remove_share or set with_allow_remove_all(true))"
        )));
    }
    Ok(())
}
