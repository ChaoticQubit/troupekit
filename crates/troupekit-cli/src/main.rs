//! The `troupekit` binary. 0.2 ships one command, `ingest`.
//! Exit codes: 0 success, 1 a failed run or failed files, 2 bad usage.

mod args;

use std::process::ExitCode;

use args::{EmbedderChoice, IngestArgs, Kind, Quant};
use troupekit::context::chunk::ChunkConfig;
use troupekit::context::embed::{Embedder, HashingEmbedder};
use troupekit::core::secret::Secret;
use troupekit::knowledge::{ingest_dir, IngestReport, IngestSettings};
use troupekit::{HnswParams, IndexKind, Quantization, SqliteStore, VectorSettings};
use troupekit_providers::{Catalog, CatalogError};

const BUILTIN_CATALOG: &str = include_str!("../../../catalog/providers.toml");

fn main() -> ExitCode {
    let argv: Vec<String> = std::env::args_os().skip(1).map(|a| a.to_string_lossy().into_owned()).collect();
    let call = match args::parse(&argv) {
        Ok(call) => call,
        Err(message) => {
            eprintln!("troupekit: {message}");
            eprintln!("{}", args::USAGE);
            return ExitCode::from(2);
        }
    };
    let runtime = match tokio::runtime::Builder::new_current_thread().enable_all().build() {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("troupekit: cannot start the async runtime: {e}");
            return ExitCode::from(1);
        }
    };
    match runtime.block_on(run(&call)) {
        Ok(report) => {
            println!("{}", summary(&call, &report));
            for (path, why) in &report.failed {
                eprintln!("failed: {path}: {why}");
            }
            if report.failed.is_empty() {
                ExitCode::SUCCESS
            } else {
                ExitCode::from(1)
            }
        }
        Err(message) => {
            eprintln!("troupekit: {message}");
            ExitCode::from(1)
        }
    }
}

/// Formats the one summary line printed to stdout after an ingest.
/// Input: the parsed call and the report. Output: the line, with every
/// count in a fixed order so scripts and tests can match it.
fn summary(call: &IngestArgs, r: &IngestReport) -> String {
    format!(
        "ingested {} into {} index {}: {} added, {} replaced, {} unchanged, {} failed, {} chunks, {} unsupported, {} symlinks skipped",
        call.dir.display(),
        call.db.display(),
        call.index,
        r.added,
        r.replaced,
        r.skipped,
        r.failed.len(),
        r.chunks,
        r.unsupported,
        r.symlinks
    )
}

/// Builds the embedder the call names. `hashing:<dims>` needs nothing;
/// a `vendor/model` row reads its key from the variable named by
/// `--key-env`. Output: the embedder, or a message that never holds the key.
fn embedder(call: &IngestArgs) -> Result<Box<dyn Embedder>, String> {
    let selector = match &call.embedder {
        EmbedderChoice::Hashing(dims) => return Ok(Box::new(HashingEmbedder::new(*dims))),
        EmbedderChoice::Catalog(selector) => selector,
    };
    let text = match &call.catalog {
        Some(path) => std::fs::read_to_string(path).map_err(|e| format!("cannot read catalog {}: {e}", path.display()))?,
        None => BUILTIN_CATALOG.to_owned(),
    };
    let catalog = Catalog::load(&text).map_err(|e| format!("catalog: {e}"))?;
    let key = match std::env::var(&call.key_env) {
        Ok(value) if !value.is_empty() => Some(Secret::new(value).map_err(|_| format!("environment variable {} does not hold a usable key", call.key_env))?),
        _ => None,
    };
    match catalog.embedder(selector, key) {
        Ok(e) => Ok(Box::new(e)),
        Err(e @ CatalogError::MissingKey(_)) => Err(format!("{e}: set the environment variable {}", call.key_env)),
        Err(e) => Err(e.to_string()),
    }
}

/// Runs one ingest: builds the embedder and checks the chunk settings
/// before touching the store, opens the store and the named index, then
/// ingests the folder.
/// Output: the report, or a message for a run that could not start.
async fn run(call: &IngestArgs) -> Result<IngestReport, String> {
    let embedder = embedder(call)?;
    let kind = match call.kind {
        Kind::Brute => IndexKind::BruteForce,
        Kind::Hnsw => IndexKind::Hnsw(HnswParams::default()),
    };
    let quantization = match call.quant {
        Quant::F32 => Quantization::F32,
        Quant::F16 => Quantization::F16,
        Quant::I8 => Quantization::I8,
    };
    let mut ingest = IngestSettings::default().with_scope(call.scope.clone());
    if call.chunk_tokens.is_some() || call.overlap_tokens.is_some() {
        let base = ingest.chunk();
        let chunk = ChunkConfig { max_tokens: call.chunk_tokens.unwrap_or(base.max_tokens), overlap_tokens: call.overlap_tokens.unwrap_or(base.overlap_tokens) };
        if chunk.overlap_tokens >= chunk.max_tokens {
            return Err(format!("--overlap-tokens {} must be below the chunk size {}", chunk.overlap_tokens, chunk.max_tokens));
        }
        ingest = ingest.with_chunk(chunk);
    }
    let store = SqliteStore::open(&call.db).await.map_err(|e| format!("cannot open {}: {e}", call.db.display()))?;
    let settings = VectorSettings::new(embedder.dims()).with_kind(kind).with_quantization(quantization);
    let vectors = store.vectors(&call.index, settings).await.map_err(|e| format!("index {}: {e}", call.index))?;
    let report = ingest_dir(&vectors, embedder.as_ref(), &call.dir, &ingest).await.map_err(|e| format!("ingest of {}: {e}", call.dir.display()))?;
    Ok(report)
}

use troupekit_trace as _;
