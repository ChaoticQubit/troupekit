//! A support agent over a folder of markdown articles. Run from the repo
//! root with `TROUPEKIT_MODEL=vendor/model TROUPEKIT_API_KEY=...
//! cargo run -p troupekit --features knowledge --example support_agent --
//! "How long do I have to return an item?"`. The store file is
//! `TROUPEKIT_DB`, by default `troupekit-support.db` in the temp folder.
//! `TROUPEKIT_EMBEDDER` is `hashing:<dims>` or a catalog `vendor/model`
//! embedding row, keyed by `TROUPEKIT_EMBEDDER_KEY` or `TROUPEKIT_API_KEY`.

#[path = "support/agent.rs"]
mod agent;

use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;
use troupekit::context::embed::{Embedder, HashingEmbedder};
use troupekit::core::secret::Secret;
use troupekit_providers::catalog::Catalog;

const CATALOG: &str = include_str!("../../../catalog/providers.toml");
const EMBEDDER_DEFAULT: &str = "hashing:256";
const USER: &str = "demo-user";
const SESSION: &str = "demo-session";

/// Builds the embedder named by `selector`: `hashing:<dims>` offline, or
/// a catalog `vendor/model` embedding row over the network with `key`.
/// Input: catalog, selector and key. Output: the embedder or the error.
fn embedder(catalog: &Catalog, selector: &str, key: Option<Secret>) -> agent::Outcome<Arc<dyn Embedder>> {
    let Some(dims) = selector.strip_prefix("hashing:") else {
        return Ok(Arc::new(catalog.embedder(selector, key)?));
    };
    let dims = dims
        .parse::<usize>()
        .ok()
        .filter(|d| (1..=65_536).contains(d))
        .ok_or_else(|| format!("TROUPEKIT_EMBEDDER hashing:<dims> needs dims 1 to 65536, got {selector}"))?;
    Ok(Arc::new(HashingEmbedder::new(dims)))
}

/// Reads environment variable `name` as a key, or `None` when unset.
/// Input: the name. Output: the key, or an error for a blank value.
/// Keys never reach logs or errors.
fn env_key(name: &str) -> agent::Outcome<Option<Secret>> {
    Ok(std::env::var(name).ok().map(Secret::new).transpose()?)
}

/// Ingests the articles, then answers the question given as arguments.
/// Input: environment and arguments. Output: the answer on stdout.
async fn run() -> agent::Outcome<()> {
    let question = std::env::args().skip(1).collect::<Vec<_>>().join(" ");
    if question.trim().is_empty() {
        return Err("usage: support_agent <question>".into());
    }
    let selector = std::env::var("TROUPEKIT_MODEL").map_err(|_| "set TROUPEKIT_MODEL to vendor/model")?;
    let key = env_key("TROUPEKIT_API_KEY")?;
    let catalog = Catalog::load(CATALOG)?;
    let embed_key = env_key("TROUPEKIT_EMBEDDER_KEY")?.or_else(|| key.clone());
    let embedder = embedder(&catalog, &std::env::var("TROUPEKIT_EMBEDDER").unwrap_or_else(|_| EMBEDDER_DEFAULT.to_string()), embed_key)?;
    let price = catalog.price(&selector)?;
    let model = Arc::from(catalog.connect(&selector, key)?);
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/support");
    let db = std::env::var_os("TROUPEKIT_DB").map_or_else(|| std::env::temp_dir().join("troupekit-support.db"), PathBuf::from);
    let (store, vectors, report) = agent::open(&db, &root.join("docs"), embedder.as_ref()).await?;
    eprintln!("ingested: {} added, {} replaced, {} unchanged", report.added, report.replaced, report.skipped);
    let kit = agent::kit(&store, &vectors, embedder, USER, SESSION)?;
    println!("{}", agent::answer(&store, &kit, &selector, model, price, &question).await?);
    Ok(())
}

#[tokio::main]
async fn main() -> ExitCode {
    match run().await {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("support_agent: {e}");
            ExitCode::FAILURE
        }
    }
}
