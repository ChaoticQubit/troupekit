//! A support agent over a folder of markdown articles. Run from the repo
//! root with `TROUPEKIT_MODEL=vendor/model TROUPEKIT_API_KEY=...
//! cargo run -p troupekit --features knowledge --example support_agent --
//! "How long do I have to return an item?"`. The store file is
//! `TROUPEKIT_DB`, by default `troupekit-support.db` in the temp folder.

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

/// Builds the embedder named by `selector`. Only `hashing:<dims>` exists
/// until the provider embeddings endpoint lands. Input: the selector.
/// Output: the embedder, or an error naming the accepted form.
fn embedder(selector: &str) -> agent::Outcome<Arc<dyn Embedder>> {
    let dims = selector
        .strip_prefix("hashing:")
        .and_then(|d| d.parse::<usize>().ok())
        .filter(|d| (1..=65_536).contains(d))
        .ok_or_else(|| format!("TROUPEKIT_EMBEDDER must be hashing:<dims>, got {selector}"))?;
    Ok(Arc::new(HashingEmbedder::new(dims)))
}

/// Ingests the articles, then answers the question given as arguments.
/// Input: environment and arguments. Output: the answer on stdout.
async fn run() -> agent::Outcome<()> {
    let question = std::env::args().skip(1).collect::<Vec<_>>().join(" ");
    if question.trim().is_empty() {
        return Err("usage: support_agent <question>".into());
    }
    let selector = std::env::var("TROUPEKIT_MODEL").map_err(|_| "set TROUPEKIT_MODEL to vendor/model")?;
    let key = std::env::var("TROUPEKIT_API_KEY").ok().map(Secret::new).transpose()?;
    let embedder = embedder(&std::env::var("TROUPEKIT_EMBEDDER").unwrap_or_else(|_| EMBEDDER_DEFAULT.to_string()))?;
    let catalog = Catalog::load(CATALOG)?;
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
