//! A crew that writes a short report: a researcher collects facts, a
//! checker checks them beside the run, and a writer writes the report.
//! Run from the repo root with `TROUPEKIT_MODEL=vendor/model
//! TROUPEKIT_API_KEY=... cargo run -p troupekit --example crew_report --
//! "the history of the bicycle"`. The store file is `TROUPEKIT_DB`, by
//! default `troupekit-crew.db` in the temp folder.

#[path = "teams/flows.rs"]
mod flows;

use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;
use troupekit::core::secret::Secret;
use troupekit::Troupe;
use troupekit_providers::catalog::Catalog;

const CATALOG: &str = include_str!("../../../catalog/providers.toml");

/// Opens the store, runs the report crew on the topic given as arguments
/// through the host runner and prints the report.
/// Input: environment and arguments. Output: the report on stdout.
async fn run() -> flows::Outcome<()> {
    let topic = std::env::args().skip(1).collect::<Vec<_>>().join(" ");
    if topic.trim().is_empty() {
        return Err("usage: crew_report <topic>".into());
    }
    let selector =
        std::env::var("TROUPEKIT_MODEL").map_err(|_| "set TROUPEKIT_MODEL to vendor/model")?;
    let key = std::env::var("TROUPEKIT_API_KEY")
        .ok()
        .map(Secret::new)
        .transpose()?;
    let catalog = Catalog::load(CATALOG)?;
    let price = catalog.price(&selector)?;
    let model = Arc::from(catalog.connect(&selector, key)?);
    let db = std::env::var_os("TROUPEKIT_DB").map_or_else(
        || std::env::temp_dir().join("troupekit-crew.db"),
        PathBuf::from,
    );
    let troupe = Troupe::builder().sqlite(&db).build().await?;
    let io = flows::io(&troupe, model, price);
    let id = flows::start(&troupe, io.clone(), "report.toml", flows::REPORT, &topic).await?;
    eprintln!("run {id} in {}", db.display());
    let run = flows::work(&troupe, io, id).await?;
    println!("{}", flows::answer(&run)?);
    Ok(())
}

#[tokio::main]
async fn main() -> ExitCode {
    match run().await {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("crew_report: {e}");
            ExitCode::FAILURE
        }
    }
}
