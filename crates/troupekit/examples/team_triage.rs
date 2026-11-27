//! A support desk team: the triage lead routes each request to billing or
//! tech, and a person approves before billing answers. Run from the repo
//! root with `TROUPEKIT_MODEL=vendor/model TROUPEKIT_API_KEY=... cargo run
//! -p troupekit --example team_triage -- "Please refund my last order"`.
//! The store file is `TROUPEKIT_DB`, by default `troupekit-triage.db` in
//! the temp folder.

#[path = "teams/flows.rs"]
mod flows;

use std::io::BufRead;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;
use troupekit::core::secret::Secret;
use troupekit::graph::GraphStatus;
use troupekit::Troupe;
use troupekit_providers::catalog::Catalog;

const CATALOG: &str = include_str!("../../../catalog/providers.toml");
const APPROVAL: &str = "approve_billing";

/// Asks on the terminal whether billing may answer.
/// Input: none. Output: true only for an answer starting with `y`.
/// A closed stdin counts as no.
fn ask_approval() -> flows::Outcome<bool> {
    eprint!("Billing will answer this request. Approve? [y/N] ");
    let mut line = String::new();
    std::io::stdin().lock().read_line(&mut line)?;
    Ok(line.trim_start().to_ascii_lowercase().starts_with('y'))
}

/// Opens the store, runs the triage team on the request given as
/// arguments through the host runner, asks for approval when billing is
/// picked and prints the answer. Output: the answer on stdout.
async fn run() -> flows::Outcome<()> {
    let request = std::env::args().skip(1).collect::<Vec<_>>().join(" ");
    if request.trim().is_empty() {
        return Err("usage: team_triage <request>".into());
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
        || std::env::temp_dir().join("troupekit-triage.db"),
        PathBuf::from,
    );
    let troupe = Troupe::builder().sqlite(&db).build().await?;
    let io = flows::io(&troupe, model, price);
    let id = flows::start(&troupe, io.clone(), "triage.toml", flows::TRIAGE, &request).await?;
    let mut run = flows::work(&troupe, io.clone(), id).await?;
    if matches!(run.status(), GraphStatus::WaitingHuman { .. }) {
        let yes = ask_approval()?;
        flows::approve(&troupe, io.clone(), id, APPROVAL, yes).await?;
        run = flows::work(&troupe, io, id).await?;
        if !yes {
            println!("Not approved; billing did not answer.");
            return Ok(());
        }
    }
    println!("{}", flows::answer(&run)?);
    Ok(())
}

#[tokio::main]
async fn main() -> ExitCode {
    match run().await {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("team_triage: {e}");
            ExitCode::FAILURE
        }
    }
}
