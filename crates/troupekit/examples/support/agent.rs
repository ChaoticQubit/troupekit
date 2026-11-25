//! The support agent shared by the `support_agent` example and its
//! end-to-end test: ingest the article folder, bind a knowledge kit to one
//! user and session, and run the support persona over a model.

use std::error::Error;
use std::path::Path;
use std::sync::Arc;
use troupekit::agent::run::{drive, AgentRun, StepOutcome};
use troupekit::agent::{Limits, Persona, SimpleIo};
use troupekit::context::embed::Embedder;
use troupekit::context::tools::KnowledgeKit;
use troupekit::core::message::{Message, Role};
use troupekit::knowledge::{ingest_dir, IngestReport, IngestSettings};
use troupekit::{SqliteStore, SqliteVectors, VectorSettings};
use troupekit_providers::{Model, Price};
use troupekit_tools::ToolRegistry;

/// Result type of the support agent; every error is boxed with its text.
pub type Outcome<T> = Result<T, Box<dyn Error + Send + Sync>>;

const PERSONA: &str = include_str!("support.toml");
const INDEX: &str = "support";

/// Opens the store at `db`, opens the `support` index at the embedder's
/// dimension and ingests `docs` into it. Input: paths and embedder.
/// Output: the store, the index and the ingest report.
pub async fn open(db: &Path, docs: &Path, embedder: &dyn Embedder) -> Outcome<(SqliteStore, Arc<SqliteVectors>, IngestReport)> {
    let store = SqliteStore::open(db).await?;
    let vectors = Arc::new(store.vectors(INDEX, VectorSettings::new(embedder.dims())).await?);
    let report = ingest_dir(&vectors, embedder, docs, &IngestSettings::default()).await?;
    Ok((store, vectors, report))
}

/// Builds the knowledge kit for `user` and `session` over the store.
/// Input: store, index, embedder and the two ids. Output: the kit, with
/// every cap at its safe default.
pub fn kit(store: &SqliteStore, vectors: &Arc<SqliteVectors>, embedder: Arc<dyn Embedder>, user: &str, session: &str) -> Outcome<KnowledgeKit> {
    Ok(KnowledgeKit::builder(user, session)
        .vectors(vectors.clone())
        .keywords(vectors.clone())
        .embedder(embedder)
        .log(Arc::new(store.clone()))
        .blobs(Arc::new(store.clone()))
        .build()?)
}

/// Runs the support persona once on `question`, with the kit's facts and
/// summary prepended, until the model answers. Input: store, kit, model
/// selector, model, price and question. Output: the final answer text.
pub async fn answer(store: &SqliteStore, kit: &KnowledgeKit, selector: &str, model: Arc<dyn Model>, price: Price, question: &str) -> Outcome<String> {
    let mut tools = ToolRegistry::new();
    tools.register_toolkit(kit.toolkit())?;
    let io = SimpleIo::new(tools).with_model(selector, model, price);
    let mut persona = Persona::parse("support/support.toml", PERSONA)?;
    persona.model = selector.to_string();
    let preamble = kit.preamble().await?;
    let input = vec![Message::text(Role::User, format!("{preamble}{question}"))];
    let mut run = AgentRun::new(&persona, input, Limits::default(), &io).await?;
    match drive(&mut run, &io, store).await? {
        StepOutcome::Done(value) => Ok(value.as_str().map_or_else(|| value.to_string(), str::to_string)),
        other => Err(format!("support run ended without an answer: {other:?}").into()),
    }
}
