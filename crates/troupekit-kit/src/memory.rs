use crate::agent::Agent;
use crate::error::KitError;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use troupekit_context::embed::{Embedder, HashingEmbedder};
use troupekit_context::tools::{
    KnowledgeKit, FORGET_FACT, READ_SESSION_SUMMARY, RECALL_FACTS, REMEMBER_FACT,
    UPDATE_SESSION_SUMMARY,
};
use troupekit_core::message::{ContentPart, Message, Role};
use troupekit_store_sqlite::{SqliteStore, VectorSettings};

const PLACEHOLDER_INDEX: &str = "troupekit_memory";
const PLACEHOLDER_DIMS: usize = 8;
const MEMORY_TOOLS: [&str; 5] = [
    REMEMBER_FACT,
    FORGET_FACT,
    RECALL_FACTS,
    UPDATE_SESSION_SUMMARY,
    READ_SESSION_SUMMARY,
];

/// Long-term memory for one user and one session: facts about the user
/// and a session summary, kept in a SQLite file. The agent gets the five
/// memory tools, bound to this user and session, and by default the facts
/// and summary are prepended to every run's input.
#[derive(Debug, Clone)]
pub struct AgentMemory {
    user: String,
    session: String,
    sqlite: Option<PathBuf>,
    preamble: bool,
    fact_caps: Option<(usize, usize, usize)>,
    max_summary_bytes: Option<usize>,
    preamble_budget_tokens: Option<usize>,
}

impl AgentMemory {
    /// Starts memory bound to `user`, whose facts the tools touch, and
    /// `session`, whose summary they touch. Both are chosen by the host,
    /// never by the model; 1 to 256 bytes each, checked at `build`.
    pub fn new(user: impl Into<String>, session: impl Into<String>) -> Self {
        AgentMemory {
            user: user.into(),
            session: session.into(),
            sqlite: None,
            preamble: true,
            fact_caps: None,
            max_summary_bytes: None,
            preamble_budget_tokens: None,
        }
    }

    /// Sets the SQLite file facts and summaries live in.
    /// Required in 0.2; `build` fails without it.
    /// Returns the memory settings.
    pub fn sqlite(mut self, path: impl AsRef<Path>) -> Self {
        self.sqlite = Some(path.as_ref().to_path_buf());
        self
    }

    /// Turns the preamble of facts and summary before each run on or off.
    /// Default `true`.
    /// Returns the memory settings.
    pub fn preamble(mut self, on: bool) -> Self {
        self.preamble = on;
        self
    }

    /// Caps one fact's bytes, the facts per user and the fact log length.
    /// Defaults 1024, 200 and 1000; zero is refused at `build`.
    /// Returns the memory settings.
    pub fn fact_caps(mut self, max_fact_bytes: usize, max_facts: usize, max_log: usize) -> Self {
        self.fact_caps = Some((max_fact_bytes, max_facts, max_log));
        self
    }

    /// Caps the session summary in bytes.
    /// Default 4096; zero is refused at `build`.
    /// Returns the memory settings.
    pub fn max_summary_bytes(mut self, bytes: usize) -> Self {
        self.max_summary_bytes = Some(bytes);
        self
    }

    /// Caps the preamble in estimated tokens; facts that do not fit are
    /// left out. Default 800; zero is refused at `build`.
    /// Returns the memory settings.
    pub fn preamble_budget_tokens(mut self, tokens: usize) -> Self {
        self.preamble_budget_tokens = Some(tokens);
        self
    }
}

pub(crate) async fn attach(mut agent: Agent) -> Result<(Agent, Option<KnowledgeKit>), KitError> {
    let Some(memory) = agent.memory.take() else {
        return Ok((agent, None));
    };
    let Some(path) = memory.sqlite.as_ref() else {
        return Err(KitError::config(
            agent.name(),
            "memory needs a SQLite file: call AgentMemory::sqlite(..)",
        ));
    };
    let store = SqliteStore::open(path).await?;
    let vectors = Arc::new(
        store
            .vectors(PLACEHOLDER_INDEX, VectorSettings::new(PLACEHOLDER_DIMS))
            .await?,
    );
    let embedder: Arc<dyn Embedder> = Arc::new(HashingEmbedder::new(PLACEHOLDER_DIMS));
    let store = Arc::new(store);
    let mut builder = KnowledgeKit::builder(memory.user.clone(), memory.session.clone())
        .vectors(vectors.clone())
        .keywords(vectors)
        .embedder(embedder)
        .log(store.clone())
        .blobs(store)
        .scopes([PLACEHOLDER_INDEX])
        .tools(MEMORY_TOOLS);
    if let Some((bytes, facts, log)) = memory.fact_caps {
        builder = builder.fact_caps(bytes, facts, log);
    }
    if let Some(bytes) = memory.max_summary_bytes {
        builder = builder.max_summary_bytes(bytes);
    }
    if let Some(tokens) = memory.preamble_budget_tokens {
        builder = builder.preamble_budget_tokens(tokens);
    }
    let kit = builder.build()?;
    let preamble = memory.preamble.then(|| kit.clone());
    Ok((agent.toolkit(kit.toolkit()), preamble))
}

pub(crate) async fn prepend(
    kit: &KnowledgeKit,
    mut messages: Vec<Message>,
) -> Result<Vec<Message>, KitError> {
    let block = kit.preamble().await?;
    if block.is_empty() {
        return Ok(messages);
    }
    match messages.first_mut() {
        Some(first) if first.role == Role::User => first.parts.insert(
            0,
            ContentPart::Text {
                text: format!("{block}\n\n"),
            },
        ),
        _ => messages.insert(0, Message::text(Role::User, block)),
    }
    Ok(messages)
}
