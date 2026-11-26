use crate::agent::Agent;
use crate::error::KitError;
use crate::knowledge::{ingest_dir, IngestSettings, Knowledge};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use troupekit_context::embed::{Embedder, HashingEmbedder};
use troupekit_context::tools::{
    KnowledgeKit, FORGET_FACT, READ_SESSION_SUMMARY, RECALL_FACTS, REMEMBER_FACT, SEARCH_KNOWLEDGE,
    UPDATE_SESSION_SUMMARY,
};
use troupekit_core::message::{ContentPart, Message, Role};
use troupekit_store_sqlite::{SqliteStore, SqliteVectors, VectorSettings};

const DEFAULT_STORE: &str = "troupekit.db";
const PLACEHOLDER_INDEX: &str = "troupekit_memory";
const PLACEHOLDER_DIMS: usize = 8;
const MAX_LISTED_FAILURES: usize = 10;
const KNOWLEDGE_ONLY_IDENTITY: &str = "knowledge";
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

    /// Sets the SQLite file facts and summaries live in. Required in 0.2
    /// unless the agent also has `.knowledge(..)`, whose store is used.
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
    let memory = agent.memory.take();
    let knowledge = agent.knowledge.take();
    if memory.is_none() && knowledge.is_none() {
        return Ok((agent, None));
    }
    let name = agent.name().to_string();
    let mut tools: Vec<&str> = Vec::with_capacity(MEMORY_TOOLS.len() + 1);
    let found = match &knowledge {
        Some(k) => {
            let store_path = k
                .store
                .clone()
                .or_else(|| agent.storage.sqlite_path().map(Path::to_path_buf))
                .unwrap_or_else(|| PathBuf::from(DEFAULT_STORE));
            tools.push(SEARCH_KNOWLEDGE);
            let embedder = match k.embedder.clone() {
                Some(embedder) => embedder,
                None => agent.default_embedder()?,
            };
            Some((
                open_knowledge(&name, k, embedder, &store_path).await?,
                store_path,
            ))
        }
        None => None,
    };
    let memory_store = match (&memory, &found) {
        (Some(m), _) if m.sqlite.is_some() => {
            let path = m.sqlite.as_ref().expect("checked by the guard");
            match &found {
                Some((opened, store_path)) if store_path == path => opened.store.clone(),
                _ => SqliteStore::open(path).await?,
            }
        }
        (Some(_), Some((opened, _))) | (None, Some((opened, _))) => opened.store.clone(),
        (Some(_), None) => {
            return Err(KitError::config(
                &name,
                "memory needs a SQLite file: call AgentMemory::sqlite(..)",
            ));
        }
        (None, None) => unreachable!("returned early without memory or knowledge"),
    };
    let (vectors, embedder, scope) = match found {
        Some((opened, _)) => (opened.vectors, opened.embedder, opened.scope),
        None => {
            let vectors = memory_store
                .vectors(PLACEHOLDER_INDEX, VectorSettings::new(PLACEHOLDER_DIMS))
                .await?;
            let embedder: Arc<dyn Embedder> = Arc::new(HashingEmbedder::new(PLACEHOLDER_DIMS));
            (Arc::new(vectors), embedder, PLACEHOLDER_INDEX.to_string())
        }
    };
    let (subject, session) = match &memory {
        Some(m) => {
            tools.extend(MEMORY_TOOLS);
            (m.user.clone(), m.session.clone())
        }
        None => (
            KNOWLEDGE_ONLY_IDENTITY.to_string(),
            KNOWLEDGE_ONLY_IDENTITY.to_string(),
        ),
    };
    let store = Arc::new(memory_store);
    let mut builder = KnowledgeKit::builder(subject, session)
        .vectors(vectors.clone())
        .keywords(vectors)
        .embedder(embedder)
        .log(store.clone())
        .blobs(store)
        .scopes([scope])
        .tools(tools);
    if let Some(m) = &memory {
        if let Some((bytes, facts, log)) = m.fact_caps {
            builder = builder.fact_caps(bytes, facts, log);
        }
        if let Some(bytes) = m.max_summary_bytes {
            builder = builder.max_summary_bytes(bytes);
        }
        if let Some(tokens) = m.preamble_budget_tokens {
            builder = builder.preamble_budget_tokens(tokens);
        }
    }
    let kit = builder.build()?;
    let preamble = memory.filter(|m| m.preamble).map(|_| kit.clone());
    Ok((agent.toolkit(kit.toolkit()), preamble))
}

struct Opened {
    store: SqliteStore,
    vectors: Arc<SqliteVectors>,
    embedder: Arc<dyn Embedder>,
    scope: String,
}

async fn open_knowledge(
    name: &str,
    k: &Knowledge,
    embedder: Arc<dyn Embedder>,
    path: &Path,
) -> Result<Opened, KitError> {
    let store = SqliteStore::open(path).await?;
    let vectors = store
        .vectors(&k.index, VectorSettings::new(embedder.dims()))
        .await?;
    let settings = IngestSettings::default()
        .with_scope(k.index.clone())
        .with_chunk(k.chunk)
        .with_load(k.load.clone());
    let report = ingest_dir(&vectors, embedder.as_ref(), &k.dir, &settings).await?;
    if !report.failed.is_empty() && !k.allow_failed {
        return Err(KitError::config(name, failed_message(&report.failed)));
    }
    Ok(Opened {
        store,
        vectors: Arc::new(vectors),
        embedder,
        scope: k.index.clone(),
    })
}

fn failed_message(failed: &[(String, String)]) -> String {
    assert!(!failed.is_empty(), "called only when files failed");
    let listed: Vec<String> = failed
        .iter()
        .take(MAX_LISTED_FAILURES)
        .map(|(source, error)| format!("{source} ({error})"))
        .collect();
    let more = failed.len().saturating_sub(MAX_LISTED_FAILURES);
    let more = if more > 0 {
        format!("; and {more} more")
    } else {
        String::new()
    };
    let noun = if failed.len() == 1 { "file" } else { "files" };
    format!(
        "knowledge ingest failed for {} {noun}: {}{more}; fix the files or call Knowledge::allow_failed(true)",
        failed.len(),
        listed.join("; ")
    )
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
