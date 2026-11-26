use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use troupekit::knowledge::Embedder;
use troupekit::{agent, Agent, AgentMemory, Graph, KitError, Knowledge, Storage};
use troupekit_context::embed::HashingEmbedder;
use troupekit_context::ContextError;
use troupekit_core::event::Usage;
use troupekit_core::message::{ContentPart, Message, Role};
use troupekit_graph::testing::{text_reply, FnModel};
use troupekit_providers::{CompletionResponse, Model, StopReason};

const MEMORY_TOOLS: [&str; 5] = [
    "remember_fact",
    "forget_fact",
    "recall_facts",
    "update_session_summary",
    "read_session_summary",
];

fn calling(tool: &'static str, args: Value) -> Arc<dyn Model> {
    Arc::new(FnModel::new("fake/m", move |req| {
        let result = req
            .messages
            .iter()
            .flat_map(|m| &m.parts)
            .find_map(|p| match p {
                ContentPart::ToolResult { content, .. } => Some(content.clone()),
                _ => None,
            });
        match result {
            Some(out) => text_reply(&out, 3),
            None => Ok(CompletionResponse {
                message: Message::new(
                    Role::Assistant,
                    vec![ContentPart::ToolCall {
                        id: "c1".into(),
                        name: tool.into(),
                        arguments: args.clone(),
                    }],
                )
                .unwrap(),
                usage: Usage {
                    input_tokens: 3,
                    output_tokens: 1,
                    cost_usd_micros: 0,
                },
                stop: StopReason::ToolUse,
            }),
        }
    }))
}

fn echo_user_text() -> Arc<dyn Model> {
    Arc::new(FnModel::new("fake/m", |req| {
        let text: Vec<String> = req
            .messages
            .iter()
            .filter(|m| m.role == Role::User)
            .flat_map(|m| &m.parts)
            .filter_map(|p| match p {
                ContentPart::Text { text } => Some(text.clone()),
                _ => None,
            })
            .collect();
        text_reply(&text.join("\n"), 3)
    }))
}

fn rememberer(model: Arc<dyn Model>, memory: AgentMemory) -> Agent {
    Agent::new("rememberer")
        .instructions("Remember.")
        .model("fake/m")
        .model_client(model)
        .memory(memory)
}

fn db(dir: &tempfile::TempDir) -> PathBuf {
    dir.path().join("memory.db")
}

async fn build_error(agent: Agent) -> KitError {
    match agent.build().await {
        Err(e) => e,
        Ok(_) => panic!("expected a build error"),
    }
}

struct Counting {
    inner: HashingEmbedder,
    calls: AtomicUsize,
}

#[async_trait::async_trait]
impl Embedder for Counting {
    fn dims(&self) -> usize {
        self.inner.dims()
    }

    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, ContextError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.inner.embed(texts).await
    }
}

fn docs(dir: &Path) -> PathBuf {
    let docs = dir.join("docs");
    std::fs::create_dir_all(&docs).unwrap();
    std::fs::write(docs.join("refunds.md"), "Refunds take five days.").unwrap();
    docs
}

#[tokio::test]
async fn memory_needs_user_and_session() {
    let dir = tempfile::tempdir().unwrap();
    for (user, session, what) in [("", "s1", "subject"), ("u1", " ", "session")] {
        let memory = AgentMemory::new(user, session).sqlite(db(&dir));
        let err = build_error(rememberer(echo_user_text(), memory)).await;
        assert!(matches!(err, KitError::Context(_)), "{err}");
        assert!(err.to_string().contains(what), "{err}");
    }
    let err = build_error(rememberer(echo_user_text(), AgentMemory::new("u1", "s1"))).await;
    assert_eq!(
        err.to_string(),
        "agent `rememberer`: memory needs a SQLite file: call AgentMemory::sqlite(..)"
    );
}

#[tokio::test]
async fn memory_tools_are_registered_and_bound() {
    let dir = tempfile::tempdir().unwrap();
    let memory = AgentMemory::new("u1", "s1").sqlite(db(&dir));
    let built = rememberer(echo_user_text(), memory).build().await.unwrap();
    assert_eq!(built.persona().tools, MEMORY_TOOLS.map(String::from));
}

#[tokio::test]
async fn two_users_never_share_memory() {
    let dir = tempfile::tempdir().unwrap();
    let remember = calling("remember_fact", json!({"key": "color", "value": "blue"}));
    let ada = || AgentMemory::new("ada", "s1").sqlite(db(&dir));
    rememberer(remember, ada()).run("note it").await.unwrap();
    let recall = || calling("recall_facts", json!({}));
    let bob = AgentMemory::new("bob", "s1").sqlite(db(&dir));
    let seen = rememberer(recall(), bob).run("what?").await.unwrap();
    assert!(!seen.contains("blue"), "{seen}");
    let seen = rememberer(recall(), ada()).run("what?").await.unwrap();
    assert!(seen.contains("blue"), "{seen}");
}

#[tokio::test]
async fn preamble_is_prepended_when_on() {
    let dir = tempfile::tempdir().unwrap();
    let remember = calling("remember_fact", json!({"key": "color", "value": "blue"}));
    let ada = || AgentMemory::new("ada", "s1").sqlite(db(&dir));
    rememberer(remember, ada()).run("note it").await.unwrap();
    let seen = rememberer(echo_user_text(), ada()).run("hi").await.unwrap();
    assert!(seen.contains("blue") && seen.contains("hi"), "{seen}");
    let quiet = ada().preamble(false);
    let seen = rememberer(echo_user_text(), quiet).run("hi").await.unwrap();
    assert!(!seen.contains("blue"), "{seen}");
}

#[tokio::test]
async fn knowledge_dir_ingests_once_then_skips() {
    let dir = tempfile::tempdir().unwrap();
    let docs = docs(dir.path());
    let embedder = Arc::new(Counting {
        inner: HashingEmbedder::new(32),
        calls: AtomicUsize::new(0),
    });
    let reader = || {
        Agent::new("reader")
            .instructions("Read.")
            .model("fake/m")
            .model_client(echo_user_text())
            .checkpoints(Storage::sqlite(dir.path().join("kb.db")))
            .knowledge(Knowledge::dir(&docs).embedder(embedder.clone()))
    };
    let built = reader().build().await.unwrap();
    assert_eq!(built.persona().tools, ["search_knowledge".to_string()]);
    let after_first = embedder.calls.load(Ordering::SeqCst);
    assert!(after_first > 0);
    reader().build().await.unwrap();
    assert_eq!(embedder.calls.load(Ordering::SeqCst), after_first);
}

#[tokio::test]
async fn knowledge_without_embedder_fails_at_build() {
    let dir = tempfile::tempdir().unwrap();
    let agent = Agent::new("reader")
        .instructions("Read.")
        .model("fake/m")
        .model_client(echo_user_text())
        .knowledge(Knowledge::dir(docs(dir.path())).store(dir.path().join("kb.db")));
    assert_eq!(
        build_error(agent).await.to_string(),
        "agent `reader`: knowledge needs an embedder: call Knowledge::embedder(..)"
    );
}

#[tokio::test]
async fn knowledge_text_is_fenced_external() {
    let dir = tempfile::tempdir().unwrap();
    let knowledge = Knowledge::from(docs(dir.path()))
        .store(dir.path().join("kb.db"))
        .embedder(Arc::new(HashingEmbedder::new(32)));
    let answer = Agent::new("reader")
        .instructions("Read.")
        .model("fake/m")
        .model_client(calling("search_knowledge", json!({"query": "refunds"})))
        .knowledge(knowledge)
        .run("refunds?")
        .await
        .unwrap();
    assert!(answer.contains("<untrusted-content>"), "{answer}");
    assert!(answer.contains("five days"), "{answer}");
}

#[tokio::test]
async fn memory_and_knowledge_share_one_kit() {
    let dir = tempfile::tempdir().unwrap();
    let knowledge = Knowledge::dir(docs(dir.path()))
        .store(db(&dir))
        .embedder(Arc::new(HashingEmbedder::new(32)));
    let built = rememberer(echo_user_text(), AgentMemory::new("ada", "s1"))
        .knowledge(knowledge)
        .build()
        .await
        .unwrap();
    let mut expected = vec!["search_knowledge".to_string()];
    expected.extend(MEMORY_TOOLS.map(String::from));
    assert_eq!(built.persona().tools, expected);
}

#[tokio::test]
async fn graphs_refuse_agents_with_memory_or_knowledge() {
    let dir = tempfile::tempdir().unwrap();
    let memory = AgentMemory::new("u1", "s1").sqlite(db(&dir));
    let err = Graph::new("g")
        .agent("note", rememberer(echo_user_text(), memory))
        .build()
        .await
        .unwrap_err();
    assert_eq!(
        err.to_string(),
        "agent `graph:g`: agent node `note` sets its own memory or knowledge; graphs do not take them in 0.2"
    );
}

fn macro_memory() -> AgentMemory {
    let path = std::env::temp_dir().join(format!("kit-macro-{}.db", std::process::id()));
    AgentMemory::new("u1", "s1").sqlite(path)
}

fn macro_knowledge() -> Knowledge {
    let dir = std::env::temp_dir().join(format!("kit-macro-kb-{}", std::process::id()));
    Knowledge::dir(docs(&dir))
        .store(dir.join("kb.db"))
        .embedder(Arc::new(HashingEmbedder::new(32)))
}

/// Remembers things.
#[agent(model = "fake/m", memory = macro_memory(), knowledge = macro_knowledge())]
fn noted() {}

#[tokio::test]
async fn macro_memory_and_knowledge_keys() {
    let built = noted()
        .model_client(echo_user_text())
        .build()
        .await
        .unwrap();
    let mut expected = vec!["search_knowledge".to_string()];
    expected.extend(MEMORY_TOOLS.map(String::from));
    assert_eq!(built.persona().tools, expected);
}
