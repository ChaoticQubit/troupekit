use serde_json::{json, Value};
use std::path::PathBuf;
use std::sync::Arc;
use troupekit::{agent, Agent, AgentMemory, Graph, KitError};
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
async fn graphs_refuse_agents_with_memory() {
    let dir = tempfile::tempdir().unwrap();
    let memory = AgentMemory::new("u1", "s1").sqlite(db(&dir));
    let err = Graph::new("g")
        .agent("note", rememberer(echo_user_text(), memory))
        .build()
        .await
        .unwrap_err();
    assert_eq!(
        err.to_string(),
        "agent `graph:g`: agent node `note` sets its own memory; graphs do not take it in 0.2"
    );
}

fn macro_memory() -> AgentMemory {
    let path = std::env::temp_dir().join(format!("kit-macro-{}.db", std::process::id()));
    AgentMemory::new("u1", "s1").sqlite(path)
}

/// Remembers things.
#[agent(model = "fake/m", memory = macro_memory())]
fn noted() {}

#[tokio::test]
async fn macro_memory_key() {
    let built = noted()
        .model_client(echo_user_text())
        .build()
        .await
        .unwrap();
    assert_eq!(built.persona().tools, MEMORY_TOOLS.map(String::from));
}
