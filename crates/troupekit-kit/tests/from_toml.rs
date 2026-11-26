use serde::Deserialize;
use serde_json::json;
use std::sync::Arc;
use troupekit::{tool, Agent, Graph, KitError};
use troupekit_core::event::Usage;
use troupekit_core::message::{ContentPart, Message, Role};
use troupekit_graph::testing::{text_reply, FnModel};
use troupekit_providers::{CompletionResponse, Model, StopReason};
use troupekit_tools::Tool;

const FIXTURE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/support.toml");

#[derive(Deserialize, schemars::JsonSchema)]
struct OrderArgs {
    id: u32,
}

fn lookup_order() -> impl Tool + 'static {
    tool(
        "lookup_order",
        "Looks up an order by id",
        |x: OrderArgs| async move { format!("order {} ships Monday", x.id) },
    )
}

fn support_model() -> Arc<dyn Model> {
    Arc::new(FnModel::new("fake/m", |req| {
        let found = req
            .messages
            .iter()
            .flat_map(|m| &m.parts)
            .find_map(|p| match p {
                ContentPart::ToolResult { content, .. } => Some(content.clone()),
                _ => None,
            });
        match found {
            Some(order) => text_reply(&json!({ "answer": order }).to_string(), 10),
            None => Ok(CompletionResponse {
                message: Message::new(
                    Role::Assistant,
                    vec![ContentPart::ToolCall {
                        id: "c1".into(),
                        name: "lookup_order".into(),
                        arguments: json!({"id": 7}),
                    }],
                )
                .unwrap(),
                usage: Usage {
                    input_tokens: 10,
                    output_tokens: 5,
                    cost_usd_micros: 0,
                },
                stop: StopReason::ToolUse,
            }),
        }
    }))
}

fn fixture_text() -> String {
    std::fs::read_to_string(FIXTURE).unwrap()
}

#[tokio::test]
async fn toml_agent_builds_and_runs() {
    let outcome = Agent::from_toml_file(FIXTURE)
        .model_client(support_model())
        .tool(lookup_order())
        .run_with("When does order 7 ship?")
        .await
        .unwrap();
    assert_eq!(outcome.output(), &json!({"answer": "order 7 ships Monday"}));
    assert_eq!(outcome.model_calls(), 2);
}

#[tokio::test]
async fn toml_values_survive_build() {
    let built = Agent::from_toml(fixture_text())
        .model_client(support_model())
        .tool(lookup_order())
        .build()
        .await
        .unwrap();
    let persona = built.persona();
    assert_eq!(persona.name, "support");
    assert_eq!(persona.display_name, "Support");
    assert_eq!(persona.model, "anthropic/claude-haiku-5-5");
    assert_eq!(persona.max_turns, 8);
    assert_eq!(persona.tools, vec!["lookup_order".to_string()]);
    let schema = persona.output.as_ref().expect("output schema survives");
    assert_eq!(schema["required"], json!(["answer"]));
}

#[tokio::test]
async fn builder_setting_overrides_toml() {
    let built = Agent::from_toml_file(FIXTURE)
        .model("fake/other")
        .instructions("Be brief.")
        .display_name("Helper")
        .role("Helps.")
        .goal("Help.")
        .max_turns(3)
        .max_tokens(256)
        .model_client(support_model())
        .tool(lookup_order())
        .build()
        .await
        .unwrap();
    let persona = built.persona();
    assert_eq!(persona.name, "support");
    assert_eq!(persona.model, "fake/other");
    assert_eq!(persona.instructions, "Be brief.");
    assert_eq!(persona.display_name, "Helper");
    assert_eq!(persona.role, "Helps.");
    assert_eq!(persona.goal, "Help.");
    assert_eq!(persona.max_turns, 3);
    assert_eq!(persona.max_tokens, 256);
    assert!(persona.output.is_some());
}

#[tokio::test]
async fn toml_over_1_mib_is_refused() {
    let mut text = fixture_text();
    text.push_str(&format!(
        "context = \"{}\"\n",
        "x".repeat(troupekit::MAX_PERSONA_BYTES)
    ));
    let err = Agent::from_toml(text)
        .model_client(support_model())
        .tool(lookup_order())
        .build()
        .await
        .unwrap_err();
    assert!(matches!(err, KitError::Config { .. }), "{err:?}");
    assert!(err.to_string().contains("over 1 MiB"), "{err}");
}

#[tokio::test]
async fn toml_path_with_dotdot_is_refused() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/../tests/fixtures/support.toml"
    );
    let err = Agent::from_toml_file(path)
        .model_client(support_model())
        .tool(lookup_order())
        .build()
        .await
        .unwrap_err();
    assert!(matches!(err, KitError::Config { .. }), "{err:?}");
    assert!(err.to_string().contains("must not contain `..`"), "{err}");
}

#[tokio::test]
async fn toml_tool_must_be_registered() {
    let err = Agent::from_toml_file(FIXTURE)
        .model_client(support_model())
        .build()
        .await
        .unwrap_err();
    assert!(err.to_string().contains("lookup_order"), "{err}");
}

#[tokio::test]
async fn bad_toml_reports_source_name() {
    let err = Agent::from_toml("name = \"support\"\nrole = [")
        .model_client(support_model())
        .build()
        .await
        .unwrap_err();
    assert!(matches!(err, KitError::Persona(_)), "{err:?}");
    assert!(err.to_string().contains("agent:toml"), "{err}");
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("broken.toml");
    std::fs::write(&path, "name = \"support\"\nrole = [").unwrap();
    let err = Agent::from_toml_file(&path)
        .model_client(support_model())
        .build()
        .await
        .unwrap_err();
    assert!(err.to_string().contains("broken.toml"), "{err}");
}

#[tokio::test]
async fn toml_instructions_file_is_refused() {
    let text = fixture_text().replace(
        "instructions = \"\"\"",
        "instructions_file = \"support.md\"\ncontext = \"\"\"",
    );
    let err = Agent::from_toml(text)
        .model_client(support_model())
        .tool(lookup_order())
        .build()
        .await
        .unwrap_err();
    assert!(matches!(err, KitError::Config { .. }), "{err:?}");
    assert!(err.to_string().contains("from_toml_file"), "{err}");
}

fn persona_with_instructions_file(dir: &std::path::Path, file: &str) -> std::path::PathBuf {
    let text = fixture_text().replace(
        "instructions = \"\"\"",
        &format!("instructions_file = \"{file}\"\ncontext = \"\"\""),
    );
    let persona = dir.join("support.toml");
    std::fs::write(&persona, text).unwrap();
    persona
}

#[tokio::test]
async fn toml_file_reads_instructions_file_beside_it() {
    let dir = tempfile::tempdir().unwrap();
    let persona = persona_with_instructions_file(dir.path(), "notes/support.md");
    std::fs::create_dir(dir.path().join("notes")).unwrap();
    std::fs::write(dir.path().join("notes/support.md"), "Quote the order line.").unwrap();
    let built = Agent::from_toml_file(&persona)
        .model_client(support_model())
        .tool(lookup_order())
        .build()
        .await
        .unwrap();
    assert_eq!(built.persona().instructions, "Quote the order line.");
    assert!(built
        .persona()
        .system_prompt()
        .contains("Quote the order line."));
}

#[tokio::test]
async fn toml_file_instructions_file_must_exist() {
    let dir = tempfile::tempdir().unwrap();
    let persona = persona_with_instructions_file(dir.path(), "missing.md");
    let err = Agent::from_toml_file(&persona)
        .model_client(support_model())
        .tool(lookup_order())
        .build()
        .await
        .unwrap_err();
    assert!(matches!(err, KitError::Config { .. }), "{err:?}");
    assert!(err.to_string().contains("does not exist"), "{err}");
}

#[cfg(unix)]
#[tokio::test]
async fn toml_file_instructions_symlink_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let outside = tempfile::tempdir().unwrap();
    let secret = outside.path().join("secret.md");
    std::fs::write(&secret, "secret text").unwrap();
    std::os::unix::fs::symlink(&secret, dir.path().join("support.md")).unwrap();
    let persona = persona_with_instructions_file(dir.path(), "support.md");
    let err = Agent::from_toml_file(&persona)
        .model_client(support_model())
        .tool(lookup_order())
        .build()
        .await
        .unwrap_err();
    assert!(matches!(err, KitError::Config { .. }), "{err:?}");
    assert!(err.to_string().contains("symlink"), "{err}");
}

#[tokio::test]
async fn two_toml_agents_in_one_graph_are_told_apart() {
    let model = support_model();
    let order: Arc<dyn Tool> = Arc::new(lookup_order());
    let other = fixture_text().replace("anthropic/claude-haiku-5-5", "anthropic/claude-opus-5-5");
    let err = Graph::new("pair")
        .agent(
            "first",
            Agent::from_toml(fixture_text())
                .model_client(model.clone())
                .tool_arc(order.clone()),
        )
        .agent(
            "second",
            Agent::from_toml(other).model_client(model).tool_arc(order),
        )
        .build()
        .await
        .unwrap_err();
    assert!(
        err.to_string()
            .contains("two different agents are named `support`"),
        "{err}"
    );
}

#[cfg(unix)]
#[tokio::test]
async fn toml_file_instructions_outside_its_directory_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let outside = tempfile::tempdir().unwrap();
    std::fs::write(outside.path().join("support.md"), "secret text").unwrap();
    std::os::unix::fs::symlink(outside.path(), dir.path().join("notes")).unwrap();
    let persona = persona_with_instructions_file(dir.path(), "notes/support.md");
    let err = Agent::from_toml_file(&persona)
        .model_client(support_model())
        .tool(lookup_order())
        .build()
        .await
        .unwrap_err();
    assert!(matches!(err, KitError::Config { .. }), "{err:?}");
    assert!(
        err.to_string().contains("persona file's directory"),
        "{err}"
    );
}

fn node_personas(graph: &troupekit::BuiltGraph) -> Vec<String> {
    graph
        .workflow()
        .nodes
        .iter()
        .filter_map(|n| match &n.kind {
            troupekit_graph::NodeKind::Agent { persona, .. } => Some(persona.clone()),
            _ => None,
        })
        .collect()
}

#[tokio::test]
async fn toml_agent_in_graph_uses_its_persona_name() {
    let built = Graph::new("solo")
        .agent(
            "answer",
            Agent::from_toml_file(FIXTURE)
                .model_client(support_model())
                .tool(lookup_order()),
        )
        .build()
        .await
        .unwrap();
    assert_eq!(node_personas(&built), vec!["support".to_string()]);
    let output = built.run("When does order 7 ship?").await.unwrap();
    assert!(
        output.to_string().contains("order 7 ships Monday"),
        "{output}"
    );
}

#[tokio::test]
async fn two_toml_personas_share_a_graph() {
    let model = support_model();
    let order: Arc<dyn Tool> = Arc::new(lookup_order());
    let billing = fixture_text().replace("name = \"support\"", "name = \"billing\"");
    let built = Graph::new("pair")
        .agent(
            "first",
            Agent::from_toml(fixture_text())
                .model_client(model.clone())
                .tool_arc(order.clone()),
        )
        .agent(
            "second",
            Agent::from_toml(billing)
                .model_client(model)
                .tool_arc(order),
        )
        .edge("first", "second")
        .build()
        .await
        .unwrap();
    assert_eq!(
        node_personas(&built),
        vec!["support".to_string(), "billing".to_string()]
    );
}
