use crate::common::{echo_model, AddArgs};
use serde_json::json;
use std::sync::Arc;
use troupekit::{tool, Agent, Graph, KitError, Storage};
use troupekit_core::event::Usage;
use troupekit_core::message::{ContentPart, Message, Role};
use troupekit_graph::testing::{text_reply, FnModel};
use troupekit_providers::{CompletionResponse, Model, StopReason};

/// Uppercases the draft.
#[tool]
fn upper_draft(draft: String) -> String {
    draft.to_uppercase()
}

/// Trims the input.
#[tool]
fn clean_text(input: String) -> String {
    input.trim().to_string()
}

/// Counts characters of the cleaned text.
#[tool]
fn count_chars(clean: String) -> usize {
    clean.chars().count()
}

/// Shouts the text.
#[tool]
fn shout(text: String) -> String {
    text.to_uppercase()
}

/// Adds one.
#[tool]
fn increment(input: i64) -> i64 {
    input + 1
}

/// Takes optional text.
#[tool]
fn maybe_text(input: Option<String>) -> String {
    input.unwrap_or_default()
}

/// Takes any JSON value.
#[tool]
fn any_value(input: serde_json::Value) -> String {
    input.to_string()
}

fn drafter() -> Agent {
    Agent::new("drafter")
        .instructions("Draft.")
        .model("fake/m")
        .model_client(echo_model())
}

fn shouting_model() -> Arc<dyn Model> {
    Arc::new(FnModel::new("fake/m", |req| {
        let result = req
            .messages
            .iter()
            .flat_map(|m| &m.parts)
            .find_map(|p| match p {
                ContentPart::ToolResult { content, .. } => Some(content.clone()),
                _ => None,
            });
        match result {
            Some(out) => text_reply(&format!("got {out}"), 3),
            None => Ok(CompletionResponse {
                message: Message::new(
                    Role::Assistant,
                    vec![ContentPart::ToolCall {
                        id: "c1".into(),
                        name: "shout".into(),
                        arguments: json!({"text": "hi"}),
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

async fn config_error(graph: Graph) -> String {
    match graph.build().await {
        Err(e @ KitError::Config { .. }) => e.to_string(),
        Err(other) => panic!("expected Config, got {other}"),
        Ok(_) => panic!("expected Config, got a built graph"),
    }
}

#[tokio::test]
async fn agent_then_fn_pipeline_runs_offline() {
    let output = Graph::new("pipeline")
        .agent("draft", drafter())
        .step("upper", upper_draft)
        .run("hi")
        .await
        .unwrap();
    let text = output.as_str().unwrap();
    assert!(text.starts_with("HELLO! YOU SAID:"), "{text}");
}

#[tokio::test]
async fn no_edges_means_chain_in_order() {
    let report = Graph::new("lab3")
        .step("clean", clean_text)
        .step("count", count_chars)
        .run_with("  abc ")
        .await
        .unwrap();
    assert_eq!(report.output(), &json!("3"));
    assert_eq!(report.state()["clean"], json!("abc"));
    assert_eq!(report.state()["input"], json!("  abc "));
}

#[tokio::test]
async fn edges_set_the_chain_order() {
    let output = Graph::new("lab3")
        .step("count", count_chars)
        .step("clean", clean_text)
        .edge("clean", "count")
        .run("  ab ")
        .await
        .unwrap();
    assert_eq!(output, json!("2"));
}

#[tokio::test]
async fn branching_edges_are_refused() {
    let text = config_error(
        Graph::new("g")
            .step("clean", clean_text)
            .step("count", count_chars)
            .step("other", count_chars)
            .edge("clean", "count")
            .edge("clean", "other"),
    )
    .await;
    assert_eq!(
        text,
        "agent `graph:g`: node `clean` has two outgoing edges; fan-out needs workflow TOML"
    );
}

#[tokio::test]
async fn fn_args_must_match_feeding_nodes() {
    let text = config_error(
        Graph::new("pipeline")
            .agent("draft", drafter())
            .step("loud", shout),
    )
    .await;
    assert_eq!(
        text,
        "agent `graph:pipeline`: step `loud` takes (text) but is fed by (draft); name its arguments after the nodes that feed it"
    );
}

#[tokio::test]
async fn step_arguments_must_accept_text() {
    let text = config_error(Graph::new("g").step("inc", increment)).await;
    assert_eq!(
        text,
        "agent `graph:g`: step `inc` argument `input` must accept a string; every graph node is fed text"
    );
    assert_eq!(
        Graph::new("g")
            .step("m", maybe_text)
            .run("x")
            .await
            .unwrap(),
        json!("x")
    );
    assert!(Graph::new("g").step("a", any_value).build().await.is_ok());
}

#[tokio::test]
async fn node_with_two_inputs_is_refused() {
    let text = config_error(
        Graph::new("g")
            .step("clean", clean_text)
            .step("other", clean_text)
            .step("count", count_chars)
            .edge("clean", "count")
            .edge("other", "count"),
    )
    .await;
    assert_eq!(
        text,
        "agent `graph:g`: node `count` has two incoming edges; joins need workflow TOML"
    );
}

#[tokio::test]
async fn agent_and_step_sharing_a_tool_see_plain_output() {
    let agent = Agent::new("caller")
        .instructions("Call shout.")
        .model("fake/m")
        .model_client(shouting_model())
        .tool(shout);
    let output = Graph::new("shared")
        .agent("text", agent)
        .step("loud", shout)
        .run("go")
        .await
        .unwrap();
    assert_eq!(output, json!("GOT HI"));
}

#[tokio::test]
async fn same_selector_nodes_keep_their_own_model_clients() {
    let caller = || {
        Agent::new("caller")
            .instructions("Call shout.")
            .model("fake/m")
            .model_client(shouting_model())
            .tool(shout)
    };
    let output = Graph::new("g")
        .agent("draft", drafter())
        .agent("text", caller())
        .run("hi")
        .await
        .unwrap();
    assert_eq!(output, json!("got HI"));
    let output = Graph::new("g")
        .agent("text", caller())
        .agent("draft", drafter())
        .run("hi")
        .await
        .unwrap();
    assert!(
        output.as_str().unwrap().starts_with("Hello! You said:"),
        "{output}"
    );
}

#[tokio::test]
async fn one_agent_name_with_two_model_clients_is_refused() {
    let other = drafter().model_client(shouting_model());
    for (first, second) in [(drafter(), other.clone()), (other, drafter())] {
        let text = config_error(Graph::new("g").agent("draft", first).agent("again", second)).await;
        assert_eq!(
            text,
            "agent `graph:g`: two different agents are named `drafter`"
        );
    }
}

#[tokio::test]
async fn one_agent_name_with_two_prices_is_refused() {
    let model = echo_model();
    let priced = |price| {
        Agent::new("drafter")
            .instructions("Draft.")
            .model("fake/m")
            .model_client(model.clone())
            .price(price)
    };
    let paid = troupekit_providers::Price {
        input_usd_micros_per_mtok: 1_000_000,
        ..troupekit_providers::Price::FREE
    };
    let text = config_error(
        Graph::new("g")
            .agent("draft", priced(troupekit_providers::Price::FREE))
            .agent("again", priced(paid)),
    )
    .await;
    assert_eq!(
        text,
        "agent `graph:g`: two different agents are named `drafter`"
    );
    let built = Graph::new("g")
        .agent("draft", priced(paid))
        .agent("again", priced(paid))
        .build()
        .await;
    assert!(built.is_ok());
}

#[tokio::test]
async fn agent_limits_and_laws_belong_to_the_graph() {
    let limited = drafter().limits(troupekit_agent::Limits {
        run: Some(10),
        ..Default::default()
    });
    let text = config_error(Graph::new("g").agent("draft", limited)).await;
    assert_eq!(
        text,
        "agent `graph:g`: agent node `draft` sets its own limits; set them on the graph"
    );
    let lawful = drafter().law("def main = 1");
    let text = config_error(Graph::new("g").agent("draft", lawful)).await;
    assert_eq!(
        text,
        "agent `graph:g`: agent node `draft` sets its own laws; set them on the graph"
    );
}

#[tokio::test]
async fn finished_run_resumes_to_its_output() {
    let store = Arc::new(troupekit_core::inmem::InMemCheckpoints::default());
    let built = Graph::new("lab3")
        .step("clean", clean_text)
        .checkpoints(Storage::custom(store))
        .build()
        .await
        .unwrap();
    let report = built.run_with(" ab ").await.unwrap();
    assert_eq!(built.resume(report.run_id()).await.unwrap(), json!("ab"));
    assert_eq!(built.resume(report.run_id()).await.unwrap(), json!("ab"));
}

#[tokio::test]
async fn resume_with_a_changed_graph_is_refused() {
    let store = Arc::new(troupekit_core::inmem::InMemCheckpoints::default());
    let first = Graph::new("lab3")
        .step("clean", clean_text)
        .step("count", count_chars)
        .checkpoints(Storage::custom(store.clone()))
        .build()
        .await
        .unwrap();
    let report = first.run_with(" ab ").await.unwrap();
    let renamed = Graph::new("lab3")
        .step("clean", clean_text)
        .step("total", count_chars)
        .checkpoints(Storage::custom(store))
        .build()
        .await
        .unwrap();
    let err = renamed.resume(report.run_id()).await.unwrap_err();
    assert_eq!(
        err.to_string(),
        format!(
            "agent `graph:lab3`: run {} was started by a different workflow; resume it with the graph that started it",
            report.run_id()
        )
    );
}

#[tokio::test]
async fn cycle_is_refused() {
    let text = config_error(
        Graph::new("g")
            .step("clean", clean_text)
            .step("count", count_chars)
            .edge("clean", "count")
            .edge("count", "clean"),
    )
    .await;
    assert_eq!(text, "agent `graph:g`: edges form a cycle");
}

#[tokio::test]
async fn unknown_or_unchained_nodes_are_refused() {
    let text = config_error(
        Graph::new("g")
            .step("clean", clean_text)
            .edge("clean", "nowhere"),
    )
    .await;
    assert_eq!(text, "agent `graph:g`: edge names unknown node `nowhere`");
    let text = config_error(
        Graph::new("g")
            .step("clean", clean_text)
            .step("count", count_chars)
            .step("lone", clean_text)
            .edge("clean", "count"),
    )
    .await;
    assert_eq!(text, "agent `graph:g`: node `lone` is not on the chain");
}

#[tokio::test]
async fn duplicate_node_id_is_refused() {
    let text = config_error(
        Graph::new("g")
            .step("clean", clean_text)
            .step("clean", clean_text),
    )
    .await;
    assert_eq!(text, "agent `graph:g`: duplicate node id `clean`");
    for reserved in ["input", "end"] {
        let text = config_error(Graph::new("g").step(reserved, clean_text)).await;
        assert_eq!(
            text,
            format!("agent `graph:g`: node id `{reserved}` is reserved")
        );
    }
    let text = config_error(Graph::new("g")).await;
    assert_eq!(text, "agent `graph:g`: needs at least one node");
}

#[tokio::test]
async fn same_tool_twice_is_shared_different_tool_same_name_is_refused() {
    let model = shouting_model();
    let caller = |name: &str| {
        Agent::new(name)
            .instructions("Call shout.")
            .model("fake/m")
            .model_client(model.clone())
            .tool(shout)
    };
    let output = Graph::new("twice")
        .agent("text", caller("first"))
        .agent("again", caller("second"))
        .run("go")
        .await
        .unwrap();
    assert_eq!(output, json!("got HI"));
    let impostor = tool("shout", "Not the same tool.", |a: AddArgs| async move {
        a.a + a.b
    });
    let other = Agent::new("other")
        .instructions("x")
        .model("fake/m")
        .model_client(model.clone())
        .tool(impostor);
    let text = config_error(
        Graph::new("twice")
            .agent("text", caller("first"))
            .agent("again", other),
    )
    .await;
    assert_eq!(
        text,
        "agent `graph:twice`: tool `shout` is registered twice with different specs"
    );
}

#[tokio::test]
async fn equal_spec_tools_with_different_state_are_refused() {
    let model = shouting_model();
    let tenant = |name: &str, offset: i64| {
        let search = tool("search", "Search.", move |a: AddArgs| async move {
            a.a + a.b + offset
        });
        Agent::new(name)
            .instructions("Search.")
            .model("fake/m")
            .model_client(model.clone())
            .tool(search)
    };
    let text = config_error(
        Graph::new("g")
            .agent("text", tenant("first", 1))
            .agent("again", tenant("second", 2)),
    )
    .await;
    assert_eq!(
        text,
        "agent `graph:g`: tool `search` has two different implementations; pass one shared tool to both agents"
    );
    let shared: Arc<dyn troupekit::Tool> =
        Arc::new(tool("search", "Search.", |a: AddArgs| async move { a.a }));
    let caller = |name: &str| {
        Agent::new(name)
            .instructions("Search.")
            .model("fake/m")
            .model_client(model.clone())
            .tool_arc(shared.clone())
    };
    let built = Graph::new("g")
        .agent("text", caller("first"))
        .agent("again", caller("second"))
        .build()
        .await;
    assert!(built.is_ok());
}

#[cfg(feature = "sqlite")]
#[tokio::test]
async fn sqlite_checkpoints_allow_graph_resume() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("graph.db");
    let built = Graph::new("lab3")
        .step("clean", clean_text)
        .step("count", count_chars)
        .checkpoints(Storage::sqlite(&path))
        .build()
        .await
        .unwrap();
    let report = built.run_with(" abcd ").await.unwrap();
    assert_eq!(built.resume(report.run_id()).await.unwrap(), json!("4"));
}

#[tokio::test]
async fn graph_laws_are_stored_and_validated() {
    let built = Graph::new("g")
        .step("clean", clean_text)
        .law("def main = 1")
        .auto_laws(false)
        .build()
        .await
        .unwrap();
    assert_eq!(built.laws().inline(), ["def main = 1".to_string()]);
    assert!(!built.laws().auto());
    let text = config_error(
        Graph::new("g")
            .step("clean", clean_text)
            .laws("missing.bend"),
    )
    .await;
    assert!(
        text.starts_with("agent `graph:g`: law file `missing.bend`"),
        "{text}"
    );
}

#[tokio::test]
async fn defaults_are_the_safe_values() {
    let built = Graph::new("g")
        .step("clean", clean_text)
        .checkpoints(Storage::memory())
        .build()
        .await
        .unwrap();
    assert_eq!(built.workflow().max_retries, 3);
    assert_eq!(built.workflow().max_steps, 200);
    assert!(built.laws().auto());
    let resumed = built.resume(troupekit_core::ids::RunId::new()).await;
    assert!(matches!(resumed, Err(KitError::Config { .. })));
    let text = config_error(Graph::new("g").step("clean", clean_text).retries(51)).await;
    assert_eq!(text, "agent `graph:g`: retries must be at most 50, got 51");
}

#[tokio::test]
async fn workflow_built_by_serializer_not_splicing() {
    let hostile = "a\"\n[[nodes]]\nid = \"x";
    let err = Graph::new("g")
        .step(hostile, clean_text)
        .build()
        .await
        .unwrap_err();
    assert!(matches!(err, KitError::Workflow(_)), "{err}");
    let err = Graph::new("Bad Name")
        .step("clean", clean_text)
        .build()
        .await
        .unwrap_err();
    assert!(matches!(err, KitError::Workflow(_)), "{err}");
}
