use serde_json::json;
use std::sync::Arc;
use troupekit::agent::Limits;
use troupekit::core::event::Usage;
use troupekit::core::message::{ContentPart, Message, Role};
use troupekit::graph::testing::{text_reply, FnModel};
use troupekit::prelude::*;
use troupekit::providers::{CompletionResponse, Model, StopReason};

/// Adds a and b.
#[tool]
async fn add(a: i64, b: i64) -> i64 {
    a + b
}

/// Upper-cases the input text.
#[tool]
async fn upper(input: String) -> String {
    input.to_uppercase()
}

/// Greet the user by name.
#[agent(model = "fake/m")]
fn greeter() {}

/// Use the add tool.
#[agent(model = "fake/m", tools = [add])]
fn calc() {}

/// Answers "Hello! You said: <last message>", like lab 1.
/// Output: a scripted model for `model_client`.
fn echo_model() -> Arc<dyn Model> {
    Arc::new(FnModel::new("fake/m", |req| {
        let asked = req
            .messages
            .last()
            .map(|m| m.text_content())
            .unwrap_or_default();
        text_reply(&format!("Hello! You said: {asked}"), 10)
    }))
}

/// Calls add(2, 40) once, then answers with the tool result, like lab 2.
/// Output: a scripted model for `model_client`.
fn calc_model() -> Arc<dyn Model> {
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
            Some(sum) => text_reply(&format!("The sum is {sum}"), 10),
            None => Ok(CompletionResponse {
                message: Message::new(
                    Role::Assistant,
                    vec![ContentPart::ToolCall {
                        id: "c1".into(),
                        name: "add".into(),
                        arguments: json!({"a": 2, "b": 40}),
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

#[tokio::test]
async fn agent_macro_runs_through_the_facade() {
    let outcome = greeter()
        .model_client(echo_model())
        .limits(Limits::default())
        .run_with("I am Ada")
        .await
        .unwrap();
    assert_eq!(outcome.text(), "Hello! You said: I am Ada");
    assert_eq!(outcome.model_calls(), 1);
}

#[tokio::test]
async fn tool_macro_runs_through_the_facade() {
    let text = calc()
        .model_client(calc_model())
        .run("What is 2 + 40?")
        .await
        .unwrap();
    assert_eq!(text, "The sum is 42");
}

#[tokio::test]
async fn graph_and_storage_come_from_the_prelude() {
    let built: BuiltGraph = Graph::new("pipeline")
        .step("loud", upper)
        .checkpoints(Storage::memory())
        .build()
        .await
        .unwrap();
    assert_eq!(built.run("hi").await.unwrap(), json!("HI"));
}
