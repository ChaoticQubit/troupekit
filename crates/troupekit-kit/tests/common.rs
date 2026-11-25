use serde::Deserialize;
use serde_json::json;
use std::sync::Arc;
use troupekit::tool;
use troupekit_core::error::{ProviderError, ProviderErrorKind};
use troupekit_core::event::Usage;
use troupekit_core::message::{ContentPart, Message, Role};
use troupekit_core::secret::Secret;
use troupekit_graph::testing::{text_reply, FnModel};
use troupekit_providers::transport::{Scripted, ScriptedTransport};
use troupekit_providers::{CompletionResponse, Model, StopReason};
use troupekit_tools::Tool;

pub const SSE_HI: &str = concat!(
    "data: {\"id\":\"c\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"\"},\"finish_reason\":null}],\"usage\":null}\n\n",
    "data: {\"id\":\"c\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hi there\"},\"finish_reason\":null}],\"usage\":null}\n\n",
    "data: {\"id\":\"c\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":null}\n\n",
    "data: {\"id\":\"c\",\"choices\":[],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":2,\"total_tokens\":5}}\n\n",
    "data: [DONE]\n\n",
);

#[derive(Deserialize, schemars::JsonSchema)]
pub struct AddArgs {
    pub a: i64,
    pub b: i64,
}

pub fn add_tool() -> impl Tool + 'static {
    tool("add", "Adds a and b", |x: AddArgs| async move { x.a + x.b })
}

pub fn echo_model() -> Arc<dyn Model> {
    Arc::new(FnModel::new("fake/m", |req| {
        let asked = req
            .messages
            .last()
            .map(|m| m.text_content())
            .unwrap_or_default();
        text_reply(&format!("Hello! You said: {asked}"), 10)
    }))
}

pub fn calc_model() -> Arc<dyn Model> {
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

pub fn failing_model(kind: ProviderErrorKind) -> Arc<FnModel> {
    Arc::new(FnModel::new("fake/m", move |_req| {
        Err(ProviderError::new(kind, "busy").with_retry_after(1))
    }))
}

pub fn scripted() -> Arc<ScriptedTransport> {
    Arc::new(ScriptedTransport::new(vec![Scripted::body(200, SSE_HI)]))
}

pub fn secret(value: &str) -> Secret {
    Secret::new(value.into()).unwrap()
}

pub fn greeter() -> troupekit::Agent {
    troupekit::Agent::new("greeter")
        .instructions("Greet the user.")
        .model("fake/m")
        .model_client(echo_model())
}
