use std::sync::{Arc, Mutex};
use troupekit_ffi_core::{
    AgentBuilder, FfiEventKind, HostError, HostModel, HostModelCaps, HostReply, HostRequest,
    HostTool, TokenSink,
};

struct Scripted {
    caps: HostModelCaps,
    replies: Mutex<Vec<(Vec<String>, HostReply)>>,
    seen: Mutex<Vec<HostRequest>>,
}

impl HostModel for Scripted {
    fn id(&self) -> String {
        "host/scripted".into()
    }

    fn capabilities(&self) -> HostModelCaps {
        self.caps.clone()
    }

    fn generate(&self, request: HostRequest, sink: Arc<TokenSink>) -> Result<HostReply, HostError> {
        self.seen.lock().unwrap().push(request);
        let mut replies = self.replies.lock().unwrap();
        if replies.is_empty() {
            return Err(HostError::new("script ran out"));
        }
        let (deltas, reply) = replies.remove(0);
        for delta in deltas {
            sink.delta(delta);
        }
        Ok(reply)
    }
}

fn reply(text: &str) -> HostReply {
    let mut reply = HostReply::default();
    reply.text = text.into();
    reply.input_tokens = 10;
    reply.output_tokens = 2;
    reply
}

fn scripted(caps: HostModelCaps, replies: Vec<(Vec<String>, HostReply)>) -> Arc<Scripted> {
    Arc::new(Scripted {
        caps,
        replies: Mutex::new(replies),
        seen: Mutex::new(Vec::new()),
    })
}

struct Add;

impl HostTool for Add {
    fn name(&self) -> String {
        "add".into()
    }

    fn description(&self) -> String {
        "Adds a and b.".into()
    }

    fn schema_json(&self) -> String {
        r#"{"type":"object","properties":{"a":{"type":"number"},"b":{"type":"number"}},"required":["a","b"]}"#.into()
    }

    fn call(&self, args_json: String) -> Result<String, HostError> {
        let args: serde_json::Value = serde_json::from_str(&args_json).unwrap();
        let sum = args["a"].as_f64().unwrap() + args["b"].as_f64().unwrap();
        Ok(format!("{sum}"))
    }
}

fn agent(model: Arc<Scripted>) -> Arc<AgentBuilder> {
    AgentBuilder::new("calc".into())
        .instructions("Add numbers with the add tool.".into())
        .host_model(model)
        .tool(Arc::new(Add))
}

#[tokio::test]
async fn emulated_tool_call_parses() {
    let model = scripted(
        HostModelCaps::default(),
        vec![
            (
                vec![],
                reply(r#"{"tool": "add", "arguments": {"a": 1, "b": 2}}"#),
            ),
            (vec![], reply(r#"{"answer": "3"}"#)),
        ],
    );
    let handle = agent(model.clone()).build().await.unwrap();
    let result = handle.run_outcome("1 + 2".into()).await.unwrap();
    assert_eq!(result.text, "3");
    assert_eq!(result.usage.input_tokens, 20);
    assert_eq!(result.usage.cost_usd_micros, 0);
    let seen = model.seen.lock().unwrap().clone();
    assert_eq!(seen.len(), 2);
    assert!(seen[0].tools_json.is_none());
    assert!(seen[0].schema_json.is_none());
    let system = seen[0].system.clone().unwrap();
    assert!(system.contains("add"), "{system}");
    assert!(system.contains(r#""answer""#), "{system}");
    let last = seen[1].messages.last().unwrap();
    assert_eq!(last.role, "user");
    assert!(last.text.contains('3'), "{last:?}");
}

#[tokio::test]
async fn emulated_tool_call_uses_schema_when_supported() {
    let mut caps = HostModelCaps::default();
    caps.json_schema = true;
    let model = scripted(
        caps,
        vec![
            (
                vec![],
                reply("```json\n{\"tool\": \"add\", \"arguments\": {\"a\": 2, \"b\": 2}}\n```"),
            ),
            (vec![], reply(r#"{"answer": "4"}"#)),
        ],
    );
    let handle = agent(model.clone()).build().await.unwrap();
    assert_eq!(handle.run("2 + 2".into()).await.unwrap(), "4");
    let seen = model.seen.lock().unwrap().clone();
    let schema: serde_json::Value =
        serde_json::from_str(seen[0].schema_json.as_deref().unwrap()).unwrap();
    assert_eq!(schema["oneOf"].as_array().unwrap().len(), 2);
}

#[tokio::test]
async fn native_tool_call_passes_through() {
    let mut caps = HostModelCaps::default();
    caps.tools = true;
    let mut call = reply("");
    call.tool_calls_json =
        Some(r#"[{"id": "c1", "name": "add", "arguments": {"a": 1, "b": 2}}]"#.into());
    let model = scripted(caps, vec![(vec![], call), (vec![], reply("3"))]);
    let handle = agent(model.clone()).build().await.unwrap();
    assert_eq!(handle.run("1 + 2".into()).await.unwrap(), "3");
    let seen = model.seen.lock().unwrap().clone();
    let tools: serde_json::Value =
        serde_json::from_str(seen[0].tools_json.as_deref().unwrap()).unwrap();
    assert_eq!(tools[0]["name"], "add");
    assert_eq!(tools[0]["schema"]["type"], "object");
    let messages = &seen[1].messages;
    let asked = messages.iter().find(|m| m.role == "assistant").unwrap();
    let calls: serde_json::Value =
        serde_json::from_str(asked.tool_calls_json.as_deref().unwrap()).unwrap();
    assert_eq!(calls[0]["id"], "c1");
    let answered = messages.last().unwrap();
    assert_eq!(answered.role, "tool");
    assert_eq!(answered.tool_call_id.as_deref(), Some("c1"));
    assert_eq!(answered.text, "3");
}

#[tokio::test]
async fn unparseable_reply_gets_one_repair() {
    let model = scripted(
        HostModelCaps::default(),
        vec![
            (vec![], reply("I think the answer is 3")),
            (vec![], reply(r#"{"answer": "3"}"#)),
        ],
    );
    let handle = agent(model.clone()).build().await.unwrap();
    assert_eq!(handle.run("1 + 2".into()).await.unwrap(), "3");
    let seen = model.seen.lock().unwrap().clone();
    assert_eq!(seen.len(), 2);
    assert_eq!(seen[1].messages.len(), seen[0].messages.len() + 2);

    let model = scripted(
        HostModelCaps::default(),
        vec![(vec![], reply("three")), (vec![], reply("still three"))],
    );
    let handle = agent(model.clone()).build().await.unwrap();
    let err = handle.run("1 + 2".into()).await.unwrap_err();
    assert!(err.message.contains("JSON"), "{err}");
    assert_eq!(model.seen.lock().unwrap().len(), 2);
}

#[tokio::test]
async fn deltas_reach_event_stream() {
    let mut caps = HostModelCaps::default();
    caps.streaming = true;
    let deltas = vec!["hel".to_string(), "lo".to_string()];
    let model = scripted(caps, vec![(deltas, reply("hello"))]);
    let handle = AgentBuilder::new("greeter".into())
        .instructions("Say hello.".into())
        .host_model(model)
        .build()
        .await
        .unwrap();
    let stream = handle.stream("hi".into()).await.unwrap();
    let mut text = String::new();
    while let Some(event) = stream.next().await.unwrap() {
        if let FfiEventKind::TextDelta { text: delta } = event.kind {
            text.push_str(&delta);
        }
    }
    assert_eq!(text, "hello");
    assert_eq!(stream.result().await.unwrap().text, "hello");
}

#[tokio::test]
async fn emulated_envelope_deltas_are_not_streamed() {
    let mut caps = HostModelCaps::default();
    caps.streaming = true;
    let envelope = r#"{"answer": "3"}"#;
    let model = scripted(caps, vec![(vec![envelope.to_string()], reply(envelope))]);
    let handle = agent(model).build().await.unwrap();
    let stream = handle.stream("1 + 2".into()).await.unwrap();
    let mut deltas = Vec::new();
    while let Some(event) = stream.next().await.unwrap() {
        if let FfiEventKind::TextDelta { text } = event.kind {
            deltas.push(text);
        }
    }
    assert_eq!(deltas, vec!["3".to_string()]);
}

#[tokio::test]
async fn host_error_fails_the_run_without_retry() {
    let model = scripted(HostModelCaps::default(), vec![]);
    let handle = AgentBuilder::new("greeter".into())
        .instructions("Say hello.".into())
        .host_model(model.clone())
        .build()
        .await
        .unwrap();
    let err = handle.run("hi".into()).await.unwrap_err();
    assert!(err.message.contains("script ran out"), "{err}");
    assert_eq!(model.seen.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn priced_host_model_charges_its_price() {
    let model = scripted(HostModelCaps::default(), vec![(vec![], reply("hi"))]);
    let handle = AgentBuilder::new("greeter".into())
        .instructions("Say hello.".into())
        .host_model_priced(model, 1_000_000, 1_000_000)
        .build()
        .await
        .unwrap();
    let result = handle.run_outcome("hi".into()).await.unwrap();
    assert_eq!(result.usage.cost_usd_micros, 12);
}
