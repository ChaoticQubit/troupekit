use std::sync::Arc;
use troupekit_ffi_core::{
    AgentBuilder, ErrorCode, FfiEventKind, GraphBuilder, HostError, HostModel, HostModelCaps,
    HostReply, HostRequest, HostTool, TokenSink,
};

struct Says(&'static str);

impl HostModel for Says {
    fn id(&self) -> String {
        "host/says".into()
    }

    fn capabilities(&self) -> HostModelCaps {
        HostModelCaps::default()
    }

    fn generate(
        &self,
        _request: HostRequest,
        _sink: Arc<TokenSink>,
    ) -> Result<HostReply, HostError> {
        let mut reply = HostReply::default();
        reply.text = self.0.into();
        reply.input_tokens = 5;
        reply.output_tokens = 1;
        Ok(reply)
    }
}

fn says(name: &str, text: &'static str) -> Arc<AgentBuilder> {
    AgentBuilder::new(name.into())
        .instructions("Reply.".into())
        .host_model(Arc::new(Says(text)))
}

struct Shout;

impl HostTool for Shout {
    fn name(&self) -> String {
        "shout".into()
    }

    fn description(&self) -> String {
        "Upper-cases the writer's text.".into()
    }

    fn schema_json(&self) -> String {
        r#"{"type":"object","properties":{"writer":{"type":"string"}},"required":["writer"]}"#
            .into()
    }

    fn call(&self, args_json: String) -> Result<String, HostError> {
        let args: serde_json::Value = serde_json::from_str(&args_json).unwrap();
        Ok(args["writer"].as_str().unwrap().to_uppercase())
    }
}

fn pipeline() -> Arc<GraphBuilder> {
    GraphBuilder::new("pipeline".into())
        .agent("writer".into(), says("writer", "draft"))
        .agent("editor".into(), says("editor", "done"))
        .edge("writer".into(), "editor".into())
}

#[tokio::test]
async fn graph_of_host_model_agents_runs() {
    let handle = pipeline().build().await.unwrap();
    assert_eq!(handle.run("go".into()).await.unwrap(), "\"done\"");
    let result = handle.run_result("go".into()).await.unwrap();
    assert_eq!(result.output_json, "\"done\"");
    assert!(result.steps >= 2, "{}", result.steps);
    let state: serde_json::Value = serde_json::from_str(&result.state_json).unwrap();
    assert_eq!(state["writer"], "draft");
    assert_eq!(result.run_id.len(), 36);
}

#[tokio::test]
async fn graph_step_calls_host_tool() {
    let handle = GraphBuilder::new("loud".into())
        .agent("writer".into(), says("writer", "draft"))
        .step("shout".into(), Arc::new(Shout))
        .build()
        .await
        .unwrap();
    let output = handle.run("go".into()).await.unwrap();
    assert!(output.contains("DRAFT"), "{output}");
}

#[tokio::test]
async fn agent_builder_is_taken_by_the_graph() {
    let writer = says("writer", "draft");
    GraphBuilder::new("one".into())
        .agent("writer".into(), writer.clone())
        .build()
        .await
        .unwrap();
    let err = writer.build().await.unwrap_err();
    assert_eq!(err.code, ErrorCode::InvalidArgument);
}

#[tokio::test]
async fn graph_builder_used_after_build_is_refused() {
    let builder = pipeline();
    builder.build().await.unwrap();
    let err = builder.build().await.unwrap_err();
    assert_eq!(err.code, ErrorCode::InvalidArgument);
}

#[tokio::test]
async fn graph_stream_yields_events_then_result() {
    let handle = pipeline().build().await.unwrap();
    let stream = handle.stream("go".into()).await.unwrap();
    let mut finished = 0;
    while let Some(event) = stream.next().await.unwrap() {
        if matches!(event.kind, FfiEventKind::RunFinished { .. }) {
            finished += 1;
        }
    }
    assert!(finished >= 1);
    let result = stream.result().await.unwrap();
    assert_eq!(result.text, "done");
    assert_eq!(result.output_json, "\"done\"");
}

#[tokio::test]
async fn graph_bad_run_id_is_invalid_argument() {
    let handle = pipeline().build().await.unwrap();
    let err = handle.resume("not-a-run-id".into()).await.unwrap_err();
    assert_eq!(err.code, ErrorCode::InvalidArgument);
}

#[tokio::test]
async fn graph_input_over_cap_is_refused() {
    let handle = pipeline().build().await.unwrap();
    let cap = troupekit_ffi_core::RuntimeSettings::default().max_input_bytes;
    let err = handle.run("x".repeat(cap + 1)).await.unwrap_err();
    assert_eq!(err.code, ErrorCode::InvalidArgument);
}

#[cfg(feature = "sqlite")]
#[tokio::test]
async fn graph_start_then_resume() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("graph.db").to_string_lossy().into_owned();
    let handle = pipeline().sqlite(path).build().await.unwrap();
    let run_id = handle.start("go".into()).await.unwrap();
    assert_eq!(handle.resume(run_id).await.unwrap(), "\"done\"");
}

#[tokio::test]
async fn graph_start_without_sqlite_is_config() {
    let handle = pipeline().build().await.unwrap();
    let err = handle.start("go".into()).await.unwrap_err();
    assert_eq!(err.code, ErrorCode::Config);
}
