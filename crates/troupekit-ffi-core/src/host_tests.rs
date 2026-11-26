use crate::host::{wrap_tool, HostTool, MAX_TOOL_OUTPUT_BYTES};
use crate::records::HostError;
use serde_json::json;
use std::sync::{Arc, Mutex};
use troupekit_tools::ToolContext;

struct Echo {
    seen: Mutex<Vec<String>>,
    reply: Result<String, String>,
}

impl HostTool for Echo {
    fn name(&self) -> String {
        "echo".into()
    }

    fn description(&self) -> String {
        "Echoes its arguments.".into()
    }

    fn schema_json(&self) -> String {
        r#"{"type":"object","properties":{"text":{"type":"string"}}}"#.into()
    }

    fn call(&self, args_json: String) -> Result<String, HostError> {
        self.seen.lock().unwrap().push(args_json);
        self.reply.clone().map_err(HostError::new)
    }
}

fn echo(reply: Result<String, String>) -> Arc<Echo> {
    Arc::new(Echo {
        seen: Mutex::new(Vec::new()),
        reply,
    })
}

#[tokio::test]
async fn host_tool_called_with_json_args() {
    let host = echo(Ok("done".into()));
    let tool = wrap_tool(host.clone()).unwrap();
    assert_eq!(tool.spec().name, "echo");
    assert_eq!(tool.spec().schema["type"], "object");
    let out = tool
        .call(json!({"text": "hi"}), &ToolContext::default())
        .await
        .unwrap();
    assert_eq!(out.content, "done");
    assert!(!out.is_error);
    let seen = host.seen.lock().unwrap().clone();
    assert_eq!(seen.len(), 1);
    let args: serde_json::Value = serde_json::from_str(&seen[0]).unwrap();
    assert_eq!(args, json!({"text": "hi"}));
}

#[tokio::test]
async fn host_tool_error_reaches_model_as_error_output() {
    let tool = wrap_tool(echo(Err("order not found".into()))).unwrap();
    let out = tool.call(json!({}), &ToolContext::default()).await.unwrap();
    assert!(out.is_error);
    assert_eq!(out.content, "order not found");
}

#[tokio::test]
async fn host_tool_output_over_cap_is_error() {
    let big = "x".repeat(MAX_TOOL_OUTPUT_BYTES + 1);
    let tool = wrap_tool(echo(Ok(big))).unwrap();
    let out = tool.call(json!({}), &ToolContext::default()).await.unwrap();
    assert!(out.is_error);
    assert_eq!(out.content, "tool output over 1 MiB");
    let at_cap = "x".repeat(MAX_TOOL_OUTPUT_BYTES);
    let tool = wrap_tool(echo(Ok(at_cap))).unwrap();
    let out = tool.call(json!({}), &ToolContext::default()).await.unwrap();
    assert!(!out.is_error);
}

struct Panics;

impl HostTool for Panics {
    fn name(&self) -> String {
        "boom".into()
    }

    fn description(&self) -> String {
        "Panics.".into()
    }

    fn schema_json(&self) -> String {
        r#"{"type":"object"}"#.into()
    }

    fn call(&self, _args_json: String) -> Result<String, HostError> {
        panic!("adapter bug");
    }
}

#[tokio::test]
async fn host_tool_panic_reaches_model_as_error_output() {
    let tool = wrap_tool(Arc::new(Panics)).unwrap();
    let out = tool.call(json!({}), &ToolContext::default()).await.unwrap();
    assert!(out.is_error);
    assert_eq!(out.content, "host tool `boom` panicked");
}
