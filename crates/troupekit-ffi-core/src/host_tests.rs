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

#[cfg(feature = "knowledge")]
mod embedder {
    use crate::host::{wrap_embedder, HostEmbedder, MAX_EMBED_BATCH};
    use crate::records::HostError;
    use crate::ErrorCode;
    use std::sync::{Arc, Mutex};
    use troupekit_context::embed::embed_checked;

    struct Shaped {
        dims: u32,
        batches: Mutex<Vec<usize>>,
        shape: fn(usize, u32) -> Vec<Vec<f32>>,
    }

    impl HostEmbedder for Shaped {
        fn dims(&self) -> u32 {
            self.dims
        }

        fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, HostError> {
            self.batches.lock().unwrap().push(texts.len());
            Ok((self.shape)(texts.len(), self.dims))
        }
    }

    fn shaped(dims: u32, shape: fn(usize, u32) -> Vec<Vec<f32>>) -> Arc<Shaped> {
        Arc::new(Shaped {
            dims,
            batches: Mutex::new(Vec::new()),
            shape,
        })
    }

    fn good(count: usize, dims: u32) -> Vec<Vec<f32>> {
        vec![vec![0.5; dims as usize]; count]
    }

    fn texts(count: usize) -> Vec<String> {
        (0..count).map(|i| format!("text {i}")).collect()
    }

    #[tokio::test]
    async fn embedder_batches_at_256() {
        let host = shaped(3, good);
        let embedder = wrap_embedder(host.clone()).unwrap();
        assert_eq!(embedder.dims(), 3);
        let vectors = embed_checked(&*embedder, &texts(600)).await.unwrap();
        assert_eq!(vectors.len(), 600);
        assert_eq!(MAX_EMBED_BATCH, 256);
        assert_eq!(*host.batches.lock().unwrap(), vec![256, 256, 88]);
    }

    #[tokio::test]
    async fn embedder_shape_checked() {
        let short = wrap_embedder(shaped(3, |count, dims| good(count - 1, dims))).unwrap();
        assert!(embed_checked(&*short, &texts(2)).await.is_err());
        let narrow = wrap_embedder(shaped(3, |count, _| good(count, 2))).unwrap();
        assert!(embed_checked(&*narrow, &texts(2)).await.is_err());
        let nan = wrap_embedder(shaped(2, |count, _| vec![vec![f32::NAN, 0.0]; count])).unwrap();
        assert!(embed_checked(&*nan, &texts(1)).await.is_err());
    }

    #[test]
    fn embedder_zero_dims_refused() {
        let err = wrap_embedder(shaped(0, good)).err().unwrap();
        assert_eq!(err.code, ErrorCode::InvalidArgument);
    }
}
