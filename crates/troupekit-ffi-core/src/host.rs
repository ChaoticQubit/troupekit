use crate::error::{ErrorCode, FfiError};
use crate::records::{
    FfiEvent, HostError, HostHeader, HostHttpRequest, HostModelCaps, HostReply, HostRequest,
};
use async_trait::async_trait;
use serde_json::Value;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Arc;
use troupekit_core::tool::ToolSpec;
use troupekit_tools::{Tool, ToolContext, ToolError, ToolOutput};

pub(crate) const MAX_TOOL_NAME_BYTES: usize = 64;
pub(crate) const MAX_TOOL_DESCRIPTION_BYTES: usize = 16 * 1024;
pub(crate) const MAX_TOOL_OUTPUT_BYTES: usize = 1024 * 1024;

/// Receives every event of a stream as it happens, as the callback form
/// of `RunStream::next`. Called on one thread per stream,
/// `troupekit-listener`, never the caller's; a slow listener only
/// slows its own delivery, and the buffer drops text deltas meanwhile.
pub trait EventListener: Send + Sync {
    /// Handles one `event`, in `seq` order within its run id.
    /// Input: the event. Output: none; a panic here is caught and
    /// stops only this listener.
    fn on_event(&self, event: FfiEvent);
}

/// A tool written in the host language: a name the model calls it by,
/// a description, a JSON Schema for its arguments and the call itself.
/// troupekit checks name and schema when it is added.
pub trait HostTool: Send + Sync {
    /// Returns the tool name: 1 to 64 bytes of `[a-z0-9_]`.
    /// Called once when the tool is added.
    /// Output: the name the model uses.
    fn name(&self) -> String;

    /// Returns what the tool does, for the model, at most 16 KiB.
    /// Called once when the tool is added.
    /// Output: the description text.
    fn description(&self) -> String;

    /// Returns the JSON Schema of the arguments as JSON object text.
    /// Called once when the tool is added; text that is not a JSON
    /// object is refused with `InvalidArgument`.
    fn schema_json(&self) -> String;

    /// Runs the tool on `args_json`, arguments that match the schema.
    /// Output: the result text, at most 1 MiB, or a `HostError` the
    /// model sees as an error result. Runs on a blocking worker thread.
    fn call(&self, args_json: String) -> Result<String, HostError>;
}

/// A model run by the host, such as the phone's built-in model.
/// troupekit renders each request for the model's `capabilities` and
/// parses tool calls out of plain text when it has no native tools.
pub trait HostModel: Send + Sync {
    /// Returns the model id, for example `apple/foundation`.
    /// Used in events, errors and the catalog key.
    /// Output: a non-empty id.
    fn id(&self) -> String;

    /// Returns what the model supports.
    /// Read once per request, so it may change when the device does.
    /// Output: the capabilities record.
    fn capabilities(&self) -> HostModelCaps;

    /// Answers `request`, pushing text deltas to `sink` when streaming.
    /// Output: the whole reply with token counts, or a `HostError`.
    /// Runs on a blocking worker thread.
    fn generate(&self, request: HostRequest, sink: Arc<TokenSink>) -> Result<HostReply, HostError>;
}

/// An embedding model run by the host, for knowledge search.
/// Every vector must have `dims` numbers; troupekit checks the shape
/// and sends at most 256 texts per call.
#[cfg(feature = "knowledge")]
pub trait HostEmbedder: Send + Sync {
    /// Returns the length of every vector `embed` returns.
    /// Read once at build.
    /// Output: a dimension count above zero.
    fn dims(&self) -> u32;

    /// Embeds `texts`, one vector per text, in order.
    /// Output: the vectors, or a `HostError`.
    /// Runs on a blocking worker thread.
    fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, HostError>;
}

/// An HTTP client in the host language, used only for model and
/// embedding calls to the resolved vendor. The host reports the
/// response through `HttpSink`: `head` once, then `chunk` per part.
pub trait HostTransport: Send + Sync {
    /// Sends `request` and reports the response to `sink` before it
    /// returns. Output: `Ok` once the body is done, or a `HostError`
    /// for a network failure. Runs on a blocking worker thread.
    fn send(&self, request: HostHttpRequest, sink: Arc<HttpSink>) -> Result<(), HostError>;
}

/// Where a host model pushes text deltas while it generates.
/// Each delta reaches the run's stream as a `TextDelta` event.
/// Built by troupekit for each `HostModel::generate` call.
#[derive(Debug)]
pub struct TokenSink {
    _private: (),
}

impl TokenSink {
    /// Pushes one text delta of the reply in progress.
    /// Input: the new text only, not the text so far.
    /// Output: none; deltas past the reply cap are dropped.
    pub fn delta(&self, _text: String) {}
}

/// Where a host transport reports an HTTP response: `head` once with
/// the status, the final URL and the headers, then `chunk` for each
/// part of the body. Built by troupekit for each `send` call.
#[derive(Debug)]
pub struct HttpSink {
    _private: (),
}

impl HttpSink {
    /// Reports the response head: `status`, the final `url` after any
    /// redirect, and the `headers`. Call it once, before any chunk.
    /// Output: none.
    pub fn head(&self, _status: u16, _url: String, _headers: Vec<HostHeader>) {}

    /// Reports the next part of the response body.
    /// Input: the bytes, in order. Output: none; it blocks while the
    /// reader is behind, and bytes past the reply cap end the body.
    pub fn chunk(&self, _bytes: Vec<u8>) {}
}

/// Checks host tool `host` and wraps it as a kit tool: name 1 to 64
/// bytes of `[a-z0-9_]`, description at most 16 KiB, schema a JSON
/// object. Output: the tool, or `InvalidArgument` naming the bad part.
pub(crate) fn wrap_tool(host: Arc<dyn HostTool>) -> Result<Arc<dyn Tool>, FfiError> {
    let name = host.name();
    let name_ok = !name.is_empty()
        && name.len() <= MAX_TOOL_NAME_BYTES
        && name
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_');
    if !name_ok {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "host tool name must be 1 to 64 bytes of [a-z0-9_]",
        ));
    }
    let description = host.description();
    if description.len() > MAX_TOOL_DESCRIPTION_BYTES {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            format!("host tool `{name}` description is over 16 KiB"),
        ));
    }
    let schema_text = host.schema_json();
    crate::runtime::check_input("host tool schema", &schema_text)?;
    let schema = match serde_json::from_str::<Value>(&schema_text) {
        Ok(schema @ Value::Object(_)) => schema,
        _ => {
            return Err(FfiError::new(
                ErrorCode::InvalidArgument,
                format!("host tool `{name}` schema must be a JSON object"),
            ))
        }
    };
    assert!(schema.is_object());
    let spec = ToolSpec {
        name,
        description,
        schema,
        hosted: None,
    };
    Ok(Arc::new(HostToolAdapter { host, spec }))
}

struct HostToolAdapter {
    host: Arc<dyn HostTool>,
    spec: ToolSpec,
}

#[async_trait]
impl Tool for HostToolAdapter {
    fn spec(&self) -> &ToolSpec {
        &self.spec
    }

    async fn call(&self, args: Value, _ctx: &ToolContext) -> Result<ToolOutput, ToolError> {
        let host = self.host.clone();
        let args_json = args.to_string();
        let called = tokio::task::spawn_blocking(move || {
            catch_unwind(AssertUnwindSafe(|| host.call(args_json)))
        })
        .await;
        let panicked = || ToolOutput::error(format!("host tool `{}` panicked", self.spec.name));
        let output = match called {
            Ok(Ok(Ok(text))) => ToolOutput::text(text),
            Ok(Ok(Err(error))) => ToolOutput::error(error.message),
            Ok(Err(_)) | Err(_) => panicked(),
        };
        if output.content.len() > MAX_TOOL_OUTPUT_BYTES {
            return Ok(ToolOutput::error("tool output over 1 MiB"));
        }
        assert!(output.content.len() <= MAX_TOOL_OUTPUT_BYTES);
        Ok(output)
    }
}
