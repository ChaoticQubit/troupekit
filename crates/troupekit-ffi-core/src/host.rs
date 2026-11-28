use crate::error::{ErrorCode, FfiError};
use crate::records::{
    FfiEvent, HostError, HostHeader, HostHttpRequest, HostModelCaps, HostReply, HostRequest,
};
use crate::runtime::check_input;
use async_trait::async_trait;
use serde_json::Value;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{Arc, Mutex, PoisonError};
use troupekit_core::http::{HttpError, HttpErrorKind};
use troupekit_core::tool::ToolSpec;
use troupekit_tools::{Tool, ToolContext, ToolError, ToolOutput};
use troupekit_trace::{CallbackSink, Trace, TraceSink};

pub(crate) const MAX_TOOL_NAME_BYTES: usize = 64;
pub(crate) const MAX_TOOL_DESCRIPTION_BYTES: usize = 16 * 1024;
pub(crate) const MAX_TOOL_OUTPUT_BYTES: usize = 1024 * 1024;
#[cfg(feature = "knowledge")]
pub(crate) const MAX_EMBED_BATCH: usize = 256;
#[cfg(feature = "knowledge")]
const MAX_EMBED_DIMS: u32 = 65_536;

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

/// Receives every trace record of a traced build or run, as the
/// callback form of `trace_jsonl`. Called on the thread that writes the
/// record, so it should return fast; keys are scrubbed before the call.
pub trait TraceListener: Send + Sync {
    /// Handles one record: one JSON object on one line, no newline.
    /// Input: the record. Output: none; a panic here is caught and
    /// turns this listener off for the rest of the trace.
    fn on_record(&self, line: String);
}

/// Wraps `listener` as a trace sink that catches its panics.
/// Input: the host listener. Output: the sink for `trace_to`.
pub(crate) fn trace_sink(listener: Arc<dyn TraceListener>) -> Arc<dyn TraceSink> {
    Arc::new(CallbackSink::new(move |line: &str| {
        listener.on_record(line.to_owned())
    }))
}

/// Parses trace category names as 0.3 spells them, such as `tools`.
/// Input: the names. Output: the categories, or `InvalidArgument`
/// naming the first unknown name and every known one.
pub(crate) fn trace_categories(names: &[String]) -> Result<Vec<Trace>, FfiError> {
    let mut found = Vec::with_capacity(names.len());
    for name in names {
        check_input("trace", name)?;
        match name.parse::<Trace>() {
            Ok(category) => found.push(category),
            Err(_) => {
                let known: Vec<&str> = Trace::ALL.iter().map(|t| t.as_str()).collect();
                return Err(FfiError::new(
                    ErrorCode::InvalidArgument,
                    format!(
                        "trace: unknown category `{name}`; known: {}",
                        known.join(", ")
                    ),
                ));
            }
        }
    }
    assert_eq!(found.len(), names.len());
    Ok(found)
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

    /// Returns the JSON Schema of the arguments as JSON object text
    /// with `"type": "object"`. Called once when the tool is added; any
    /// other text is refused with `InvalidArgument`.
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
/// embedding calls to the resolved vendor. The host must turn off
/// redirect following and report a 3xx as received; troupekit refuses
/// it. The response goes through `HttpSink`: `head`, then `chunk`s.
pub trait HostTransport: Send + Sync {
    /// Sends `request` and reports the response to `sink` before it
    /// returns. Output: `Ok` once the body is done, or a `HostError`
    /// for a network failure. Runs on a blocking worker thread.
    fn send(&self, request: HostHttpRequest, sink: Arc<HttpSink>) -> Result<(), HostError>;
}

/// Where a host model pushes text deltas while it generates. Each
/// delta reaches the run's stream as a `TextDelta` event, except when
/// troupekit emulates tools: then the parsed answer is sent once.
/// Built by troupekit for each `HostModel::generate` call.
#[derive(Debug)]
pub struct TokenSink {
    deltas: Option<tokio::sync::mpsc::Sender<String>>,
    state: Mutex<SinkText>,
    cap: usize,
}

#[derive(Debug, Default)]
struct SinkText {
    bytes: usize,
    pending: String,
}

impl TokenSink {
    /// Pushes one text delta of the reply in progress. Never blocks.
    /// Input: the new text only, not the text so far. Output: none.
    /// Empty deltas and deltas past the reply cap are dropped; while the
    /// queue is full, deltas are joined and sent as one once it drains.
    pub fn delta(&self, text: String) {
        let Some(deltas) = &self.deltas else {
            return;
        };
        if text.is_empty() {
            return;
        }
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        let total = state.bytes.saturating_add(text.len());
        if total > self.cap {
            return;
        }
        state.bytes = total;
        state.pending.push_str(&text);
        let joined = std::mem::take(&mut state.pending);
        if let Err(tokio::sync::mpsc::error::TrySendError::Full(back)) = deltas.try_send(joined) {
            state.pending = back;
        }
        assert!(state.bytes <= self.cap);
        assert!(state.pending.len() <= state.bytes);
    }

    /// Takes the text still waiting because the queue was full, so the
    /// run can send it after the host returns. Output: that text, or an
    /// empty string.
    pub(crate) fn take_pending(&self) -> String {
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        std::mem::take(&mut state.pending)
    }

    pub(crate) fn channel(
        forward: bool,
        cap: usize,
        queue: usize,
    ) -> (Arc<TokenSink>, tokio::sync::mpsc::Receiver<String>) {
        assert!(queue > 0);
        let (sender, receiver) = tokio::sync::mpsc::channel(queue);
        let sink = TokenSink {
            deltas: forward.then_some(sender),
            state: Mutex::new(SinkText::default()),
            cap,
        };
        (Arc::new(sink), receiver)
    }
}

/// Where a host transport reports an HTTP response: `head` once with
/// the status, the final URL and the headers, then `chunk` for each
/// part of the body. Built by troupekit for each `send` call.
pub struct HttpSink {
    state: Mutex<SinkState>,
    cap: usize,
}

struct SinkState {
    head: Option<tokio::sync::oneshot::Sender<crate::host_transport::Head>>,
    chunks: Option<tokio::sync::mpsc::Sender<Result<Vec<u8>, HttpError>>>,
    bytes: usize,
}

impl std::fmt::Debug for HttpSink {
    /// Writes the byte cap and the bytes taken so far.
    /// The body is not shown.
    /// Output: the debug text.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let bytes = self
            .state
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .bytes;
        f.debug_struct("HttpSink")
            .field("cap", &self.cap)
            .field("bytes", &bytes)
            .finish_non_exhaustive()
    }
}

impl HttpSink {
    pub(crate) fn open(
        head: tokio::sync::oneshot::Sender<crate::host_transport::Head>,
        chunks: tokio::sync::mpsc::Sender<Result<Vec<u8>, HttpError>>,
        cap: usize,
    ) -> HttpSink {
        HttpSink {
            state: Mutex::new(SinkState {
                head: Some(head),
                chunks: Some(chunks),
                bytes: 0,
            }),
            cap,
        }
    }

    pub(crate) fn close(&self) {
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        state.head = None;
        state.chunks = None;
    }

    /// Reports the response head: `status`, the `url` it came from (the
    /// request URL, since redirects are never followed), and `headers`.
    /// Call it once, before any chunk; a second is ignored. Output: none.
    pub fn head(&self, status: u16, url: String, headers: Vec<HostHeader>) {
        let sender = self
            .state
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .head
            .take();
        if let Some(sender) = sender {
            let headers = headers.into_iter().map(|h| (h.name, h.value)).collect();
            let _ = sender.send(crate::host_transport::Head {
                status,
                url,
                headers,
            });
        }
    }

    /// Reports the next part of the response body. Call it from the
    /// host's own thread, after `head`. Output: none; it blocks while the
    /// reader is behind, and bytes past the reply cap end the body.
    pub fn chunk(&self, bytes: Vec<u8>) {
        let (sender, over) = {
            let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
            if state.head.is_some() {
                return;
            }
            let total = state.bytes.saturating_add(bytes.len());
            let over = total > self.cap;
            state.bytes = total.min(self.cap.saturating_add(1));
            let sender = if over {
                state.chunks.take()
            } else {
                state.chunks.clone()
            };
            (sender, over)
        };
        let Some(sender) = sender else {
            return;
        };
        let sent = if over {
            Err(HttpError::new(
                HttpErrorKind::Rejected,
                "host transport reply is over the reply cap",
            ))
        } else {
            Ok(bytes)
        };
        let _ = sender.blocking_send(sent);
    }
}

/// Runs `call`, a read of host code, and turns a panic in it into a
/// `Panic` error naming `what`, so no host panic crosses the boundary.
/// Output: what `call` returns, or that error.
pub(crate) fn guarded<T>(
    what: &str,
    call: impl FnOnce() -> Result<T, FfiError>,
) -> Result<T, FfiError> {
    match catch_unwind(AssertUnwindSafe(call)) {
        Ok(result) => result,
        Err(_) => Err(FfiError::new(
            ErrorCode::Panic,
            format!("host {what} panicked"),
        )),
    }
}

/// Checks host tool `host` and wraps it as a kit tool: name 1 to 64
/// bytes of `[a-z0-9_]`, description at most 16 KiB, schema a JSON
/// object schema with `"type": "object"` that compiles. Output: the
/// tool, or `InvalidArgument` naming the bad part.
pub(crate) fn wrap_tool(host: Arc<dyn HostTool>) -> Result<Arc<dyn Tool>, FfiError> {
    let tool: Arc<dyn Tool> = Arc::new(host_tool(host)?);
    Ok(tool)
}

/// Checks host tool `host` as `wrap_tool` does and returns the adapter
/// itself, for a graph step that takes a tool by value.
/// Output: the adapter, or `InvalidArgument` naming the bad part.
pub(crate) fn host_tool(host: Arc<dyn HostTool>) -> Result<HostToolAdapter, FfiError> {
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
    if let Err(why) = troupekit_tools::compile_schema(&schema) {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            format!("host tool `{name}` schema: {why}"),
        ));
    }
    assert!(schema.is_object());
    let spec = ToolSpec {
        name,
        description,
        schema,
        hosted: None,
    };
    Ok(HostToolAdapter { host, spec })
}

pub(crate) struct HostToolAdapter {
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

/// Checks host embedder `host` and wraps it as a kit embedder: `dims`
/// read once, 1 to 65 536. Each call sends at most 256 texts to the host
/// and checks every batch's count, length and finite values.
/// Output: the embedder, or `InvalidArgument` for a bad `dims`.
#[cfg(feature = "knowledge")]
pub(crate) fn wrap_embedder(
    host: Arc<dyn HostEmbedder>,
) -> Result<Arc<dyn troupekit_context::embed::Embedder>, FfiError> {
    let dims = host.dims();
    if dims == 0 || dims > MAX_EMBED_DIMS {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "host embedder dims must be 1 to 65536",
        ));
    }
    assert!(dims > 0);
    Ok(Arc::new(HostEmbedderAdapter {
        host,
        dims: dims as usize,
    }))
}

#[cfg(feature = "knowledge")]
struct HostEmbedderAdapter {
    host: Arc<dyn HostEmbedder>,
    dims: usize,
}

#[cfg(feature = "knowledge")]
#[async_trait]
impl troupekit_context::embed::Embedder for HostEmbedderAdapter {
    fn dims(&self) -> usize {
        self.dims
    }

    async fn embed(
        &self,
        texts: &[String],
    ) -> Result<Vec<Vec<f32>>, troupekit_context::ContextError> {
        use troupekit_context::ContextError;
        let mut vectors = Vec::with_capacity(texts.len());
        for batch in texts.chunks(MAX_EMBED_BATCH) {
            let host = self.host.clone();
            let sent = batch.to_vec();
            let called = tokio::task::spawn_blocking(move || {
                catch_unwind(AssertUnwindSafe(|| host.embed(sent)))
            })
            .await;
            let got = match called {
                Ok(Ok(Ok(got))) => got,
                Ok(Ok(Err(error))) => {
                    return Err(ContextError::Embed(format!(
                        "host embedder: {}",
                        error.message
                    )))
                }
                Ok(Err(_)) | Err(_) => {
                    return Err(ContextError::Embed("host embedder panicked".into()))
                }
            };
            if got.len() != batch.len() {
                return Err(ContextError::Embed(format!(
                    "host embedder returned {} vectors for {} texts",
                    got.len(),
                    batch.len()
                )));
            }
            if got
                .iter()
                .any(|v| v.len() != self.dims || v.iter().any(|x| !x.is_finite()))
            {
                return Err(ContextError::Embed(format!(
                    "host embedder vectors must be {} finite numbers",
                    self.dims
                )));
            }
            vectors.extend(got);
        }
        assert_eq!(vectors.len(), texts.len());
        Ok(vectors)
    }
}
