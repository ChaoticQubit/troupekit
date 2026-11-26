use crate::records::{
    FfiEvent, HostError, HostHeader, HostHttpRequest, HostModelCaps, HostReply, HostRequest,
};
use std::sync::Arc;

/// Receives every event of a stream as it happens, as the callback form
/// of `RunStream::next`. Called on a troupekit worker thread, never the
/// caller's; it must return quickly and must not block on the run.
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
