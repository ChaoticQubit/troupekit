use crate::error::HostError;
use crate::records::{Event, HostHeader, HostHttpRequest, HostModelCaps, HostReply, HostRequest};
use std::sync::Arc;
use troupekit_ffi_core as neutral;

/// Receives every event of a run, implemented in Swift or Kotlin.
/// Called on a Rust worker thread, never on the caller's thread.
/// Input: one event. Output: none.
#[uniffi::export(with_foreign)]
pub trait EventListener: Send + Sync {
    fn on_event(&self, event: Event);
}

/// A tool implemented in Swift or Kotlin and called by the agent.
/// `schema_json` is a JSON Schema object; `call` gets the arguments as JSON
/// and returns the output text, or a `HostError` the model sees.
#[uniffi::export(with_foreign)]
pub trait HostTool: Send + Sync {
    fn name(&self) -> String;
    fn description(&self) -> String;
    fn schema_json(&self) -> String;
    fn call(&self, args_json: String) -> Result<String, HostError>;
}

/// A model implemented in Swift or Kotlin, such as an on-device model.
/// `generate` pushes text deltas into `sink` and returns the whole reply;
/// Rust renders the request and parses the reply.
#[uniffi::export(with_foreign)]
pub trait HostModel: Send + Sync {
    fn id(&self) -> String;
    fn capabilities(&self) -> HostModelCaps;
    fn generate(&self, request: HostRequest, sink: Arc<TokenSink>) -> Result<HostReply, HostError>;
}

/// An embedder implemented in Swift or Kotlin, used for knowledge.
/// `embed` returns one vector of `dims` floats per text, in order.
/// Rust checks the shape and the batch size.
#[cfg(feature = "knowledge")]
#[uniffi::export(with_foreign)]
pub trait HostEmbedder: Send + Sync {
    fn dims(&self) -> u32;
    fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, HostError>;
}

/// An HTTP stack implemented in Swift or Kotlin, used for model calls only.
/// `send` calls `sink.head` once, then `sink.chunk` per body chunk, then
/// returns; Rust has already checked the URL against the vendor.
#[uniffi::export(with_foreign)]
pub trait HostTransport: Send + Sync {
    fn send(&self, request: HostHttpRequest, sink: Arc<HttpSink>) -> Result<(), HostError>;
}

/// Where a host model pushes its text deltas, wrapping the neutral sink.
/// Input: text pieces in order. Output: none; Rust caps the total bytes.
/// Created by Rust and handed to `HostModel::generate`.
#[derive(Debug, uniffi::Object)]
pub struct TokenSink(Arc<neutral::TokenSink>);

#[uniffi::export]
impl TokenSink {
    pub fn delta(&self, text: String) {
        self.0.delta(text)
    }
}

/// Where a host transport pushes the reply, wrapping the neutral sink.
/// Input: the head once, then body chunks. Output: none; Rust caps the body.
/// Created by Rust and handed to `HostTransport::send`.
#[derive(Debug, uniffi::Object)]
pub struct HttpSink(Arc<neutral::HttpSink>);

#[uniffi::export]
impl HttpSink {
    pub fn head(&self, status: u16, url: String, headers: Vec<HostHeader>) {
        self.0
            .head(status, url, headers.into_iter().map(Into::into).collect())
    }

    pub fn chunk(&self, bytes: Vec<u8>) {
        self.0.chunk(bytes)
    }
}

/// Receives every trace record of a traced build or run, implemented in
/// Swift or Kotlin: the callback form of `trace_jsonl`. Called on the
/// thread that writes the record; keys are scrubbed before the call.
/// Input: one JSON object on one line, no newline. Output: none.
#[uniffi::export(with_foreign)]
pub trait TraceListener: Send + Sync {
    fn on_record(&self, line: String);
}

/// Presents a foreign `TraceListener` as the neutral trait.
/// Input: the foreign object. Output: a neutral listener that forwards
/// each record unchanged.
pub(crate) struct TraceBridge(pub(crate) Arc<dyn TraceListener>);

impl neutral::TraceListener for TraceBridge {
    fn on_record(&self, line: String) {
        self.0.on_record(line)
    }
}

/// Presents a foreign `EventListener` as the neutral trait.
/// Input: the foreign object. Output: a neutral listener that converts
/// each event and forwards it unchanged.
pub(crate) struct ListenerBridge(pub(crate) Arc<dyn EventListener>);

impl neutral::EventListener for ListenerBridge {
    fn on_event(&self, event: neutral::FfiEvent) {
        self.0.on_event(event.into())
    }
}

/// Presents a foreign `HostTool` as the neutral trait.
/// Input: the foreign object. Output: a neutral tool whose calls and
/// errors are converted and forwarded unchanged.
pub(crate) struct ToolBridge(pub(crate) Arc<dyn HostTool>);

impl neutral::HostTool for ToolBridge {
    fn name(&self) -> String {
        self.0.name()
    }

    fn description(&self) -> String {
        self.0.description()
    }

    fn schema_json(&self) -> String {
        self.0.schema_json()
    }

    fn call(&self, args_json: String) -> Result<String, neutral::HostError> {
        self.0.call(args_json).map_err(Into::into)
    }
}

/// Presents a foreign `HostModel` as the neutral trait.
/// Input: the foreign object. Output: a neutral model whose requests,
/// sinks, replies and errors are converted and forwarded unchanged.
pub(crate) struct ModelBridge(pub(crate) Arc<dyn HostModel>);

impl neutral::HostModel for ModelBridge {
    fn id(&self) -> String {
        self.0.id()
    }

    fn capabilities(&self) -> neutral::HostModelCaps {
        self.0.capabilities().into()
    }

    fn generate(
        &self,
        request: neutral::HostRequest,
        sink: Arc<neutral::TokenSink>,
    ) -> Result<neutral::HostReply, neutral::HostError> {
        let sink = Arc::new(TokenSink(sink));
        self.0
            .generate(request.into(), sink)
            .map(Into::into)
            .map_err(Into::into)
    }
}

/// Presents a foreign `HostEmbedder` as the neutral trait.
/// Input: the foreign object. Output: a neutral embedder whose results and
/// errors are forwarded unchanged.
#[cfg(feature = "knowledge")]
pub(crate) struct EmbedderBridge(pub(crate) Arc<dyn HostEmbedder>);

#[cfg(feature = "knowledge")]
impl neutral::HostEmbedder for EmbedderBridge {
    fn dims(&self) -> u32 {
        self.0.dims()
    }

    fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, neutral::HostError> {
        self.0.embed(texts).map_err(Into::into)
    }
}

/// Presents a foreign `HostTransport` as the neutral trait.
/// Input: the foreign object. Output: a neutral transport whose requests,
/// sinks and errors are converted and forwarded unchanged.
pub(crate) struct TransportBridge(pub(crate) Arc<dyn HostTransport>);

impl neutral::HostTransport for TransportBridge {
    fn send(
        &self,
        request: neutral::HostHttpRequest,
        sink: Arc<neutral::HttpSink>,
    ) -> Result<(), neutral::HostError> {
        let sink = Arc::new(HttpSink(sink));
        self.0.send(request.into(), sink).map_err(Into::into)
    }
}
