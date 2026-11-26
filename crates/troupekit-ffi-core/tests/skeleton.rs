use std::sync::Arc;
use troupekit_ffi_core::{
    version, AgentBuilder, AgentHandle, ErrorCode, EventListener, FfiError, FfiEvent, GraphBuilder,
    GraphHandle, HostError, HostHttpRequest, HostModel, HostModelCaps, HostReply, HostRequest,
    HostTool, HostTransport, HttpSink, RunStream, TokenSink,
};

fn send_sync<T: Send + Sync + ?Sized>() {}

struct Echo;

impl HostTool for Echo {
    fn name(&self) -> String {
        "echo".into()
    }
    fn description(&self) -> String {
        "Echoes its input".into()
    }
    fn schema_json(&self) -> String {
        "{\"type\":\"object\"}".into()
    }
    fn call(&self, args_json: String) -> Result<String, HostError> {
        Ok(args_json)
    }
}

struct Silent;

impl EventListener for Silent {
    fn on_event(&self, _event: FfiEvent) {}
}

struct Fixed;

impl HostModel for Fixed {
    fn id(&self) -> String {
        "host/fixed".into()
    }
    fn capabilities(&self) -> HostModelCaps {
        HostModelCaps::default()
    }
    fn generate(
        &self,
        _request: HostRequest,
        sink: Arc<TokenSink>,
    ) -> Result<HostReply, HostError> {
        sink.delta("hi".into());
        let mut reply = HostReply::default();
        reply.text = "hi".into();
        Ok(reply)
    }
}

struct Offline;

impl HostTransport for Offline {
    fn send(&self, _request: HostHttpRequest, _sink: Arc<HttpSink>) -> Result<(), HostError> {
        Err(HostError::new("offline"))
    }
}

#[test]
fn handles_are_send_and_sync() {
    send_sync::<AgentBuilder>();
    send_sync::<AgentHandle>();
    send_sync::<GraphBuilder>();
    send_sync::<GraphHandle>();
    send_sync::<RunStream>();
    send_sync::<TokenSink>();
    send_sync::<HttpSink>();
    send_sync::<FfiError>();
    send_sync::<dyn HostTool>();
    send_sync::<dyn HostModel>();
    send_sync::<dyn HostTransport>();
    send_sync::<dyn EventListener>();
}

#[test]
fn host_traits_are_object_safe() {
    let tool: Arc<dyn HostTool> = Arc::new(Echo);
    let model: Arc<dyn HostModel> = Arc::new(Fixed);
    let transport: Arc<dyn HostTransport> = Arc::new(Offline);
    let listener: Arc<dyn EventListener> = Arc::new(Silent);
    assert_eq!(tool.call("{}".into()).unwrap(), "{}");
    assert_eq!(model.id(), "host/fixed");
    let builder = AgentBuilder::new("helper".into())
        .tool(tool)
        .host_model(model)
        .host_transport(transport);
    send_sync::<Arc<AgentBuilder>>();
    drop((builder, listener));
}

#[cfg(feature = "knowledge")]
#[test]
fn embedder_trait_is_object_safe() {
    struct Flat;
    impl troupekit_ffi_core::HostEmbedder for Flat {
        fn dims(&self) -> u32 {
            2
        }
        fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, HostError> {
            Ok(texts.iter().map(|_| vec![0.0, 1.0]).collect())
        }
    }
    let embedder: Arc<dyn troupekit_ffi_core::HostEmbedder> = Arc::new(Flat);
    send_sync::<dyn troupekit_ffi_core::HostEmbedder>();
    assert_eq!(
        embedder.embed(vec!["a".into()]).unwrap(),
        vec![vec![0.0, 1.0]]
    );
}

#[test]
fn version_matches_core() {
    assert_eq!(version(), env!("CARGO_PKG_VERSION"));
}

#[tokio::test]
async fn empty_graph_is_refused_at_build() {
    let err = GraphBuilder::new("chain".into()).build().await.unwrap_err();
    assert_ne!(err.code, ErrorCode::Unsupported);
    assert!(err.run_id.is_none());
}

#[test]
fn defaults_are_the_safe_values() {
    let settings = troupekit_ffi_core::RuntimeSettings::default();
    assert_eq!(settings.worker_threads, 2);
    assert_eq!(settings.max_input_bytes, 1024 * 1024);
    assert_eq!(settings.event_buffer, 4096);
    assert!(!HostModelCaps::default().tools);
}

#[cfg(feature = "knowledge")]
#[test]
fn record_defaults_match_the_kit() {
    let memory = troupekit_ffi_core::MemoryConfig::new("ada", "s1");
    assert!(memory.preamble);
    assert!(memory.sqlite.is_none());
    let knowledge = troupekit_ffi_core::KnowledgeConfig::new("docs");
    assert!(!knowledge.allow_failed);
    assert_eq!(knowledge.dir, "docs");
}
