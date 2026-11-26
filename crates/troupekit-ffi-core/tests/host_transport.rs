use std::sync::Arc;
use troupekit_ffi_core::{
    AgentBuilder, ErrorCode, HostError, HostHttpRequest, HostTransport, HttpSink,
};

struct Offline;

impl HostTransport for Offline {
    fn send(&self, _request: HostHttpRequest, _sink: Arc<HttpSink>) -> Result<(), HostError> {
        Err(HostError::new("offline"))
    }
}

#[tokio::test]
async fn host_transport_builds_for_a_catalog_vendor() {
    AgentBuilder::new("calc".into())
        .instructions("Add numbers.".into())
        .model("openai/gpt-6-luna".into())
        .api_key("k".into())
        .host_transport(Arc::new(Offline))
        .build()
        .await
        .unwrap();
}

#[tokio::test]
async fn host_transport_needs_a_catalog_model() {
    let err = AgentBuilder::new("calc".into())
        .instructions("Add numbers.".into())
        .model("my-alias".into())
        .host_transport(Arc::new(Offline))
        .build()
        .await
        .unwrap_err();
    assert_eq!(err.code, ErrorCode::Config);
    assert!(err.message.contains("host_transport"), "{err}");
    assert!(err.message.contains("`my-alias`"), "{err}");
}

#[tokio::test]
async fn model_calls_go_through_host_transport() {
    let handle = AgentBuilder::new("calc".into())
        .instructions("Add numbers.".into())
        .model("openai/gpt-6-luna".into())
        .api_key("k".into())
        .retries(0)
        .host_transport(Arc::new(Offline))
        .build()
        .await
        .unwrap();
    let err = handle.run("1 + 2".into()).await.unwrap_err();
    assert_eq!(err.code, ErrorCode::Retries);
    assert!(err.message.contains("offline"), "{err}");
}
