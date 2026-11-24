mod common;

use common::*;
use serde_json::Value;
use std::sync::Arc;
use troupekit_core::secret::Secret;
use troupekit_providers::catalog::{Catalog, CatalogError};
use troupekit_providers::request::CompletionRequest;
use troupekit_providers::transport::{Scripted, ScriptedTransport, Transport};

const CATALOG: &str = include_str!("../../../catalog/providers.toml");

fn key() -> Option<Secret> {
    Some(Secret::new("test-key".into()).unwrap())
}

#[tokio::test]
async fn connect_with_returns_a_metered_model_for_the_vendor() {
    let catalog = Catalog::load(CATALOG).unwrap();
    let transport = Arc::new(ScriptedTransport::new(vec![Scripted::body(
        200,
        golden_text("anthropic_reply.json"),
    )]));
    let model = catalog
        .connect_with("anthropic/claude-opus-5-5", key(), transport.clone())
        .unwrap();
    assert_eq!(model.id(), "anthropic/claude-opus-5-5");
    let mut req = CompletionRequest::user("hi", 64);
    req.temperature = Some(0.3);
    let reply = model.complete(&req).await.unwrap();
    assert_eq!(reply.usage.cost_usd_micros, 648);
    let sent = &transport.sent()[0];
    assert_eq!(sent.url, "https://api.anthropic.com/v1/messages");
    assert_eq!(sent.header("x-api-key"), Some("test-key"));
    let body: Value = serde_json::from_slice(sent.body.as_deref().unwrap()).unwrap();
    assert_eq!(body["model"], "claude-opus-5-5");
    assert!(body.get("temperature").is_none());
}

#[tokio::test]
async fn local_vendors_take_unlisted_models_and_send_no_key() {
    let catalog = Catalog::load(CATALOG).unwrap();
    let transport = Arc::new(ScriptedTransport::new(vec![Scripted::body(
        200,
        golden_text("openai_reply.json"),
    )]));
    let model = catalog
        .connect_with("ollama/qwen3:8b", key(), transport.clone())
        .unwrap();
    assert_eq!(
        model
            .complete(&CompletionRequest::user("hi", 64))
            .await
            .unwrap()
            .usage
            .cost_usd_micros,
        0
    );
    let sent = &transport.sent()[0];
    assert_eq!(sent.url, "http://localhost:11434/v1/chat/completions");
    assert!(sent.header("authorization").is_none());
}

#[test]
fn connect_errors_name_the_problem() {
    let catalog = Catalog::load(CATALOG).unwrap();
    let transport = || Arc::new(ScriptedTransport::default()) as Arc<dyn Transport>;
    assert!(matches!(
        catalog.connect_with("anthropic", key(), transport()),
        Err(CatalogError::BadSelector(_))
    ));
    assert!(matches!(
        catalog.connect_with("nobody/x", key(), transport()),
        Err(CatalogError::UnknownVendor(_))
    ));
    assert!(matches!(
        catalog.connect_with("anthropic/claude-nope", key(), transport()),
        Err(CatalogError::UnknownModel(_))
    ));
    assert!(matches!(
        catalog.connect_with("anthropic/claude-opus-5-5", None, transport()),
        Err(CatalogError::MissingKey(_))
    ));
    assert!(catalog
        .connect_with("lmstudio/local-model", None, transport())
        .is_ok());
    assert!(catalog
        .connect_with("openrouter/anthropic/claude-opus-5.5", key(), transport())
        .is_ok());
}

#[cfg(feature = "reqwest")]
#[test]
fn connect_builds_a_reqwest_backed_model_offline() {
    let catalog = Catalog::load(CATALOG).unwrap();
    let model = catalog.connect("google/gemini-3.8-flash", key()).unwrap();
    assert_eq!(model.id(), "google/gemini-3.8-flash");
}
