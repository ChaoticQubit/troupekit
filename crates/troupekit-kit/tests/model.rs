use crate::common::*;
use troupekit::{Agent, KitError};
use troupekit_core::error::ProviderErrorKind;

fn calc() -> Agent {
    Agent::new("calc").instructions("Add numbers.")
}

#[tokio::test]
async fn unknown_model_fails_at_build_with_known_list() {
    let err = calc()
        .model("openai/gpt-4.1-mini")
        .api_key(secret("k"))
        .transport(scripted())
        .build()
        .await
        .unwrap_err();
    assert!(matches!(err, KitError::Catalog { .. }), "{err}");
    let text = err.to_string();
    assert!(text.starts_with("agent `calc`: "), "{text}");
    assert!(text.contains("openai/gpt-4.1-mini"), "{text}");
    assert!(
        text.ends_with("known models for openai: gpt-6.2-sol, gpt-6.1-terra, gpt-6-luna"),
        "{text}"
    );
}

#[tokio::test]
async fn unknown_vendor_fails_at_build() {
    let err = calc()
        .model("nobody/x")
        .transport(scripted())
        .build()
        .await
        .unwrap_err();
    let text = err.to_string();
    assert!(matches!(err, KitError::Catalog { .. }), "{text}");
    assert!(text.contains("unknown vendor: nobody"), "{text}");
    assert!(text.contains("known vendors: "), "{text}");
    assert!(text.contains("openai"), "{text}");
}

#[tokio::test]
async fn missing_key_names_env_var_not_value() {
    let err = calc()
        .model("openai/gpt-6-luna")
        .api_key_env("TROUPEKIT_KIT_TEST_NEVER_SET")
        .transport(scripted())
        .build()
        .await
        .unwrap_err();
    assert!(
        matches!(err, KitError::MissingKey { ref vendor, ref env }
            if vendor == "openai" && env == "TROUPEKIT_KIT_TEST_NEVER_SET"),
        "{err}"
    );
    assert_eq!(
        err.to_string(),
        "vendor `openai` needs an API key: set TROUPEKIT_KIT_TEST_NEVER_SET or call .api_key(..)"
    );
}

#[tokio::test]
async fn api_key_override_beats_env() {
    let transport = scripted();
    let text = calc()
        .model("openai/gpt-6-luna")
        .api_key_env("TROUPEKIT_KIT_TEST_NEVER_SET")
        .api_key(secret("override-key"))
        .transport(transport.clone())
        .run("hi")
        .await
        .unwrap();
    assert_eq!(text, "Hi there");
    assert_eq!(
        transport.sent()[0].header("authorization"),
        Some("Bearer override-key")
    );
}

#[tokio::test]
async fn catalog_model_over_scripted_transport_needs_no_network() {
    let transport = scripted();
    let outcome = calc()
        .model("openai/gpt-6-luna")
        .api_key(secret("k"))
        .transport(transport.clone())
        .run_with("hi")
        .await
        .unwrap();
    assert_eq!(outcome.text(), "Hi there");
    assert!(outcome.usage().cost_usd_micros > 0);
    assert_eq!(
        transport.sent()[0].url,
        "https://api.openai.com/v1/chat/completions"
    );
}

#[tokio::test]
async fn open_models_vendor_accepts_any_id_at_free_price() {
    let transport = scripted();
    let outcome = calc()
        .model("ollama/qwen3:8b")
        .transport(transport.clone())
        .run_with("hi")
        .await
        .unwrap();
    assert_eq!(outcome.text(), "Hi there");
    assert_eq!(outcome.usage().cost_usd_micros, 0);
    assert_eq!(transport.sent()[0].header("authorization"), None);
}

#[tokio::test]
async fn alias_selector_needs_model_client() {
    let err = calc().model("fast").build().await.unwrap_err();
    assert_eq!(
        err.to_string(),
        "agent `calc`: model alias `fast` needs .model_client(..)"
    );
    let text = calc()
        .model("fast")
        .model_client(echo_model())
        .run("hi")
        .await
        .unwrap();
    assert_eq!(text, "Hello! You said: hi");
}

#[tokio::test]
async fn key_never_appears_in_debug_or_errors() {
    let key = "sk-test-secret-value-9731";
    let agent = calc()
        .model("openai/gpt-4.1-mini")
        .api_key(secret(key))
        .transport(scripted());
    assert!(!format!("{agent:?}").contains(key));
    let err = agent.build().await.unwrap_err();
    assert!(!err.to_string().contains(key));
    assert!(!format!("{err:?}").contains(key));
    let built = calc()
        .model("openai/gpt-6-luna")
        .api_key(secret(key))
        .transport(scripted())
        .build()
        .await
        .unwrap();
    assert!(!format!("{built:?}").contains(key));
    let failed = calc()
        .model("fake/m")
        .api_key(secret(key))
        .model_client(failing_model(ProviderErrorKind::Auth))
        .run("hi")
        .await
        .unwrap_err();
    assert!(!failed.to_string().contains(key));
}
