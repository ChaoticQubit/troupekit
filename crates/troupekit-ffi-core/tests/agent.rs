#[cfg(feature = "sqlite")]
use troupekit_ffi_core::Progress;
use troupekit_ffi_core::{init, AgentBuilder, ErrorCode, RuntimeSettings};

fn calc() -> std::sync::Arc<AgentBuilder> {
    AgentBuilder::new("calc".into())
        .instructions("Add numbers.".into())
        .model("openai/gpt-6-luna".into())
        .api_key("k".into())
}

#[tokio::test]
async fn builder_used_after_build_is_refused() {
    let builder = calc();
    builder.build().await.unwrap();
    let err = builder.build().await.unwrap_err();
    assert_eq!(err.code, ErrorCode::InvalidArgument);
    assert_eq!(err.message, "builder already built");
    let err = builder.role("Adds.".into()).build().await.unwrap_err();
    assert_eq!(err.code, ErrorCode::InvalidArgument);
}

#[tokio::test]
async fn input_over_cap_is_refused() {
    let cap = RuntimeSettings::default().max_input_bytes;
    let err = calc()
        .instructions("x".repeat(cap + 1))
        .build()
        .await
        .unwrap_err();
    assert_eq!(err.code, ErrorCode::InvalidArgument);
    assert!(err.message.contains("instructions"), "{err}");
    assert!(err.message.contains(&cap.to_string()), "{err}");
    let handle = calc().build().await.unwrap();
    let err = handle.run("x".repeat(cap + 1)).await.unwrap_err();
    assert_eq!(err.code, ErrorCode::InvalidArgument);
}

#[test]
fn init_twice_with_other_settings_is_refused() {
    init(RuntimeSettings::default()).unwrap();
    init(RuntimeSettings::default()).unwrap();
    let mut other = RuntimeSettings::default();
    other.worker_threads = 3;
    let err = init(other).unwrap_err();
    assert_eq!(err.code, ErrorCode::InvalidArgument);
    assert!(err.message.contains("already"), "{err}");
}

#[test]
fn settings_out_of_range_are_refused() {
    let cases: [(fn(&mut RuntimeSettings), &str); 6] = [
        (|s| s.worker_threads = 0, "worker_threads"),
        (|s| s.worker_threads = 65, "worker_threads"),
        (|s| s.max_input_bytes = 1023, "max_input_bytes"),
        (
            |s| s.max_input_bytes = 64 * 1024 * 1024 + 1,
            "max_input_bytes",
        ),
        (|s| s.event_buffer = 15, "event_buffer"),
        (|s| s.event_buffer = 1_048_577, "event_buffer"),
    ];
    for (change, field) in cases {
        let mut settings = RuntimeSettings::default();
        change(&mut settings);
        let err = init(settings).unwrap_err();
        assert_eq!(err.code, ErrorCode::InvalidArgument);
        assert!(err.message.contains(field), "{err}");
    }
}

#[tokio::test]
async fn missing_key_names_env_not_value() {
    let err = AgentBuilder::new("calc".into())
        .instructions("Add numbers.".into())
        .model("openai/gpt-6-luna".into())
        .api_key_env("TROUPEKIT_FFI_TEST_NEVER_SET".into())
        .build()
        .await
        .unwrap_err();
    assert_eq!(err.code, ErrorCode::MissingKey);
    assert!(
        err.message.contains("TROUPEKIT_FFI_TEST_NEVER_SET"),
        "{err}"
    );
    assert!(err.run_id.is_none());
}

#[tokio::test]
async fn missing_model_is_config_error() {
    let err = AgentBuilder::new("calc".into())
        .instructions("Add numbers.".into())
        .build()
        .await
        .unwrap_err();
    assert_eq!(err.code, ErrorCode::Config, "{err}");
}

#[cfg(feature = "sqlite")]
#[tokio::test]
async fn start_then_zero_budget_is_pending() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("runs.db").display().to_string();
    let handle = calc().sqlite(path).build().await.unwrap();
    let run_id = handle.start("1 + 2".into()).await.unwrap();
    let progress = handle.resume_for(run_id.clone(), 0).await.unwrap();
    assert_eq!(progress, Progress::Pending { run_id });
}

#[tokio::test]
async fn bad_run_id_is_invalid_argument() {
    let handle = calc().build().await.unwrap();
    let err = handle.resume("not-a-run-id".into()).await.unwrap_err();
    assert_eq!(err.code, ErrorCode::InvalidArgument);
    assert!(!err.message.contains("not-a-run-id"), "{err}");
}

#[tokio::test]
async fn bad_api_key_is_refused_without_echo() {
    let err = calc().api_key("   ".into()).build().await.unwrap_err();
    assert_eq!(err.code, ErrorCode::InvalidArgument);
    assert!(err.message.contains("api_key"), "{err}");
}

#[tokio::test]
async fn stream_returns_once_the_run_id_is_known() {
    let handle = calc().build().await.unwrap();
    let stream = handle.stream("1 + 2".into()).await.unwrap();
    stream.cancel();
    assert_eq!(stream.run_id().len(), 36);
    assert_eq!(stream.dropped_deltas(), 0);
}

#[tokio::test]
async fn stream_input_over_cap_is_refused() {
    let handle = calc().build().await.unwrap();
    let cap = RuntimeSettings::default().max_input_bytes;
    let err = handle.stream("x".repeat(cap + 1)).await.unwrap_err();
    assert_eq!(err.code, ErrorCode::InvalidArgument);
}
