use troupekit_ffi_core::{HostHeader, HostHttpRequest};

const KEY: &str = "sk-test-very-secret-value";

#[test]
fn host_request_debug_hides_key() {
    let mut request = HostHttpRequest::default();
    request.method = "POST".into();
    request.url = "https://api.openai.com/v1/chat/completions".into();
    request.headers = vec![
        HostHeader::new("Authorization", format!("Bearer {KEY}")),
        HostHeader::new("x-api-key", KEY),
        HostHeader::new("X-Goog-Api-Key", KEY),
        HostHeader::new("content-type", "application/json"),
    ];
    let shown = format!("{request:?}");
    assert!(!shown.contains(KEY), "{shown}");
    assert!(shown.contains("[redacted]"), "{shown}");
    assert!(shown.contains("application/json"), "{shown}");
    assert!(request.headers[0].sensitive);
    assert!(request.headers[1].sensitive);
    assert!(request.headers[2].sensitive);
    assert!(!request.headers[3].sensitive);
}

#[test]
fn sensitive_header_is_redacted_by_flag_or_name() {
    let mut custom = HostHeader::new("x-session-token", KEY);
    assert!(!custom.sensitive);
    custom.sensitive = true;
    assert!(!format!("{custom:?}").contains(KEY));
    let mut renamed = HostHeader::new("content-type", "text/plain");
    renamed.name = "authorization".into();
    renamed.value = KEY.into();
    assert!(!format!("{renamed:?}").contains(KEY));
    assert_eq!(renamed.value, KEY);
}
