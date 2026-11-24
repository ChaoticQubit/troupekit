use troupekit_core::error::ProviderErrorKind;
use troupekit_providers::status::{error_detail, kind_for_status, retry_after_ms, status_error};

fn h(pairs: &[(&str, &str)]) -> Vec<(String, String)> {
    pairs
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
}

#[test]
fn maps_statuses_to_kinds() {
    use ProviderErrorKind::*;
    let table = [
        (401, Auth),
        (403, Auth),
        (408, Stalled),
        (429, RateLimited),
        (400, BadRequest),
        (404, BadRequest),
        (422, BadRequest),
        (500, Server),
        (503, Server),
        (529, Server),
        (302, Server),
    ];
    for (status, kind) in table {
        assert_eq!(kind_for_status(status), kind, "status {status}");
    }
}

#[test]
fn reads_retry_after_headers() {
    assert_eq!(retry_after_ms(&h(&[("Retry-After", "3")])), Some(3000));
    assert_eq!(retry_after_ms(&h(&[("retry-after", "1.5")])), Some(1500));
    assert_eq!(
        retry_after_ms(&h(&[("retry-after", "3"), ("retry-after-ms", "250")])),
        Some(250)
    );
    assert_eq!(
        retry_after_ms(&h(&[("retry-after", "Wed, 21 Oct 2026 07:28:00 GMT")])),
        None
    );
    assert_eq!(retry_after_ms(&h(&[("retry-after", "-1")])), None);
    assert_eq!(retry_after_ms(&[]), None);
}

#[test]
fn extracts_error_detail() {
    assert_eq!(
        error_detail(br#"{"error":{"type":"x","message":"bad key"}}"#),
        "bad key"
    );
    assert_eq!(error_detail(br#"{"message":"quota"}"#), "quota");
    assert_eq!(error_detail(b"  plain text  "), "plain text");
    assert_eq!(error_detail(b""), "empty body");
    assert_eq!(
        error_detail("\u{e9}".repeat(400).as_bytes())
            .chars()
            .count(),
        300
    );
}

#[test]
fn builds_status_errors() {
    let e = status_error(
        429,
        &h(&[("retry-after", "2")]),
        br#"{"error":{"message":"slow down"}}"#,
    );
    assert_eq!(e.kind, ProviderErrorKind::RateLimited);
    assert_eq!(e.retry_after_ms, Some(2000));
    assert_eq!(e.message, "HTTP 429: slow down");
    let e = status_error(401, &[], b"nope");
    assert_eq!((e.kind, e.retry_after_ms), (ProviderErrorKind::Auth, None));
}
