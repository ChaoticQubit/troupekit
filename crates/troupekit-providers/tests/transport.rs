use troupekit_core::error::ProviderErrorKind;
use troupekit_providers::transport::{
    read_body, HttpError, HttpErrorKind, HttpRequest, Scripted, ScriptedTransport, Transport,
};

fn req(url: &str) -> HttpRequest {
    HttpRequest::post(
        url,
        vec![("Authorization".into(), "Bearer sk-test-not-real".into())],
        b"{}".to_vec(),
    )
}

#[tokio::test]
async fn replays_in_order_and_records_requests() {
    let t = ScriptedTransport::new(vec![
        Scripted::chunked("abcdefg", 3).with_header("x-id", "1"),
        Scripted::Fail(HttpError::new(HttpErrorKind::Network, "down")),
    ]);
    let mut reply = t.send(req("https://a.test/1")).await.unwrap();
    assert_eq!(reply.status, 200);
    assert_eq!(reply.headers, vec![("x-id".to_string(), "1".to_string())]);
    assert_eq!(reply.body.chunk().await.unwrap(), Some(b"abc".to_vec()));
    assert_eq!(read_body(reply.body.as_mut(), 100).await.unwrap(), b"defg");
    assert_eq!(
        t.send(req("https://a.test/2")).await.err().unwrap().kind,
        HttpErrorKind::Network
    );
    assert_eq!(
        t.send(req("https://a.test/3")).await.err().unwrap().message,
        "script has no more replies"
    );
    let urls: Vec<String> = t.sent().into_iter().map(|r| r.url).collect();
    assert_eq!(
        urls,
        ["https://a.test/1", "https://a.test/2", "https://a.test/3"]
    );
}

#[tokio::test]
async fn read_body_enforces_its_limit() {
    let t = ScriptedTransport::new(vec![Scripted::chunked("0123456789", 4)]);
    let mut reply = t.send(req("https://a.test")).await.unwrap();
    assert_eq!(
        read_body(reply.body.as_mut(), 9).await.unwrap_err().kind,
        ProviderErrorKind::Decode
    );
}

#[test]
fn debug_never_shows_header_values() {
    let shown = format!("{:?}", req("https://a.test"));
    assert!(shown.contains("Authorization"));
    assert!(!shown.contains("sk-test-not-real"));
}

#[test]
fn header_lookup_ignores_case() {
    assert_eq!(
        req("u").header("authorization"),
        Some("Bearer sk-test-not-real")
    );
    assert_eq!(req("u").header("x-missing"), None);
}
