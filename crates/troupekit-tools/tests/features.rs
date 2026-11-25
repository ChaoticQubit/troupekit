use async_trait::async_trait;
use troupekit_core::http::ChunkBody;
use troupekit_tools::{
    is_public_ip, receive, HttpError, HttpLimits, HttpRequest, HttpResponse, Received, Transport,
};

#[cfg(feature = "reqwest")]
#[test]
fn guarded_http_is_exported_by_default() {
    let client = troupekit_tools::GuardedHttp::new(HttpLimits::default());
    let module: troupekit_tools::http::GuardedHttp = client.clone();
    assert!(format!("{module:?}").contains("GuardedHttp"));
}

struct Fixed;

#[async_trait]
impl Transport for Fixed {
    async fn send(&self, req: HttpRequest) -> Result<HttpResponse, HttpError> {
        Ok(HttpResponse {
            status: 200,
            url: req.url,
            headers: vec![("content-type".into(), "text/plain".into())],
            body: Box::new(ChunkBody::new(vec![b"hi".to_vec()])),
            truncated: false,
        })
    }
}

#[tokio::test]
async fn transport_types_exist_without_reqwest() {
    assert!(HttpLimits::default().max_redirects > 0);
    assert!(is_public_ip("8.8.8.8".parse().unwrap()));
    assert!(!is_public_ip("127.0.0.1".parse().unwrap()));
    let got: Received = receive(&Fixed, HttpRequest::get("https://example.com/"))
        .await
        .unwrap();
    assert_eq!(got.body, b"hi".to_vec());
    assert_eq!(got.content_type, "text/plain");
}
