#![cfg(feature = "reqwest")]

use std::io::{Read, Write};
use std::net::TcpListener;
use std::thread;
use std::time::Duration;
use troupekit_core::error::{ProviderError, ProviderErrorKind};
use troupekit_providers::reqwest_transport::ReqwestTransport;
use troupekit_providers::transport::{read_body, HttpRequest, Method, Transport};

fn serve_once(reply: &'static str, delay: Duration) -> (String, thread::JoinHandle<String>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}/v1/messages", listener.local_addr().unwrap());
    let handle = thread::spawn(move || {
        let (mut socket, _) = listener.accept().unwrap();
        let mut seen = Vec::new();
        let mut buf = [0u8; 4096];
        while !request_complete(&seen) {
            let n = socket.read(&mut buf).unwrap();
            if n == 0 {
                break;
            }
            seen.extend_from_slice(&buf[..n]);
        }
        thread::sleep(delay);
        let _ = socket.write_all(reply.as_bytes());
        String::from_utf8_lossy(&seen).into_owned()
    });
    (url, handle)
}

fn request_complete(seen: &[u8]) -> bool {
    let text = String::from_utf8_lossy(seen);
    let Some((head, body)) = text.split_once("\r\n\r\n") else {
        return false;
    };
    let len = head
        .lines()
        .find_map(|line| {
            let line = line.to_ascii_lowercase();
            line.strip_prefix("content-length:")
                .map(|v| v.trim().parse::<usize>().unwrap_or(0))
        })
        .unwrap_or(0);
    body.len() >= len
}

fn post(url: &str) -> HttpRequest {
    HttpRequest::post(
        url,
        vec![
            ("content-type".into(), "application/json".into()),
            ("x-api-key".into(), "test-key".into()),
        ],
        br#"{"model":"m"}"#.to_vec(),
    )
}

#[tokio::test]
async fn sends_a_post_and_reads_status_headers_and_body() {
    let (url, server) = serve_once(
        "HTTP/1.1 429 Too Many Requests\r\nretry-after: 3\r\ncontent-length: 2\r\nconnection: close\r\n\r\n{}",
        Duration::ZERO,
    );
    let t = ReqwestTransport::new().unwrap();
    let mut reply = t.send(post(&url)).await.unwrap();
    assert_eq!(reply.status, 429);
    assert!(reply
        .headers
        .iter()
        .any(|(k, v)| k == "retry-after" && v == "3"));
    assert_eq!(read_body(reply.body.as_mut(), 1024).await.unwrap(), b"{}");
    let seen = server.join().unwrap();
    assert!(seen.starts_with("POST /v1/messages HTTP/1.1\r\n"));
    assert!(seen.to_ascii_lowercase().contains("x-api-key: test-key"));
    assert!(seen.ends_with(r#"{"model":"m"}"#));
}

#[tokio::test]
async fn a_silent_server_is_a_stall() {
    let (url, _server) = serve_once(
        "HTTP/1.1 200 OK\r\ncontent-length: 0\r\n\r\n",
        Duration::from_millis(1500),
    );
    let t = ReqwestTransport::with_timeouts(Duration::from_secs(5), Duration::from_millis(200))
        .unwrap();
    assert_eq!(
        ProviderError::from(t.send(post(&url)).await.err().unwrap()).kind,
        ProviderErrorKind::Stalled
    );
}

#[tokio::test]
async fn a_refused_connection_is_a_network_error() {
    let port = TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port();
    let t = ReqwestTransport::new().unwrap();
    let err = t
        .send(post(&format!("http://127.0.0.1:{port}/x")))
        .await
        .err()
        .unwrap();
    assert_eq!(ProviderError::from(err).kind, ProviderErrorKind::Network);
}

#[tokio::test]
async fn honours_the_request_method() {
    let (url, server) = serve_once(
        "HTTP/1.1 204 No Content\r\ncontent-length: 0\r\nconnection: close\r\n\r\n",
        Duration::ZERO,
    );
    let req = HttpRequest {
        method: Method::Put,
        ..post(&url)
    };
    let reply = ReqwestTransport::new().unwrap().send(req).await.unwrap();
    assert_eq!(reply.status, 204);
    assert!(reply.url.ends_with("/v1/messages"));
    assert!(server
        .join()
        .unwrap()
        .starts_with("PUT /v1/messages HTTP/1.1\r\n"));
}
