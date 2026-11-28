use crate::host::{HostTransport, HttpSink};
use crate::host_transport::wrap_transport;
use crate::records::{HostError, HostHeader, HostHttpRequest};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use troupekit_core::http::{HttpErrorKind, HttpRequest, Transport};

const BASE: &str = "https://api.vendor.test/v1";

type Script = Box<dyn Fn(HostHttpRequest, Arc<HttpSink>) -> Result<(), HostError> + Send + Sync>;

struct Host {
    calls: Mutex<Vec<HostHttpRequest>>,
    script: Script,
}

impl HostTransport for Host {
    fn send(&self, request: HostHttpRequest, sink: Arc<HttpSink>) -> Result<(), HostError> {
        self.calls.lock().unwrap().push(request.clone());
        (self.script)(request, sink)
    }
}

fn host(script: Script) -> Arc<Host> {
    Arc::new(Host {
        calls: Mutex::new(Vec::new()),
        script,
    })
}

fn ok_reply() -> Script {
    Box::new(|request, sink| {
        sink.head(
            200,
            request.url,
            vec![HostHeader::new("content-type", "text/plain")],
        );
        sink.chunk(b"hel".to_vec());
        sink.chunk(b"lo".to_vec());
        Ok(())
    })
}

fn post(url: &str) -> HttpRequest {
    HttpRequest::post(
        url,
        vec![("authorization".into(), "Bearer sk-secret".into())],
        b"{}".to_vec(),
    )
}

async fn body_of(transport: &dyn Transport, url: &str) -> Result<Vec<u8>, HttpErrorKind> {
    let mut reply = transport.send(post(url)).await.map_err(|e| e.kind)?;
    let mut body = Vec::new();
    while let Some(chunk) = reply.body.chunk().await.map_err(|e| e.kind)? {
        body.extend(chunk);
    }
    Ok(body)
}

#[tokio::test]
async fn transport_sends_and_marks_the_key_sensitive() {
    let host = host(ok_reply());
    let transport = wrap_transport(host.clone(), BASE, false, 1024);
    let body = body_of(&*transport, &format!("{BASE}/chat/completions")).await;
    assert_eq!(body.unwrap(), b"hello".to_vec());
    let calls = host.calls.lock().unwrap().clone();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].method, "POST");
    assert_eq!(calls[0].body, b"{}".to_vec());
    let auth = calls[0]
        .headers
        .iter()
        .find(|h| h.name == "authorization")
        .unwrap();
    assert!(auth.sensitive);
}

#[tokio::test]
async fn transport_refuses_other_origin() {
    let host = host(ok_reply());
    let transport = wrap_transport(host.clone(), BASE, false, 1024);
    for url in [
        "https://evil.test/v1/chat",
        "https://api.vendor.test.evil.test/v1/chat",
        "https://api.vendor.test/v10/chat",
        "https://api.vendor.test@evil.test/v1/chat",
    ] {
        let err = transport.send(post(url)).await.err().unwrap();
        assert_eq!(err.kind, HttpErrorKind::Blocked, "{url}");
        assert!(!err.message.contains("sk-secret"));
    }
    assert!(host.calls.lock().unwrap().is_empty());
}

#[tokio::test]
async fn transport_refuses_plain_http_for_keyed_vendor() {
    let host = host(ok_reply());
    let keyed = wrap_transport(host.clone(), "http://localhost:11434/v1", false, 1024);
    let err = keyed
        .send(post("http://localhost:11434/v1/chat"))
        .await
        .err()
        .unwrap();
    assert_eq!(err.kind, HttpErrorKind::Blocked);
    assert!(host.calls.lock().unwrap().is_empty());
    let no_key = wrap_transport(host.clone(), "http://localhost:11434/v1", true, 1024);
    let body = body_of(&*no_key, "http://localhost:11434/v1/chat").await;
    assert_eq!(body.unwrap(), b"hello".to_vec());
}

#[tokio::test]
async fn transport_rejects_final_url_on_another_origin() {
    let host = host(Box::new(|_, sink| {
        sink.head(200, "https://evil.test/v1/chat".into(), Vec::new());
        sink.chunk(b"x".to_vec());
        Ok(())
    }));
    let transport = wrap_transport(host, BASE, false, 1024);
    let err = transport
        .send(post(&format!("{BASE}/chat")))
        .await
        .err()
        .unwrap();
    assert_eq!(err.kind, HttpErrorKind::Rejected);
}

#[tokio::test]
async fn transport_caps_body() {
    let host = host(Box::new(|request, sink| {
        sink.head(200, request.url, Vec::new());
        for _ in 0..4 {
            sink.chunk(vec![b'a'; 400]);
        }
        Ok(())
    }));
    let transport = wrap_transport(host, BASE, false, 1000);
    let err = body_of(&*transport, &format!("{BASE}/chat"))
        .await
        .unwrap_err();
    assert_eq!(err, HttpErrorKind::Rejected);
}

#[tokio::test]
async fn transport_streams_chunks_before_send_returns() {
    let (read_tx, read_rx) = mpsc::channel::<()>();
    let read_rx = Mutex::new(read_rx);
    let host = host(Box::new(move |request, sink| {
        sink.head(200, request.url, Vec::new());
        sink.chunk(b"first".to_vec());
        read_rx
            .lock()
            .unwrap()
            .recv_timeout(std::time::Duration::from_secs(10))
            .map_err(|_| HostError::new("reader never read the first chunk"))?;
        sink.chunk(b"second".to_vec());
        Ok(())
    }));
    let transport = wrap_transport(host, BASE, false, 1024);
    let mut reply = transport.send(post(&format!("{BASE}/chat"))).await.unwrap();
    assert_eq!(reply.status, 200);
    assert_eq!(
        reply.body.chunk().await.unwrap().unwrap(),
        b"first".to_vec()
    );
    read_tx.send(()).unwrap();
    assert_eq!(
        reply.body.chunk().await.unwrap().unwrap(),
        b"second".to_vec()
    );
    assert!(reply.body.chunk().await.unwrap().is_none());
}

#[tokio::test]
async fn transport_without_head_is_network_error() {
    let silent = host(Box::new(|_, _| Ok(())));
    let transport = wrap_transport(silent, BASE, false, 1024);
    let err = transport
        .send(post(&format!("{BASE}/chat")))
        .await
        .err()
        .unwrap();
    assert_eq!(err.kind, HttpErrorKind::Network);
    let failing = host(Box::new(|_, _| Err(HostError::new("offline"))));
    let transport = wrap_transport(failing, BASE, false, 1024);
    let err = transport
        .send(post(&format!("{BASE}/chat")))
        .await
        .err()
        .unwrap();
    assert_eq!(err.kind, HttpErrorKind::Network);
    assert!(err.message.contains("offline"), "{err}");
}

#[tokio::test]
async fn transport_ends_body_when_host_keeps_the_sink() {
    let kept: Arc<Mutex<Option<Arc<HttpSink>>>> = Arc::default();
    let keep = kept.clone();
    let host = host(Box::new(move |request, sink| {
        sink.head(200, request.url, Vec::new());
        sink.chunk(b"only".to_vec());
        *keep.lock().unwrap() = Some(sink);
        Ok(())
    }));
    let transport = wrap_transport(host, BASE, false, 1024);
    let body = body_of(&*transport, &format!("{BASE}/chat")).await;
    assert_eq!(body.unwrap(), b"only".to_vec());
    assert!(kept.lock().unwrap().is_some());
}

#[tokio::test]
async fn transport_refuses_a_reported_redirect() {
    for status in [301u16, 302, 303, 307, 308] {
        let host = host(Box::new(move |request, sink| {
            sink.head(
                status,
                request.url,
                vec![HostHeader::new("location", "https://evil.test/steal")],
            );
            Ok(())
        }));
        let transport = wrap_transport(host.clone(), BASE, false, 1024);
        let err = body_of(&*transport, &format!("{BASE}/chat/completions")).await;
        assert_eq!(err.unwrap_err(), HttpErrorKind::Rejected, "{status}");
        assert_eq!(host.calls.lock().unwrap().len(), 1);
    }
}
