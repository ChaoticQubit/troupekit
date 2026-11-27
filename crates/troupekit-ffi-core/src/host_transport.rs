use crate::host::{HostTransport, HttpSink};
use crate::records::{HostError, HostHeader, HostHttpRequest};
use async_trait::async_trait;
use std::any::Any;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Arc;
use tokio::sync::{mpsc, oneshot};
use tokio::task::JoinHandle;
use troupekit_core::http::{Body, HttpError, HttpErrorKind, HttpRequest, HttpResponse, Transport};

pub(crate) const CHUNK_QUEUE: usize = 64;
const REDIRECTS: [u16; 5] = [301, 302, 303, 307, 308];

type Sent = Result<Result<(), HostError>, Box<dyn Any + Send>>;

/// The response head a host transport reports first.
#[derive(Debug)]
pub(crate) struct Head {
    pub status: u16,
    pub url: String,
    pub headers: Vec<(String, String)>,
}

/// Wraps host transport `host` as a core transport for one vendor: it
/// sends only to URLs under `base`, plain `http` only when `plain_ok`,
/// refuses a reported redirect and a final URL on another origin, and
/// ends a body past `cap` bytes with `Rejected`. Output: the transport.
pub(crate) fn wrap_transport(
    host: Arc<dyn HostTransport>,
    base: &str,
    plain_ok: bool,
    cap: usize,
) -> Arc<dyn Transport> {
    let base = base.trim_end_matches('/').to_string();
    assert!(base.starts_with("https://") || base.starts_with("http://"));
    assert!(cap > 0);
    Arc::new(HostTransportAdapter {
        host,
        base,
        plain_ok,
        cap,
    })
}

struct HostTransportAdapter {
    host: Arc<dyn HostTransport>,
    base: String,
    plain_ok: bool,
    cap: usize,
}

impl HostTransportAdapter {
    fn check(&self, url: &str) -> Result<(), HttpError> {
        if url.starts_with("http://") && !self.plain_ok {
            return Err(HttpError::new(
                HttpErrorKind::Blocked,
                "host transport refuses plain http for a vendor with a key",
            ));
        }
        let under_base = url
            .strip_prefix(&self.base)
            .is_some_and(|rest| rest.is_empty() || rest.starts_with('/') || rest.starts_with('?'));
        if !under_base {
            return Err(HttpError::new(
                HttpErrorKind::Blocked,
                format!("host transport only sends to {}", self.base),
            ));
        }
        Ok(())
    }
}

#[async_trait]
impl Transport for HostTransportAdapter {
    async fn send(&self, req: HttpRequest) -> Result<HttpResponse, HttpError> {
        self.check(&req.url)?;
        let asked = origin(&req.url);
        assert!(asked.is_some());
        let mut request = HostHttpRequest::default();
        request.method = req.method.as_str().to_string();
        request.url = req.url.clone();
        request.headers = req
            .headers
            .iter()
            .map(|(name, value)| HostHeader::new(name.clone(), value.clone()))
            .collect();
        request.body = req.body.clone().unwrap_or_default();
        let (head_tx, mut head_rx) = oneshot::channel();
        let (chunk_tx, chunk_rx) = mpsc::channel(CHUNK_QUEUE);
        let sink = Arc::new(HttpSink::open(head_tx, chunk_tx, self.cap));
        let host = self.host.clone();
        let job_sink = sink.clone();
        let mut job = tokio::task::spawn_blocking(move || {
            catch_unwind(AssertUnwindSafe(|| host.send(request, job_sink)))
        });
        let (head, finished) = tokio::select! {
            head = &mut head_rx => (head.ok(), None),
            sent = &mut job => (head_rx.try_recv().ok(), Some(sent)),
        };
        let Some(head) = head else {
            sink.close();
            return Err(HttpError::new(HttpErrorKind::Network, no_head(finished)));
        };
        if REDIRECTS.contains(&head.status) {
            sink.close();
            return Err(HttpError::new(
                HttpErrorKind::Rejected,
                format!(
                    "host transport reported redirect {}; troupekit never follows one",
                    head.status
                ),
            ));
        }
        if origin(&head.url) != asked {
            sink.close();
            return Err(HttpError::new(
                HttpErrorKind::Rejected,
                "host transport's final URL is on another origin",
            ));
        }
        let body = HostBody {
            chunks: chunk_rx,
            job: match finished {
                Some(sent) => {
                    sink.close();
                    Done::Finished(failure(sent))
                }
                None => Done::Running(job),
            },
            sink,
        };
        Ok(HttpResponse {
            status: head.status,
            url: head.url,
            headers: head.headers,
            body: Box::new(body),
            truncated: false,
        })
    }
}

fn no_head(finished: Option<Result<Sent, tokio::task::JoinError>>) -> String {
    match finished.map(failure) {
        Some(Some(message)) => message,
        _ => "host transport returned without a response head".into(),
    }
}

fn failure(sent: Result<Sent, tokio::task::JoinError>) -> Option<String> {
    match sent {
        Ok(Ok(Ok(()))) => None,
        Ok(Ok(Err(error))) => Some(format!("host transport: {}", error.message)),
        Ok(Err(_)) | Err(_) => Some("host transport panicked".into()),
    }
}

enum Done {
    Running(JoinHandle<Sent>),
    Finished(Option<String>),
}

struct HostBody {
    chunks: mpsc::Receiver<Result<Vec<u8>, HttpError>>,
    job: Done,
    sink: Arc<HttpSink>,
}

#[async_trait]
impl Body for HostBody {
    async fn chunk(&mut self) -> Result<Option<Vec<u8>>, HttpError> {
        if let Done::Running(job) = &mut self.job {
            tokio::select! {
                got = self.chunks.recv() => {
                    if let Some(got) = got {
                        return got.map(Some);
                    }
                }
                sent = job => {
                    self.sink.close();
                    self.job = Done::Finished(failure(sent));
                }
            }
        }
        match self.chunks.recv().await {
            Some(got) => got.map(Some),
            None => match &mut self.job {
                Done::Finished(failed) => match failed.take() {
                    Some(message) => Err(HttpError::new(HttpErrorKind::Network, message)),
                    None => Ok(None),
                },
                Done::Running(_) => Ok(None),
            },
        }
    }
}

impl Drop for HostBody {
    fn drop(&mut self) {
        self.sink.close();
    }
}

/// Returns `scheme://authority` of `url` in lower case, or `None` for
/// a URL without a scheme or with user info in its authority.
pub(crate) fn origin(url: &str) -> Option<String> {
    let (scheme, rest) = url.split_once("://")?;
    let end = rest.find(['/', '?', '#']).unwrap_or(rest.len());
    let authority = &rest[..end];
    if authority.is_empty() || authority.contains('@') {
        return None;
    }
    Some(format!(
        "{}://{}",
        scheme.to_ascii_lowercase(),
        authority.to_ascii_lowercase()
    ))
}
