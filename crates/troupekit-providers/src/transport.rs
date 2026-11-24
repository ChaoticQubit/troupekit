use crate::status::decode_error;
use async_trait::async_trait;
use std::collections::VecDeque;
use std::sync::{Mutex, MutexGuard};
use troupekit_core::error::ProviderError;
pub use troupekit_core::http::{
    Body, ChunkBody, HttpError, HttpErrorKind, HttpRequest, HttpResponse, Method, Transport,
};

/// Reads all of `body` into memory.
/// Returns the bytes, or `Decode` as soon as more than `limit` bytes
/// have arrived, so a hostile server cannot exhaust memory.
pub async fn read_body(body: &mut dyn Body, limit: usize) -> Result<Vec<u8>, ProviderError> {
    let (bytes, over) = troupekit_core::http::read_limited(body, limit).await?;
    if over {
        return Err(decode_error(format!("reply body is over {limit} bytes")));
    }
    Ok(bytes)
}

/// One reply a `ScriptedTransport` gives back: a status with headers and
/// body chunks, or a transport failure.
/// Build replies with `body`, `chunked` and `with_header`.
#[derive(Debug, Clone)]
pub enum Scripted {
    Reply {
        status: u16,
        headers: Vec<(String, String)>,
        chunks: Vec<Vec<u8>>,
    },
    Fail(HttpError),
}

impl Scripted {
    /// A reply with `status` and the whole `body` in one chunk.
    /// Returns the scripted reply, without headers.
    /// Add headers with `with_header`.
    pub fn body(status: u16, body: impl Into<Vec<u8>>) -> Self {
        Scripted::Reply {
            status,
            headers: Vec::new(),
            chunks: vec![body.into()],
        }
    }

    /// A 200 reply whose `body` arrives in chunks of `size` bytes, the way
    /// a slow stream would. Chunks may split UTF-8 characters.
    /// Panics when `size` is zero.
    pub fn chunked(body: &str, size: usize) -> Self {
        assert!(size > 0, "chunk size must be above zero");
        let chunks = body.as_bytes().chunks(size).map(<[u8]>::to_vec).collect();
        Scripted::Reply {
            status: 200,
            headers: Vec::new(),
            chunks,
        }
    }

    /// Adds header `name` with `value` to a reply; failures are unchanged.
    /// Returns the scripted reply.
    /// Use it for `retry-after` and similar headers in tests.
    pub fn with_header(self, name: &str, value: &str) -> Self {
        match self {
            Scripted::Reply {
                status,
                mut headers,
                chunks,
            } => {
                headers.push((name.to_owned(), value.to_owned()));
                Scripted::Reply {
                    status,
                    headers,
                    chunks,
                }
            }
            fail => fail,
        }
    }
}

/// A fake transport that replays scripted replies in order and records
/// every request sent to it, so tests never touch the network.
/// Once the script runs out, `send` fails with `Network`.
#[derive(Debug, Default)]
pub struct ScriptedTransport {
    replies: Mutex<VecDeque<Scripted>>,
    sent: Mutex<Vec<HttpRequest>>,
}

impl ScriptedTransport {
    /// Builds a transport that answers with `replies`, first to last.
    /// Returns the transport.
    /// Once every reply is used, sends fail with `Network`.
    pub fn new(replies: Vec<Scripted>) -> Self {
        Self {
            replies: Mutex::new(replies.into()),
            sent: Mutex::new(Vec::new()),
        }
    }

    /// Copies every request sent so far, oldest first.
    /// Returns the copies.
    /// Tests read them to check URLs, headers and bodies.
    pub fn sent(&self) -> Vec<HttpRequest> {
        lock(&self.sent).clone()
    }
}

#[async_trait]
impl Transport for ScriptedTransport {
    async fn send(&self, request: HttpRequest) -> Result<HttpResponse, HttpError> {
        let url = request.url.clone();
        lock(&self.sent).push(request);
        let next = lock(&self.replies).pop_front();
        match next {
            Some(Scripted::Reply {
                status,
                headers,
                chunks,
            }) => Ok(HttpResponse {
                status,
                url,
                headers,
                body: Box::new(ChunkBody::new(chunks)),
                truncated: false,
            }),
            Some(Scripted::Fail(err)) => Err(err),
            None => Err(HttpError::new(
                HttpErrorKind::Network,
                "script has no more replies",
            )),
        }
    }
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}
