use crate::tool::ToolError;
#[cfg(any(feature = "reqwest", test))]
use async_trait::async_trait;
#[cfg(feature = "reqwest")]
use reqwest::header::{ACCEPT, CONTENT_TYPE, LOCATION, USER_AGENT};
#[cfg(feature = "reqwest")]
use reqwest::redirect::Policy;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};
#[cfg(feature = "reqwest")]
use std::net::SocketAddr;
use std::time::Duration;
pub use troupekit_core::http::{
    media_type, read_limited, Body, ChunkBody, HttpError, HttpErrorKind, HttpRequest, HttpResponse,
    Method, Transport,
};
#[cfg(feature = "reqwest")]
use url::{Host, Url};

#[cfg(feature = "reqwest")]
const AGENT: &str = concat!("troupekit/", env!("CARGO_PKG_VERSION"));

/// A tool's view of an answer read into memory: status, final URL, lower-case
/// media type without parameters, the body cut at the size limit, and
/// `truncated` when bytes were dropped.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Received {
    pub status: u16,
    pub url: String,
    pub content_type: String,
    pub body: Vec<u8>,
    pub truncated: bool,
}

/// Sends `req` through `http` and reads the whole body into memory, at most
/// `HttpLimits::default().max_body_bytes` even when the transport has no cap.
/// Returns the answer, or the transport failure as a `ToolError`.
pub async fn receive(http: &dyn Transport, req: HttpRequest) -> Result<Received, ToolError> {
    let mut res = http.send(req).await?;
    let content_type = res.content_type();
    let (body, over) =
        read_limited(res.body.as_mut(), HttpLimits::default().max_body_bytes).await?;
    Ok(Received {
        status: res.status,
        url: res.url,
        content_type,
        body,
        truncated: res.truncated || over,
    })
}

/// Limits for one request, redirects included.
/// The default is 2 MiB of body, 15 seconds in total and 5 redirects.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HttpLimits {
    pub max_body_bytes: usize,
    pub timeout: Duration,
    pub max_redirects: usize,
}

impl Default for HttpLimits {
    fn default() -> Self {
        Self {
            max_body_bytes: 2 * 1024 * 1024,
            timeout: Duration::from_secs(15),
            max_redirects: 5,
        }
    }
}

/// HTTP client that only reaches allowed addresses, public unicast by
/// default. Each hop's host is resolved, every address checked and the
/// connection pinned to a checked address; GET redirects are re-checked and
/// a cross-origin hop drops all request headers. The body is read eagerly
/// inside the time and size limits. No proxy is used.
#[cfg(feature = "reqwest")]
#[derive(Debug, Clone)]
pub struct GuardedHttp {
    limits: HttpLimits,
    allow: fn(IpAddr) -> bool,
}

#[cfg(feature = "reqwest")]
impl GuardedHttp {
    /// Builds a client that reaches public unicast addresses only.
    /// Returns the client with `limits` applied to every request.
    pub fn new(limits: HttpLimits) -> Self {
        Self {
            limits,
            allow: is_public_ip,
        }
    }

    /// Builds a client with its own address rule, for tests and for
    /// operator-configured self-hosted services. Crate-private so model
    /// input can never widen the rule.
    pub(crate) fn with_policy(limits: HttpLimits, allow: fn(IpAddr) -> bool) -> Self {
        Self { limits, allow }
    }

    async fn follow(&self, req: HttpRequest) -> Result<HttpResponse, HttpError> {
        let mut url = Url::parse(&req.url).map_err(|e| {
            HttpError::new(
                HttpErrorKind::Invalid,
                format!("bad url `{}`: {e}", req.url),
            )
        })?;
        let method = reqwest::Method::from_bytes(req.method.as_str().as_bytes())
            .map_err(|e| HttpError::new(HttpErrorKind::Invalid, e.to_string()))?;
        let mut headers = req.headers;
        for _ in 0..=self.limits.max_redirects {
            let addr = pinned_addr(&url, self.allow).await?;
            let mut builder = reqwest::Client::builder()
                .redirect(Policy::none())
                .no_proxy()
                .timeout(self.limits.timeout);
            if let Some(Host::Domain(domain)) = url.host() {
                builder = builder.resolve(domain, addr);
            }
            let client = builder.build().map_err(|e| network(e.to_string()))?;
            let mut rb = client
                .request(method.clone(), url.clone())
                .header(USER_AGENT, AGENT);
            if !req.accept.is_empty() {
                rb = rb.header(ACCEPT, req.accept.join(", "));
            }
            for (name, value) in &headers {
                rb = rb.header(name.as_str(), value.as_str());
            }
            if let Some(body) = &req.body {
                rb = rb.body(body.clone());
            }
            let mut response = rb
                .send()
                .await
                .map_err(|e| network(format!("{url}: {}", e.without_url())))?;
            let status = response.status();
            if status.is_redirection() && req.method == Method::Get {
                let location = response
                    .headers()
                    .get(LOCATION)
                    .and_then(|l| l.to_str().ok())
                    .ok_or_else(|| network(format!("{url}: redirect without a valid Location")))?;
                let next = url
                    .join(location)
                    .map_err(|e| network(format!("bad redirect `{location}`: {e}")))?;
                if next.origin() != url.origin() {
                    headers.clear();
                }
                url = next;
                continue;
            }
            let content_type = response
                .headers()
                .get(CONTENT_TYPE)
                .and_then(|v| v.to_str().ok())
                .map(media_type)
                .unwrap_or_default();
            if status.is_success()
                && !req.accept.is_empty()
                && !req
                    .accept
                    .iter()
                    .any(|a| a.eq_ignore_ascii_case(&content_type))
            {
                return Err(HttpError::new(
                    HttpErrorKind::Rejected,
                    format!(
                        "{url}: content type `{content_type}` is not one of {}",
                        req.accept.join(", ")
                    ),
                ));
            }
            let reply_headers = response
                .headers()
                .iter()
                .filter_map(|(k, v)| {
                    v.to_str()
                        .ok()
                        .map(|v| (k.as_str().to_owned(), v.to_owned()))
                })
                .collect();
            let max = self.limits.max_body_bytes;
            let mut body = Vec::new();
            let mut truncated = false;
            while let Some(chunk) = response
                .chunk()
                .await
                .map_err(|e| network(format!("{url}: {}", e.without_url())))?
            {
                let room = max - body.len();
                if chunk.len() > room {
                    body.extend_from_slice(&chunk[..room]);
                    truncated = true;
                    break;
                }
                body.extend_from_slice(&chunk);
            }
            assert!(body.len() <= max);
            return Ok(HttpResponse {
                status: status.as_u16(),
                url: url.to_string(),
                headers: reply_headers,
                body: Box::new(ChunkBody::new(vec![body])),
                truncated,
            });
        }
        Err(HttpError::new(
            HttpErrorKind::Rejected,
            format!("more than {} redirects", self.limits.max_redirects),
        ))
    }
}

#[cfg(feature = "reqwest")]
#[async_trait]
impl Transport for GuardedHttp {
    async fn send(&self, req: HttpRequest) -> Result<HttpResponse, HttpError> {
        let timeout = self.limits.timeout;
        tokio::time::timeout(timeout, self.follow(req))
            .await
            .map_err(|_| {
                HttpError::new(
                    HttpErrorKind::Timeout,
                    format!("timed out after {} ms", timeout.as_millis()),
                )
            })?
    }
}

#[cfg(any(feature = "reqwest", test))]
fn network(message: String) -> HttpError {
    HttpError::new(HttpErrorKind::Network, message)
}

#[cfg(feature = "reqwest")]
fn blocked(message: String) -> HttpError {
    HttpError::new(HttpErrorKind::Blocked, message)
}

#[cfg(feature = "reqwest")]
async fn pinned_addr(url: &Url, allow: fn(IpAddr) -> bool) -> Result<SocketAddr, HttpError> {
    if !matches!(url.scheme(), "http" | "https") {
        return Err(blocked(format!(
            "scheme `{}` is not http or https",
            url.scheme()
        )));
    }
    let port = url
        .port_or_known_default()
        .ok_or_else(|| blocked(format!("{url}: no port")))?;
    let addrs: Vec<SocketAddr> = match url.host() {
        Some(Host::Ipv4(ip)) => vec![SocketAddr::new(IpAddr::V4(ip), port)],
        Some(Host::Ipv6(ip)) => vec![SocketAddr::new(IpAddr::V6(ip), port)],
        Some(Host::Domain(domain)) => tokio::net::lookup_host((domain, port))
            .await
            .map_err(|e| network(format!("{domain}: {e}")))?
            .collect(),
        None => return Err(blocked(format!("{url}: no host"))),
    };
    if let Some(bad) = addrs.iter().find(|a| !allow(a.ip())) {
        return Err(blocked(format!(
            "{url}: address {} is not public",
            bad.ip()
        )));
    }
    addrs
        .first()
        .copied()
        .ok_or_else(|| network(format!("{url}: host has no address")))
}

/// Whether `ip` is a public unicast address a tool may reach.
/// Private, loopback, link-local, unspecified, multicast, broadcast, CGNAT,
/// documentation, reserved and unique-local ranges are not, nor IPv6 forms
/// that embed such an IPv4 address.
pub fn is_public_ip(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => is_public_v4(v4),
        IpAddr::V6(v6) => is_public_v6(v6),
    }
}

fn is_public_v4(ip: Ipv4Addr) -> bool {
    let [a, b, c, _] = ip.octets();
    let blocked = a == 0
        || a == 10
        || a == 127
        || (a == 100 && (b & 0xc0) == 64)
        || (a == 169 && b == 254)
        || (a == 172 && (b & 0xf0) == 16)
        || (a == 192 && b == 0 && (c == 0 || c == 2))
        || (a == 192 && b == 168)
        || (a == 198 && (b & 0xfe) == 18)
        || (a == 198 && b == 51 && c == 100)
        || (a == 203 && b == 0 && c == 113)
        || a >= 224;
    !blocked
}

fn is_public_v6(ip: Ipv6Addr) -> bool {
    if let Some(v4) = ip.to_ipv4_mapped() {
        return is_public_v4(v4);
    }
    let s = ip.segments();
    let embedded = |hi: u16, lo: u16| Ipv4Addr::from(((hi as u32) << 16) | lo as u32);
    if s[0] == 0x64 && s[1] == 0xff9b {
        return s[2..6] == [0, 0, 0, 0] && is_public_v4(embedded(s[6], s[7]));
    }
    if s[0] == 0x2002 {
        return is_public_v4(embedded(s[1], s[2]));
    }
    let global_unicast = (s[0] & 0xe000) == 0x2000;
    let teredo = s[0] == 0x2001 && s[1] == 0;
    let documentation = (s[0] == 0x2001 && s[1] == 0x0db8) || (s[0] & 0xfff0) == 0x3ff0;
    global_unicast && !teredo && !documentation
}

/// Test transport: answers from a queue and records every request.
/// An empty queue answers a `Network` error. A response with an empty
/// `url` gets the request's URL.
#[cfg(test)]
pub(crate) struct FakeHttp {
    responses: std::sync::Mutex<std::collections::VecDeque<Result<Received, HttpError>>>,
    requests: std::sync::Mutex<Vec<HttpRequest>>,
}

#[cfg(test)]
impl FakeHttp {
    /// Builds a fake that returns `responses` in order.
    pub(crate) fn new(responses: Vec<Result<Received, HttpError>>) -> std::sync::Arc<Self> {
        std::sync::Arc::new(Self {
            responses: std::sync::Mutex::new(responses.into()),
            requests: std::sync::Mutex::new(Vec::new()),
        })
    }

    /// Builds a response with `status`, media type `mime` and `body`.
    pub(crate) fn reply(status: u16, mime: &str, body: &str) -> Result<Received, HttpError> {
        Ok(Received {
            status,
            url: String::new(),
            content_type: mime.into(),
            body: body.as_bytes().to_vec(),
            truncated: false,
        })
    }

    /// Returns copies of the requests received so far.
    pub(crate) fn requests(&self) -> Vec<HttpRequest> {
        self.requests.lock().unwrap().clone()
    }
}

#[cfg(test)]
#[async_trait]
impl Transport for FakeHttp {
    async fn send(&self, req: HttpRequest) -> Result<HttpResponse, HttpError> {
        let url = req.url.clone();
        self.requests.lock().unwrap().push(req);
        let next = self
            .responses
            .lock()
            .unwrap()
            .pop_front()
            .unwrap_or_else(|| Err(network("no fake response left".into())));
        next.map(|r| HttpResponse {
            status: r.status,
            url: if r.url.is_empty() { url } else { r.url },
            headers: vec![("content-type".into(), r.content_type)],
            body: Box::new(ChunkBody::new(vec![r.body])),
            truncated: r.truncated,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "reqwest")]
    use std::net::{Ipv4Addr, SocketAddr};
    #[cfg(feature = "reqwest")]
    use std::sync::{Arc, Mutex};
    #[cfg(feature = "reqwest")]
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    #[cfg(feature = "reqwest")]
    use tokio::net::TcpListener;

    fn ip(s: &str) -> IpAddr {
        s.parse().unwrap()
    }

    #[test]
    fn blocks_non_public_addresses() {
        for bad in [
            "127.0.0.1",
            "10.1.2.3",
            "172.16.0.1",
            "192.168.1.1",
            "169.254.169.254",
            "0.0.0.0",
            "100.64.0.1",
            "224.0.0.1",
            "255.255.255.255",
            "192.0.2.1",
            "198.51.100.7",
            "203.0.113.9",
            "::1",
            "::",
            "fe80::1",
            "fc00::1",
            "fd12::1",
            "ff02::1",
            "2001:db8::1",
            "::ffff:127.0.0.1",
            "::ffff:10.0.0.1",
            "::ffff:169.254.169.254",
            "64:ff9b::a00:1",
        ] {
            assert!(!is_public_ip(ip(bad)), "{bad} must be blocked");
        }
        for good in [
            "8.8.8.8",
            "1.1.1.1",
            "2606:4700:4700::1111",
            "::ffff:8.8.8.8",
        ] {
            assert!(is_public_ip(ip(good)), "{good} must be allowed");
        }
    }

    #[cfg(feature = "reqwest")]
    #[tokio::test]
    async fn rejects_bad_schemes_and_private_hosts() {
        let http = GuardedHttp::new(HttpLimits::default());
        for url in [
            "file:///etc/passwd",
            "ftp://example.com/",
            "gopher://x/",
            "http://127.0.0.1/",
            "http://[::1]/",
            "http://10.0.0.1/",
            "http://localhost/",
        ] {
            let err = receive(&http, HttpRequest::get(url)).await.unwrap_err();
            assert!(matches!(err, ToolError::Blocked(_)), "{url}: {err}");
        }
        assert!(matches!(
            receive(&http, HttpRequest::get("not a url")).await,
            Err(ToolError::InvalidArgs(_))
        ));
    }

    #[cfg(feature = "reqwest")]
    fn test_policy(ip: IpAddr) -> bool {
        ip == IpAddr::V4(Ipv4Addr::LOCALHOST) || is_public_ip(ip)
    }

    #[cfg(feature = "reqwest")]
    fn local() -> GuardedHttp {
        GuardedHttp::with_policy(HttpLimits::default(), test_policy)
    }

    #[cfg(feature = "reqwest")]
    async fn serve(responses: Vec<String>) -> (SocketAddr, Arc<Mutex<Vec<String>>>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let seen = Arc::new(Mutex::new(Vec::new()));
        let log = seen.clone();
        tokio::spawn(async move {
            for body in responses {
                let (mut sock, _) = listener.accept().await.unwrap();
                let mut buf = vec![0u8; 8192];
                let n = sock.read(&mut buf).await.unwrap_or(0);
                log.lock()
                    .unwrap()
                    .push(String::from_utf8_lossy(&buf[..n]).to_ascii_lowercase());
                sock.write_all(body.as_bytes()).await.unwrap();
                sock.shutdown().await.ok();
            }
        });
        (addr, seen)
    }

    #[cfg(feature = "reqwest")]
    fn redirect(to: &str) -> String {
        format!("HTTP/1.1 302 Found\r\nLocation: {to}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
    }

    #[cfg(feature = "reqwest")]
    fn ok(mime: &str, body: &str) -> String {
        format!("HTTP/1.1 200 OK\r\nContent-Type: {mime}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len())
    }

    #[cfg(feature = "reqwest")]
    #[tokio::test]
    async fn redirect_to_private_address_rejected_per_hop() {
        for target in [
            "http://10.0.0.1/x",
            "http://[::ffff:192.168.0.1]/",
            "http://169.254.169.254/latest",
        ] {
            let (addr, seen) = serve(vec![redirect(target)]).await;
            let err = receive(&local(), HttpRequest::get(&format!("http://{addr}/start")))
                .await
                .unwrap_err();
            assert_eq!(seen.lock().unwrap().len(), 1, "first hop must be served");
            assert!(
                matches!(err, ToolError::Blocked(ref m) if m.contains("not public")),
                "{target}: {err}"
            );
        }
    }

    #[cfg(feature = "reqwest")]
    #[tokio::test]
    async fn follows_public_redirect() {
        let (addr, seen) = serve(vec![
            redirect("/page"),
            ok("text/html; charset=utf-8", "<p>hi</p>"),
        ])
        .await;
        let mut req = HttpRequest::get(&format!("http://{addr}/start"));
        req.accept = vec!["text/html".into()];
        let got = receive(&local(), req).await.unwrap();
        assert_eq!(seen.lock().unwrap().len(), 2);
        assert_eq!(
            (got.status, got.content_type.as_str(), got.body.as_slice()),
            (200, "text/html", b"<p>hi</p>".as_slice())
        );
        assert!(got.url.ends_with("/page") && !got.truncated);
    }

    #[cfg(feature = "reqwest")]
    #[tokio::test]
    async fn too_many_redirects_rejected() {
        let (addr, _) = serve((0..7).map(|_| redirect("/again")).collect()).await;
        let err = receive(&local(), HttpRequest::get(&format!("http://{addr}/")))
            .await
            .unwrap_err();
        assert!(
            matches!(err, ToolError::Http(ref m) if m.contains("redirects")),
            "{err}"
        );
    }

    #[cfg(feature = "reqwest")]
    #[tokio::test]
    async fn cross_origin_redirect_drops_headers() {
        let (second, seen_second) = serve(vec![ok("text/plain", "done")]).await;
        let (first, seen_first) = serve(vec![
            redirect("/same"),
            redirect(&format!("http://{second}/other")),
        ])
        .await;
        let mut req = HttpRequest::get(&format!("http://{first}/start"));
        req.headers.push(("X-Key".into(), "k-123".into()));
        let got = receive(&local(), req).await.unwrap();
        assert_eq!(got.body, b"done");
        let first_log = seen_first.lock().unwrap().clone();
        assert!(
            first_log.iter().all(|r| r.contains("x-key: k-123")),
            "same origin keeps headers"
        );
        assert!(
            !seen_second.lock().unwrap()[0].contains("x-key"),
            "other origin must not see headers"
        );
    }

    #[cfg(feature = "reqwest")]
    #[tokio::test]
    async fn body_is_capped_and_flagged() {
        let (addr, _) = serve(vec![ok("text/plain", &"a".repeat(3000))]).await;
        let limits = HttpLimits {
            max_body_bytes: 1024,
            ..HttpLimits::default()
        };
        let got = receive(
            &GuardedHttp::with_policy(limits, test_policy),
            HttpRequest::get(&format!("http://{addr}/")),
        )
        .await
        .unwrap();
        assert_eq!(got.body.len(), 1024);
        assert!(got.truncated);
    }

    #[cfg(feature = "reqwest")]
    #[tokio::test]
    async fn rejects_unaccepted_content_type() {
        let (addr, _) = serve(vec![ok("image/png", "png")]).await;
        let mut req = HttpRequest::get(&format!("http://{addr}/"));
        req.accept = vec!["text/html".into()];
        let err = receive(&local(), req).await.unwrap_err();
        assert!(
            matches!(err, ToolError::Http(ref m) if m.contains("image/png")),
            "{err}"
        );
    }

    #[cfg(feature = "reqwest")]
    #[tokio::test]
    async fn non_get_redirect_is_returned_not_followed() {
        let (addr, seen) = serve(vec![redirect("/elsewhere")]).await;
        let req = HttpRequest {
            method: Method::Post,
            body: Some(b"{}".to_vec()),
            ..HttpRequest::get(&format!("http://{addr}/"))
        };
        let got = receive(&local(), req).await.unwrap();
        assert_eq!(got.status, 302);
        assert_eq!(seen.lock().unwrap().len(), 1);
    }

    #[cfg(feature = "reqwest")]
    #[tokio::test]
    async fn times_out() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let (_sock, _) = listener.accept().await.unwrap();
            tokio::time::sleep(Duration::from_secs(5)).await;
        });
        let limits = HttpLimits {
            timeout: Duration::from_millis(200),
            ..HttpLimits::default()
        };
        let err = receive(
            &GuardedHttp::with_policy(limits, test_policy),
            HttpRequest::get(&format!("http://{addr}/")),
        )
        .await
        .unwrap_err();
        assert!(
            matches!(err, ToolError::Http(ref m) if m.contains("timed out")),
            "{err}"
        );
    }

    #[test]
    fn request_debug_hides_header_values() {
        let mut req = HttpRequest::get("https://example.com/");
        req.headers
            .push(("Authorization".into(), "Bearer k-123".into()));
        let dbg = format!("{req:?}");
        assert!(
            dbg.contains("Authorization") && !dbg.contains("k-123"),
            "{dbg}"
        );
    }
}
