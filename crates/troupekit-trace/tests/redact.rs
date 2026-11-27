use crate::support::{lock, num, Collect};
use std::sync::Arc;
use troupekit_trace::{content, trace, trace_span, url_for_trace, Trace, TraceSet, Tracer};

const SECRET: &str = "sk-canary-0123456789";

fn tracer(sink: &Arc<Collect>, content: bool) -> Tracer {
    Tracer::builder()
        .categories(TraceSet::all())
        .content(content)
        .scrub(SECRET)
        .scrub("")
        .sink(sink.clone())
        .name(&format!("run-{SECRET}"))
        .build()
        .unwrap()
}

#[test]
fn scrubber_replaces_registered_secret_in_every_field() {
    let _guard = lock();
    let sink = Collect::new();
    let t = tracer(&sink, true);
    let inside = format!("Bearer{SECRET}tail");
    t.enter(|| {
        let mut span = trace_span!(Trace::Models, "model.call", url = inside.as_str());
        span.record(&format!("k{SECRET}"), "v");
        span.record("reply", content(format!("echo {SECRET}")));
        span.fail(format!("401 for key {SECRET}"));
        trace!(Trace::Tools, "tool.call", args = content(inside.clone()));
    });
    drop(t);
    let lines = sink.lines();
    assert!(lines.len() >= 5);
    for line in &lines {
        assert!(!line.contains(SECRET), "secret leaked in {line}");
        assert!(!line.contains("canary"), "secret part leaked in {line}");
    }
    let start = sink.named("model.call");
    assert_eq!(start["fields"]["url"], "Bearer[redacted]tail");
    assert!(lines.iter().any(|l| l.contains("\"k[redacted]\"")));
}

#[test]
fn denylisted_field_names_are_redacted() {
    let _guard = lock();
    let sink = Collect::new();
    let t = tracer(&sink, true);
    t.enter(|| {
        let mut span = trace_span!(Trace::Models, "model.call", api_key = "abc", tokens = 7u64);
        span.record("Authorization", "Bearer xyz");
        span.record("X-Api-Key", "k1");
        span.record("user_PASSWORD", content("hunter2"));
        span.record("session_cookie", "c=1");
        span.record("client_secret", "s");
        span.record("refresh_token", "r");
        span.record("model", "small");
    });
    drop(t);
    let start = sink.named("model.call");
    assert_eq!(start["fields"]["api_key"], "[redacted]");
    assert_eq!(start["fields"]["tokens"], 7);
    let end = &sink.of("end")[0];
    for key in [
        "Authorization",
        "X-Api-Key",
        "user_PASSWORD",
        "session_cookie",
        "client_secret",
        "refresh_token",
    ] {
        assert_eq!(end["fields"][key], "[redacted]", "{key}");
    }
    assert_eq!(end["fields"]["model"], "small");
}

#[test]
fn content_fields_absent_without_trace_content() {
    let _guard = lock();
    let hidden = Collect::new();
    let t = tracer(&hidden, false);
    t.enter(|| {
        trace!(
            Trace::Memory,
            "memory.remember",
            fact = content("likes tea"),
            n = 1u64
        )
    });
    drop(t);
    let record = hidden.named("memory.remember");
    assert_eq!(record["fields"]["fact"]["bytes"], 9);
    assert!(hidden.lines().iter().all(|l| !l.contains("likes tea")));
    let shown = Collect::new();
    let t = tracer(&shown, true);
    t.enter(|| {
        trace!(
            Trace::Memory,
            "memory.remember",
            fact = content("likes tea")
        )
    });
    drop(t);
    assert_eq!(
        shown.named("memory.remember")["fields"]["fact"],
        "likes tea"
    );
}

#[test]
fn long_strings_are_cut_at_char_boundary() {
    let _guard = lock();
    let sink = Collect::new();
    let t = tracer(&sink, true);
    let long = "\u{e9}".repeat(10_000) + "x";
    t.enter(|| trace!(Trace::Knowledge, "knowledge.search", query = long.as_str()));
    drop(t);
    let value = sink.named("knowledge.search")["fields"]["query"]
        .as_str()
        .unwrap()
        .to_owned();
    let (kept, tail) = value.split_once("...[cut ").expect("cut marker");
    assert!(kept.len() <= 16 * 1024);
    assert!(kept.len() > 16 * 1024 - 4);
    assert!(kept.chars().all(|c| c == '\u{e9}'));
    assert_eq!(tail, format!("{} bytes]", long.len() - kept.len()));
    assert_eq!(num(&sink.of("footer")[0], "truncated"), 1);
}

#[test]
fn field_count_and_name_length_are_capped() {
    let _guard = lock();
    let sink = Collect::new();
    let t = tracer(&sink, false);
    t.enter(|| {
        let mut span = trace_span!(Trace::Graphs, "graph.step");
        for i in 0..40u64 {
            span.record(&format!("f{i}"), i);
        }
        span.record(&"n".repeat(100), 1u64);
    });
    drop(t);
    let end = &sink.of("end")[0];
    assert_eq!(end["fields"].as_object().unwrap().len(), 32);
    assert_eq!(num(&sink.of("footer")[0], "dropped"), 9);
    let sink = Collect::new();
    let t = tracer(&sink, false);
    t.enter(|| {
        let mut span = trace_span!(Trace::Graphs, "graph.step");
        span.record(&"n".repeat(100), 1u64);
    });
    drop(t);
    let end = &sink.of("end")[0];
    let names: Vec<&String> = end["fields"].as_object().unwrap().keys().collect();
    assert_eq!(names, [&"n".repeat(64)]);
}

#[test]
fn url_for_trace_drops_query_and_userinfo() {
    let cases = [
        (
            "https://user:pw@api.example.com:8443/v1/chat?key=abc#frag",
            "https://api.example.com:8443/v1/chat",
        ),
        ("http://example.com", "http://example.com"),
        ("https://example.com/a/b?x=1", "https://example.com/a/b"),
        ("https://example.com#top", "https://example.com"),
        ("https://a@b@example.com/p", "https://example.com/p"),
        ("/relative/path?token=1", "/relative/path"),
        ("user@host/path?q", "host/path"),
        ("https:///alice:pw@example.com/v1", "https://example.com/v1"),
        ("https:////alice:pw@example.com", "https://example.com"),
        ("https:/alice:pw@example.com/v1", "https://example.com/v1"),
        ("https:\\\\alice:pw@example.com\\v1", "https://example.com\\v1"),
        ("HTTPS://alice@example.com/p", "HTTPS://example.com/p"),
        ("", ""),
    ];
    for (input, expected) in cases {
        assert_eq!(url_for_trace(input), expected, "{input}");
    }
}
