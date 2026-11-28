use crate::support::{lock, num, Collect};
use serde_json::Value as Json;
use std::sync::Arc;
use troupekit_trace::{trace, trace_span, Trace, TraceSet, Tracer};

fn tracer(sink: &Arc<Collect>) -> Tracer {
    Tracer::builder()
        .categories(TraceSet::all())
        .sink(sink.clone())
        .name("run")
        .build()
        .unwrap()
}

fn assert_keys(record: &Json, required: &[&str], optional: &[&str]) {
    let object = record.as_object().expect("record is an object");
    for key in required {
        assert!(object.contains_key(*key), "{key} missing in {record}");
    }
    for key in object.keys() {
        let known =
            key == "t" || required.contains(&key.as_str()) || optional.contains(&key.as_str());
        assert!(known, "unexpected key {key} in {record}");
    }
    assert_eq!(record["v"], 1);
    let trace = record["trace"].as_str().unwrap();
    assert_eq!(trace.len(), 16);
    assert!(trace
        .bytes()
        .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)));
}

#[test]
fn record_lines_parse_as_json() {
    let _guard = lock();
    let sink = Collect::new();
    let t = tracer(&sink);
    t.enter(|| {
        let mut span = trace_span!(Trace::Models, "model.call", model = "m", n = 3u64);
        span.enter(|| trace!(Trace::Laws, "law.check", held = true, score = 0.5));
        span.record("tokens", 12u64);
    });
    drop(t);
    let records = sink.records();
    let kinds: Vec<&str> = records.iter().map(|r| r["t"].as_str().unwrap()).collect();
    assert_eq!(kinds, ["header", "start", "event", "end", "footer"]);
    let point = ["v", "trace", "span", "parent", "cat", "name", "ts_us"];
    assert_keys(
        &records[0],
        &[
            "v",
            "trace",
            "name",
            "started_us",
            "categories",
            "content",
            "troupekit",
        ],
        &[],
    );
    assert_keys(&records[1], &point, &["fields"]);
    assert_keys(&records[2], &point, &["fields"]);
    assert_keys(
        &records[3],
        &["v", "trace", "span", "ts_us", "dur_us", "ok"],
        &["err", "fields"],
    );
    assert_keys(&records[4], &["v", "trace", "dropped", "truncated"], &[]);
    assert_eq!(records[0]["categories"].as_array().unwrap().len(), 11);
    assert_eq!(records[1]["cat"], "models");
    assert_eq!(records[1]["fields"]["n"], 3);
    assert_eq!(num(&records[2], "span"), 0);
    assert_eq!(records[2]["fields"]["score"], 0.5);
    assert_eq!(records[3]["fields"]["tokens"], 12);
}

#[test]
fn escapes_quotes_controls_and_line_separators() {
    let _guard = lock();
    let sink = Collect::new();
    let t = tracer(&sink);
    let hostile = "q\"b\\s\n\r\t\u{0}\u{1b}[2J\u{2028}\u{2029}end";
    t.enter(|| trace!(Trace::Tools, "tool.call", text = hostile, nan = f64::NAN));
    drop(t);
    let line = sink
        .lines()
        .into_iter()
        .find(|l| l.contains("tool.call"))
        .unwrap();
    assert!(
        !line.bytes().any(|b| b < 0x20),
        "raw control byte in {line}"
    );
    assert!(!line.contains('\u{2028}') && !line.contains('\u{2029}'));
    assert!(line.contains("\\u2028") && line.contains("\\u001b"));
    let record = sink.named("tool.call");
    assert_eq!(record["fields"]["text"], hostile);
    assert!(record["fields"]["nan"].is_null());
}
