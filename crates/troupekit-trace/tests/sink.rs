use crate::support::{lock, scratch_dir};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use troupekit_trace::{
    default_trace_path, trace_span, CallbackSink, JsonlOptions, JsonlSink, MemorySink, Trace,
    TraceSet, TraceSink, Tracer,
};

#[cfg(unix)]
#[test]
fn jsonl_file_is_private() {
    use std::os::unix::fs::PermissionsExt;
    let dir = scratch_dir("private");
    let path = dir.join("a").join("b").join("t.jsonl");
    let sink = JsonlSink::create(&path).unwrap();
    sink.write("{}");
    drop(sink);
    let mode = |p: &std::path::Path| std::fs::metadata(p).unwrap().permissions().mode() & 0o777;
    assert_eq!(mode(&path), 0o600);
    assert_eq!(mode(&dir.join("a")), 0o700);
    assert_eq!(mode(&dir.join("a").join("b")), 0o700);
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn jsonl_refuses_existing_file_without_append() {
    let dir = scratch_dir("existing");
    let path = dir.join("t.jsonl");
    let first = JsonlSink::create(&path).unwrap();
    first.write("{\"n\":1}");
    drop(first);
    assert!(JsonlSink::create(&path).is_err());
    let again = JsonlSink::with_options(&path, JsonlOptions::default().with_append(true)).unwrap();
    again.write("{\"n\":2}");
    drop(again);
    assert_eq!(
        std::fs::read_to_string(&path).unwrap(),
        "{\"n\":1}\n{\"n\":2}\n"
    );
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn jsonl_stops_at_max_bytes_with_one_truncated_record() {
    let dir = scratch_dir("max");
    let path = dir.join("t.jsonl");
    let options = JsonlOptions::default().with_max_bytes(1_000);
    let sink = JsonlSink::with_options(&path, options).unwrap();
    let line = format!(
        "{{\"v\":1,\"t\":\"event\",\"trace\":\"00000000000000ab\",\"x\":\"{}\"}}",
        "y".repeat(40)
    );
    for _ in 0..100 {
        sink.write(&line);
    }
    drop(sink);
    let text = std::fs::read_to_string(&path).unwrap();
    let lines: Vec<&str> = text.lines().collect();
    let truncated: Vec<&&str> = lines
        .iter()
        .filter(|l| l.contains("\"t\":\"truncated\""))
        .collect();
    assert_eq!(truncated.len(), 1);
    assert_eq!(lines.last(), truncated.first().copied());
    let record: serde_json::Value = serde_json::from_str(truncated[0]).unwrap();
    assert_eq!(record["trace"], "00000000000000ab");
    assert!(record["ts_us"].is_u64());
    let body: usize = lines[..lines.len() - 1].iter().map(|l| l.len() + 1).sum();
    assert!(body <= 1_000);
    assert!(body + line.len() + 1 > 1_000);
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn memory_sink_drops_oldest_and_counts() {
    let sink = MemorySink::new(3);
    for i in 1..=5 {
        sink.write(&i.to_string());
    }
    assert_eq!(sink.lines(), ["3", "4", "5"]);
    assert_eq!(sink.dropped(), 2);
    assert_eq!(MemorySink::new(0).capacity(), 1);
    assert_eq!(MemorySink::default().capacity(), 100_000);
}

#[test]
fn panicking_callback_never_fails_the_run() {
    let _guard = lock();
    let calls = Arc::new(AtomicU32::new(0));
    let seen = calls.clone();
    let sink = Arc::new(CallbackSink::new(move |_line: &str| {
        if seen.fetch_add(1, Ordering::SeqCst) == 1 {
            panic!("callback failed");
        }
    }));
    let t = Tracer::builder()
        .categories(TraceSet::all())
        .sink(sink.clone())
        .build()
        .unwrap();
    let finished = t.enter(|| {
        for _ in 0..3 {
            drop(trace_span!(Trace::Tools, "tool.call"));
        }
        true
    });
    drop(t);
    assert!(finished);
    assert_eq!(sink.panics(), 1);
    assert!(sink.is_disabled());
    assert_eq!(calls.load(Ordering::SeqCst), 2);
}

#[test]
fn root_span_end_flushes_the_file() {
    let _guard = lock();
    let dir = scratch_dir("flush");
    let path = dir.join("t.jsonl");
    let options = JsonlOptions::default().with_flush_every(1_000_000);
    let sink = Arc::new(JsonlSink::with_options(&path, options).unwrap());
    let t = Tracer::builder().sink(sink.clone()).build().unwrap();
    t.enter(|| drop(trace_span!(Trace::Agents, "agent.run")));
    let text = std::fs::read_to_string(&path).unwrap();
    assert_eq!(
        text.lines().filter(|l| l.contains("\"t\":\"end\"")).count(),
        1
    );
    drop(t);
    drop(sink);
    assert!(std::fs::read_to_string(&path)
        .unwrap()
        .contains("\"t\":\"footer\""));
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn default_path_sanitizes_name() {
    let path = default_trace_path("My Agent/../x\u{e9}", 0xab);
    let file = path.file_name().unwrap().to_str().unwrap();
    assert_eq!(file, "_y__gent____x__-00000000000000ab.jsonl");
    let empty = default_trace_path("", 1);
    assert_eq!(empty.file_name().unwrap(), "trace-0000000000000001.jsonl");
    let long = default_trace_path(&"a".repeat(500), 1);
    assert_eq!(
        long.file_name().unwrap().len(),
        64 + "-0000000000000001.jsonl".len()
    );
}
