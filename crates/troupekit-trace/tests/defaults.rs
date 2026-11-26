use crate::support::{lock, Collect};
use troupekit_trace::view::{TermOptions, ViewLimits};
use troupekit_trace::{JsonlOptions, MemorySink, TraceSet, TraceSpec, Tracer};

#[test]
fn defaults_are_the_safe_values() {
    let _guard = lock();
    let sink = Collect::new();
    let t = Tracer::builder().sink(sink.clone()).build().unwrap();
    assert_eq!(t.categories(), TraceSet::all());
    drop(t);
    let header = &sink.of("header")[0];
    assert_eq!(header["content"], false);
    assert_eq!(header["name"], "trace");
    let spec = TraceSpec::default();
    assert_eq!(spec.categories(), TraceSet::all());
    assert!(!spec.content());
    assert_eq!(spec.file(), None);
    let jsonl = JsonlOptions::default()
        .with_append(false)
        .with_max_bytes(256 * 1024 * 1024)
        .with_flush_every(64);
    assert_eq!(JsonlOptions::default(), jsonl);
    assert_eq!(MemorySink::default().capacity(), 100_000);
    let limits = ViewLimits::default()
        .with_line_bytes_max(1024 * 1024)
        .with_records_max(1_000_000)
        .with_depth_max(32);
    assert_eq!(ViewLimits::default(), limits);
    let term = TermOptions::default()
        .with_width(100)
        .with_color(false)
        .with_only(TraceSet::all())
        .with_max_rows(2_000);
    assert_eq!(TermOptions::default(), term);
}
