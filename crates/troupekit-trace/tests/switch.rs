use crate::support::{lock, Collect};
use std::sync::atomic::{AtomicU32, Ordering};
use troupekit_trace::{enabled, live_categories, trace, Trace, TraceSet, Tracer};

fn bump(calls: &AtomicU32) -> u32 {
    calls.fetch_add(1, Ordering::SeqCst)
}

fn tracer(categories: TraceSet) -> Tracer {
    Tracer::builder()
        .categories(categories)
        .sink(Collect::new())
        .build()
        .unwrap()
}

#[test]
fn disabled_point_evaluates_no_field() {
    let _guard = lock();
    let calls = AtomicU32::new(0);
    assert_eq!(live_categories(), TraceSet::none());
    for t in Trace::ALL {
        assert!(!enabled(t), "{t} enabled with no tracer");
    }
    trace!(Trace::Laws, "law.check", n = bump(&calls));
    let other = tracer(TraceSet::none().with(Trace::Tools));
    other.enter(|| trace!(Trace::Laws, "law.check", n = bump(&calls)));
    trace!(Trace::Tools, "tool.call", n = bump(&calls));
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}

#[test]
fn mask_is_union_of_live_tracers() {
    let _guard = lock();
    let tools = tracer(TraceSet::none().with(Trace::Tools));
    let laws = tracer(TraceSet::none().with(Trace::Laws));
    assert_eq!(
        live_categories(),
        TraceSet::none().with(Trace::Tools).with(Trace::Laws)
    );
    drop(tools);
    assert_eq!(live_categories(), TraceSet::none().with(Trace::Laws));
    drop(laws);
}

#[test]
fn dropping_last_tracer_clears_mask() {
    let _guard = lock();
    let first = tracer(TraceSet::all());
    let second = first.clone();
    drop(first);
    assert_eq!(live_categories(), TraceSet::all());
    drop(second);
    assert_eq!(live_categories(), TraceSet::none());
}

#[test]
fn builder_without_sink_is_an_error() {
    let _guard = lock();
    assert!(Tracer::builder().build().is_err());
    assert_eq!(live_categories(), TraceSet::none());
}

#[test]
fn trace_names_round_trip() {
    let mut seen = std::collections::BTreeSet::new();
    for t in Trace::ALL {
        let name = t.as_str();
        assert!(
            name.bytes().all(|b| b.is_ascii_lowercase() || b == b'_'),
            "{name}"
        );
        assert!(seen.insert(name), "duplicate name {name}");
        assert_eq!(name.parse::<Trace>().unwrap(), t);
    }
    assert_eq!(seen.len(), 11);
    assert!("nope".parse::<Trace>().is_err());
    assert!("Tools".parse::<Trace>().is_err());
    let set = TraceSet::none().with(Trace::Routing).with(Trace::Agents);
    assert_eq!(TraceSet::from_bits(set.bits()), set);
    assert_eq!(set.iter().collect::<TraceSet>(), set);
    assert_eq!(TraceSet::from_bits(u32::MAX), TraceSet::all());
}

#[test]
fn spec_parses_toml_values() {
    use troupekit_trace::TraceSpec;
    assert_eq!(
        TraceSpec::parse_value(["all"]).unwrap().categories(),
        TraceSet::all()
    );
    assert_eq!(
        TraceSpec::parse_value(["true"]).unwrap().categories(),
        TraceSet::all()
    );
    assert_eq!(
        TraceSpec::parse_value(["false"]).unwrap().categories(),
        TraceSet::none()
    );
    let two = TraceSpec::parse_value(["tools", "laws"]).unwrap();
    assert_eq!(
        two.categories(),
        TraceSet::none().with(Trace::Tools).with(Trace::Laws)
    );
    assert!(!two.content());
    assert_eq!(two.file(), None);
    assert!(TraceSpec::parse_value(["tools", "bogus"]).is_err());
    assert!(TraceSpec::parse_value(vec!["tools"; 65]).is_err());
    let spec = TraceSpec::new(TraceSet::all())
        .with_content(true)
        .with_file("traces/a.jsonl");
    assert!(spec.content());
    assert_eq!(spec.file(), Some("traces/a.jsonl"));
}
