use crate::support::{lock, num, poll_once, Collect, YieldOnce};
use std::sync::{Arc, Barrier};
use std::task::Poll;
use troupekit_trace::{current, enabled, trace, trace_span, Trace, TraceSet, Tracer};

fn tracer(sink: &Arc<Collect>, categories: TraceSet) -> Tracer {
    Tracer::builder()
        .categories(categories)
        .sink(sink.clone())
        .name("t")
        .build()
        .unwrap()
}

#[test]
fn scope_follows_task_across_threads() {
    let _guard = lock();
    let sink = Collect::new();
    let t = tracer(&sink, TraceSet::all());
    let mut fut = Box::pin(t.in_scope(async {
        let span = trace_span!(Trace::Models, "model.call");
        span.instrument(async {
            YieldOnce::default().await;
            trace!(Trace::Models, "model.http");
        })
        .await;
        drop(span);
    }));
    assert!(poll_once(fut.as_mut()).is_pending());
    assert!(current().is_none(), "scope leaked onto the polling thread");
    let done = std::thread::spawn(move || {
        let ready = matches!(poll_once(fut.as_mut()), Poll::Ready(()));
        (ready, current().is_none())
    });
    assert_eq!(done.join().unwrap(), (true, true));
    drop(t);
    let span = sink.named("model.call");
    let event = sink.named("model.http");
    assert_eq!(num(&event, "parent"), num(&span, "span"));
    assert_eq!(event["trace"], span["trace"]);
}

#[test]
fn untraced_task_records_nothing_while_another_is_traced() {
    let _guard = lock();
    let sink = Collect::new();
    let t = tracer(&sink, TraceSet::all());
    let barrier = Arc::new(Barrier::new(2));
    let gate = barrier.clone();
    let traced = t.clone();
    let left = std::thread::spawn(move || {
        traced.enter(|| {
            gate.wait();
            for _ in 0..100 {
                trace!(Trace::Tools, "traced.point");
            }
            gate.wait();
        })
    });
    barrier.wait();
    let mut leaked = 0;
    for _ in 0..100 {
        leaked += usize::from(enabled(Trace::Tools));
        trace!(Trace::Tools, "untraced.point");
    }
    barrier.wait();
    left.join().unwrap();
    let trace_id = format!("{:016x}", t.trace_id());
    drop(t);
    assert_eq!(leaked, 0);
    let records = sink.records();
    assert!(records.iter().all(|r| r["trace"] == trace_id.as_str()));
    assert!(records.iter().all(|r| r["name"] != "untraced.point"));
    assert_eq!(
        records
            .iter()
            .filter(|r| r["name"] == "traced.point")
            .count(),
        100
    );
}

#[test]
fn current_reenters_inside_spawn_blocking() {
    let _guard = lock();
    let sink = Collect::new();
    let t = tracer(&sink, TraceSet::all());
    t.enter(|| {
        let span = trace_span!(Trace::Store, "checkpoint.save");
        span.enter(|| {
            let ctx = current().expect("scope is active");
            std::thread::spawn(move || ctx.enter(|| trace!(Trace::Store, "blocking.inner")))
                .join()
                .unwrap();
        });
    });
    drop(t);
    let parent = num(&sink.named("checkpoint.save"), "span");
    assert_eq!(num(&sink.named("blocking.inner"), "parent"), parent);
}

#[test]
fn nested_scope_unions_categories() {
    let _guard = lock();
    let outer_sink = Collect::new();
    let inner_sink = Collect::new();
    let outer = tracer(&outer_sink, TraceSet::none().with(Trace::Tools));
    let inner = Tracer::builder()
        .categories(TraceSet::none().with(Trace::Laws))
        .content(true)
        .sink(inner_sink.clone())
        .build()
        .unwrap();
    outer.enter(|| {
        assert!(enabled(Trace::Tools));
        assert!(!enabled(Trace::Laws));
        assert!(!current().unwrap().content());
        inner.enter(|| {
            assert!(enabled(Trace::Tools));
            assert!(enabled(Trace::Laws));
            assert!(current().unwrap().content());
            trace!(Trace::Laws, "law.check", held = true);
        });
        let nested = current()
            .unwrap()
            .nest(TraceSet::none().with(Trace::Budget), None);
        nested.enter(|| assert!(enabled(Trace::Budget) && enabled(Trace::Tools)));
    });
    let outer_id = format!("{:016x}", outer.trace_id());
    drop((outer, inner));
    assert_eq!(outer_sink.named("law.check")["trace"], outer_id.as_str());
    assert!(inner_sink
        .records()
        .iter()
        .all(|r| r["t"] == "header" || r["t"] == "footer"));
}

#[test]
fn span_parent_is_enclosing_span() {
    let _guard = lock();
    let sink = Collect::new();
    let t = tracer(&sink, TraceSet::all());
    t.enter(|| {
        let outer = trace_span!(Trace::Agents, "agent.run", run = "r1");
        outer.enter(|| {
            let mut inner = trace_span!(Trace::Agents, "agent.step", step = 1u64);
            inner.record("state", "thinking");
            inner.fail("boom");
        });
    });
    drop(t);
    let outer = sink.named("agent.run");
    let inner = sink.named("agent.step");
    assert_eq!(num(&outer, "parent"), 0);
    assert_eq!(num(&inner, "parent"), num(&outer, "span"));
    assert!(num(&inner, "span") > num(&outer, "span"));
    let ends = sink.of("end");
    assert_eq!(ends.len(), 2);
    assert_eq!(num(&ends[0], "span"), num(&inner, "span"));
    assert_eq!(ends[0]["ok"], false);
    assert_eq!(ends[0]["err"], "boom");
    assert_eq!(ends[0]["fields"]["state"], "thinking");
    assert_eq!(ends[1]["ok"], true);
}

#[test]
fn span_outside_scope_is_disabled() {
    let _guard = lock();
    let sink = Collect::new();
    let t = tracer(&sink, TraceSet::all());
    let span = trace_span!(Trace::Models, "model.call");
    assert!(!span.is_enabled());
    assert_eq!(span.id(), 0);
    drop(span);
    drop(t);
    assert!(sink
        .records()
        .iter()
        .all(|r| r["t"] == "header" || r["t"] == "footer"));
}
