use std::future::{ready, Future};
use std::hint::black_box;
use std::pin::pin;
use std::sync::Arc;
use std::task::{Context, Poll, Wake, Waker};
use std::time::Instant;
use troupekit_trace::{trace, trace_span, MemorySink, Trace, TraceSet, Tracer};

const CALLS: u64 = 10_000_000;
const RUNS: usize = 3;

/// Times `CALLS` disabled `trace!` points and returns nanoseconds per call.
/// Input: none. Output: the mean cost of one point in this run.
/// The field expression is `black_box` so it would show if it ran.
fn one_run() -> f64 {
    let started = Instant::now();
    for i in 0..CALLS {
        trace!(Trace::Laws, "law.check", n = black_box(i));
    }
    started.elapsed().as_nanos() as f64 / CALLS as f64
}

struct Noop;

impl Wake for Noop {
    fn wake(self: Arc<Self>) {}
}

/// Times `CALLS` disabled `trace_span!` plus `Span::instrument` around a
/// ready future, polled once each. Input: none. Output: ns per call.
/// Shows whether a disabled span costs an allocation.
fn one_span_run() -> f64 {
    let waker = Waker::from(Arc::new(Noop));
    let mut cx = Context::from_waker(&waker);
    let started = Instant::now();
    for i in 0..CALLS {
        let span = trace_span!(Trace::Models, "model.call", n = black_box(i));
        let fut = pin!(span.instrument(ready(i)));
        assert!(matches!(fut.poll(&mut cx), Poll::Ready(_)));
    }
    started.elapsed().as_nanos() as f64 / CALLS as f64
}

/// Runs `run` `RUNS` times and prints the median and the spread.
/// Input: the row label and the timed function. Output: one printed line.
fn measure(label: &str, run: fn() -> f64) {
    let mut runs: Vec<f64> = (0..RUNS).map(|_| run()).collect();
    runs.sort_by(f64::total_cmp);
    assert_eq!(runs.len(), RUNS);
    let spread = runs[RUNS - 1] - runs[0];
    println!(
        "{label:<44} median {:>7.3} ns  spread {spread:.3} ns",
        runs[RUNS / 2]
    );
}

/// Measures the cost of one disabled trace point in three states (no
/// tracer, a live tracer on another category, a live tracer on the same
/// category outside its scope) and of one disabled span with instrument.
fn main() {
    println!("calls per run {CALLS}, runs {RUNS}");
    measure("no tracer alive", one_run);
    measure("disabled span and instrument, no tracer", one_span_run);
    let sink = Arc::new(MemorySink::new(16));
    let other = Tracer::builder()
        .categories(TraceSet::none().with(Trace::Tools))
        .sink(sink.clone())
        .build()
        .expect("tracer with a sink");
    measure("tracer alive on another category", one_run);
    drop(other);
    let same = Tracer::builder()
        .categories(TraceSet::none().with(Trace::Laws))
        .sink(sink)
        .build()
        .expect("tracer with a sink");
    measure("tracer alive on the same category, no scope", one_run);
    drop(same);
}
