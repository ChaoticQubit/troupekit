use std::hint::black_box;
use std::sync::Arc;
use std::time::Instant;
use troupekit_trace::{trace, MemorySink, Trace, TraceSet, Tracer};

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

/// Runs `one_run` `RUNS` times and prints the median and the spread.
/// Input: the row label. Output: one printed line.
fn measure(label: &str) {
    let mut runs: Vec<f64> = (0..RUNS).map(|_| one_run()).collect();
    runs.sort_by(f64::total_cmp);
    assert_eq!(runs.len(), RUNS);
    let spread = runs[RUNS - 1] - runs[0];
    println!(
        "{label:<44} median {:>7.3} ns  spread {spread:.3} ns",
        runs[RUNS / 2]
    );
}

/// Measures the cost of one disabled trace point in three states: no
/// tracer, a live tracer on another category, and a live tracer on the
/// same category with the point outside its scope.
fn main() {
    println!("calls per run {CALLS}, runs {RUNS}");
    measure("no tracer alive");
    let sink = Arc::new(MemorySink::new(16));
    let other = Tracer::builder()
        .categories(TraceSet::none().with(Trace::Tools))
        .sink(sink.clone())
        .build()
        .expect("tracer with a sink");
    measure("tracer alive on another category");
    drop(other);
    let same = Tracer::builder()
        .categories(TraceSet::none().with(Trace::Laws))
        .sink(sink)
        .build()
        .expect("tracer with a sink");
    measure("tracer alive on the same category, no scope");
    drop(same);
}
