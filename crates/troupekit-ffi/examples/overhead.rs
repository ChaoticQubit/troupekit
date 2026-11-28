use std::future::Future;
use std::pin::pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::task::{Context, Poll, Wake, Waker};
use std::thread::{self, Thread};
use std::time::Instant;
use troupekit_ffi_core::{
    AgentBuilder, AgentHandle, EventListener, FfiEvent, FfiEventKind, HostError, HostModel,
    HostModelCaps, HostReply, HostRequest, HostTool, RuntimeSettings, TokenSink,
};

const ITERATIONS_DEFAULT: usize = 10_000;
const ITERATIONS_MAX: usize = 1_000_000;
const TOOL_OUTPUT: &str = "noop-ok";
const TOOL_CALL: &str = r#"{"tool": "noop", "arguments": {}}"#;
const ANSWER: &str = r#"{"answer": "done"}"#;

/// Wakes the thread that polls a future, for `block_on`.
/// Input: the thread to unpark. Output: none.
/// The neutral futures only await a tokio `JoinHandle` (decision P4).
struct ThreadWaker(Thread);

impl Wake for ThreadWaker {
    fn wake(self: Arc<Self>) {
        self.0.unpark();
    }
}

/// Drives one neutral future on the calling thread.
/// Input: the future. Output: its value.
/// Same park and unpark loop as `troupekit-py`'s `block.rs`.
fn block_on<F: Future>(future: F) -> F::Output {
    let waker = Waker::from(Arc::new(ThreadWaker(thread::current())));
    let mut context = Context::from_waker(&waker);
    let mut future = pin!(future);
    loop {
        match future.as_mut().poll(&mut context) {
            Poll::Ready(value) => return value,
            Poll::Pending => thread::park(),
        }
    }
}

/// A host model that calls the no-op tool once per run, then answers.
/// Input: none. Output: emulated tool calls (K8); stateless across runs,
/// so one handle serves every iteration.
struct BenchModel;

impl HostModel for BenchModel {
    fn id(&self) -> String {
        "bench".to_owned()
    }

    fn capabilities(&self) -> HostModelCaps {
        let mut caps = HostModelCaps::default();
        caps.tools = false;
        caps.streaming = true;
        caps.context_tokens = 4096;
        caps
    }

    fn generate(&self, request: HostRequest, sink: Arc<TokenSink>) -> Result<HostReply, HostError> {
        let called = request
            .messages
            .iter()
            .any(|message| message.text.contains(TOOL_OUTPUT));
        let text = if called { ANSWER } else { TOOL_CALL };
        sink.delta(text.to_owned());
        let mut reply = HostReply::default();
        reply.text = text.to_owned();
        reply.input_tokens = 1;
        reply.output_tokens = 1;
        Ok(reply)
    }
}

/// A host model that answers at once with one short text.
/// Input: none. Output: `x` for every request.
/// Used by the event benches, where only the run's events matter.
struct PlainModel;

impl HostModel for PlainModel {
    fn id(&self) -> String {
        "plain".to_owned()
    }

    fn capabilities(&self) -> HostModelCaps {
        BenchModel.capabilities()
    }

    fn generate(
        &self,
        _request: HostRequest,
        sink: Arc<TokenSink>,
    ) -> Result<HostReply, HostError> {
        sink.delta("x".to_owned());
        let mut reply = HostReply::default();
        reply.text = "x".to_owned();
        Ok(reply)
    }
}

/// A host tool that does nothing and counts its calls.
/// Input: none. Output: `TOOL_OUTPUT`; `calls` is the number of calls.
/// The count proves every iteration crossed the tool boundary.
struct NoopTool {
    calls: AtomicUsize,
}

impl HostTool for NoopTool {
    fn name(&self) -> String {
        "noop".to_owned()
    }

    fn description(&self) -> String {
        "Does nothing.".to_owned()
    }

    fn schema_json(&self) -> String {
        r#"{"type":"object","properties":{}}"#.to_owned()
    }

    fn call(&self, _args_json: String) -> Result<String, HostError> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        Ok(TOOL_OUTPUT.to_owned())
    }
}

/// Counts events and signals the end of each listened run.
/// Input: none. Output: `wait` returns once a terminal event arrived.
/// Reset per run by `wait`.
struct Ender {
    done: Mutex<bool>,
    ready: Condvar,
}

impl EventListener for Ender {
    fn on_event(&self, event: FfiEvent) {
        let terminal = matches!(
            event.kind,
            FfiEventKind::RunFinished { .. }
                | FfiEventKind::RunFailed { .. }
                | FfiEventKind::RunPaused { .. }
        );
        if terminal {
            *self.done.lock().expect("ender lock") = true;
            self.ready.notify_all();
        }
    }
}

impl Ender {
    /// Blocks until the current run's terminal event, then resets.
    /// Input: none. Output: none.
    /// Called once per listened run.
    fn wait(&self) {
        let mut done = self.done.lock().expect("ender lock");
        while !*done {
            done = self.ready.wait(done).expect("ender wait");
        }
        *done = false;
    }
}

/// Untimed runs before each bench's samples: a tenth of the count, at
/// least one. Input: the sample count. Output: the warm-up count.
/// Every runner uses the same rule, so the passes compare.
fn warmup_of(count: usize) -> usize {
    assert!(count > 0);
    let warmup = (count / 10).max(1);
    assert!(warmup <= count);
    warmup
}

/// Runs `body` untimed `warmup_of(iterations)` times, then times it for
/// `iterations` runs and prints one result line. Input: the bench name,
/// count, label and body. Output: none; panics on a wrong sample count.
fn bench(name: &str, iterations: usize, label: &str, mut body: impl FnMut()) {
    for _ in 0..warmup_of(iterations) {
        body();
    }
    let mut samples = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let start = Instant::now();
        body();
        samples.push(start.elapsed().as_nanos());
    }
    assert_eq!(samples.len(), iterations);
    samples.sort_unstable();
    let p50 = samples[iterations / 2];
    let p99 = samples[(iterations * 99 / 100).min(iterations - 1)];
    let first = first_pass();
    println!(
        "overhead lang=rust bench={name} n={iterations} p50_ns={p50} p99_ns={p99} \
         label={label} first={first}"
    );
}

/// Builds an agent handle on `model` with the given host tools.
/// Input: the model and tools. Output: the built handle; panics when the
/// neutral layer refuses the build.
fn handle(model: Arc<dyn HostModel>, tools: Vec<Arc<dyn HostTool>>) -> Arc<AgentHandle> {
    let mut builder = AgentBuilder::new("bench".to_owned())
        .instructions("Benchmark.".to_owned())
        .host_model(model);
    for tool in tools {
        builder = builder.tool(tool);
    }
    block_on(builder.build()).expect("bench agent builds")
}

/// Reads `TROUPEKIT_BENCH_N`, the iteration count of each bench.
/// Input: the environment. Output: the count, default 10 000, bounded.
/// The same variable drives the Swift, Kotlin and Python benches.
fn iterations() -> usize {
    let count = std::env::var("TROUPEKIT_BENCH_N")
        .ok()
        .map(|text| {
            text.parse::<usize>()
                .expect("TROUPEKIT_BENCH_N is a number")
        })
        .unwrap_or(ITERATIONS_DEFAULT);
    assert!(
        count > 0 && count <= ITERATIONS_MAX,
        "TROUPEKIT_BENCH_N out of bounds"
    );
    count
}

/// Reads `TROUPEKIT_BENCH_FIRST`, the label of the pass CI ran first.
/// Input: the environment. Output: `quiet` or `loaded`; panics on any
/// other value, so a result line always names its pass order.
fn first_pass() -> String {
    let first = std::env::var("TROUPEKIT_BENCH_FIRST").unwrap_or_else(|_| "quiet".to_owned());
    assert!(
        first == "quiet" || first == "loaded",
        "TROUPEKIT_BENCH_FIRST is {first}"
    );
    first
}

fn main() {
    let count = iterations();
    let runs = count + warmup_of(count);
    let label = std::env::var("TROUPEKIT_BENCH_LABEL").unwrap_or_else(|_| "quiet".to_owned());
    troupekit_ffi_core::init(RuntimeSettings::default()).expect("runtime init");

    let mut versions = 0usize;
    bench("version", count, &label, || {
        assert!(!troupekit_ffi_core::version().is_empty());
        versions += 1;
    });
    assert_eq!(versions, runs);

    let tool = Arc::new(NoopTool {
        calls: AtomicUsize::new(0),
    });
    let agent = handle(Arc::new(BenchModel), vec![tool.clone()]);
    bench("tool_round_trip", count, &label, || {
        let answer = block_on(agent.run("go".to_owned())).expect("tool run");
        assert_eq!(answer, "done");
    });
    assert_eq!(tool.calls.load(Ordering::Relaxed), runs);

    let plain = handle(Arc::new(PlainModel), Vec::new());
    let mut pulled = 0usize;
    bench("event_pull", count, &label, || {
        let stream = block_on(plain.stream("go".to_owned())).expect("stream starts");
        while block_on(stream.next()).expect("next event").is_some() {}
        pulled += 1;
    });
    assert_eq!(pulled, runs);

    let ender = Arc::new(Ender {
        done: Mutex::new(false),
        ready: Condvar::new(),
    });
    let mut listened = 0usize;
    bench("event_listener", count, &label, || {
        let stream = block_on(plain.stream("go".to_owned())).expect("stream starts");
        stream.set_listener(ender.clone()).expect("listener set");
        block_on(stream.result()).expect("listened run");
        ender.wait();
        listened += 1;
    });
    assert_eq!(listened, runs);
}
