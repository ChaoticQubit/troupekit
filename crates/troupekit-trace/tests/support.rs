use serde_json::Value as Json;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::task::{Context, Poll, Wake, Waker};
use troupekit_trace::TraceSink;

static GLOBAL: Mutex<()> = Mutex::new(());

/// Serializes every test that builds a tracer, because the live mask is global.
/// Input: none. Output: a guard held for the whole test.
/// A poisoned lock from a failed test is recovered, not propagated.
pub fn lock() -> MutexGuard<'static, ()> {
    GLOBAL.lock().unwrap_or_else(PoisonError::into_inner)
}

/// A sink that keeps every line it receives, for assertions.
/// Input: lines from a tracer. Output: `lines()` and `records()` copies.
/// Unbounded on purpose: tests write a handful of lines.
#[derive(Default)]
pub struct Collect {
    lines: Mutex<Vec<String>>,
}

impl Collect {
    /// Returns a new empty collector behind an `Arc`.
    /// Input: none. Output: the shared collector.
    pub fn new() -> Arc<Collect> {
        Arc::new(Collect::default())
    }

    /// Returns a copy of every line written so far, in order.
    /// Input: none. Output: the lines.
    pub fn lines(&self) -> Vec<String> {
        self.lines
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    /// Parses every line as JSON and panics on a line that does not parse.
    /// Input: none. Output: the records, in order.
    pub fn records(&self) -> Vec<Json> {
        self.lines()
            .iter()
            .map(|l| serde_json::from_str(l).expect(l))
            .collect()
    }

    /// Returns the records whose `t` key equals `kind`.
    /// Input: the record kind. Output: the matching records.
    pub fn of(&self, kind: &str) -> Vec<Json> {
        self.records()
            .into_iter()
            .filter(|r| r["t"] == kind)
            .collect()
    }

    /// Returns the single `start` or `event` record named `name`.
    /// Input: the span or event name. Output: the record.
    /// Panics unless exactly one record has that name.
    pub fn named(&self, name: &str) -> Json {
        let found: Vec<Json> = self
            .records()
            .into_iter()
            .filter(|r| r["name"] == name)
            .collect();
        assert_eq!(found.len(), 1, "records named {name}: {found:?}");
        found.into_iter().next().unwrap()
    }
}

impl TraceSink for Collect {
    fn write(&self, line: &str) {
        self.lines
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(line.to_owned());
    }
}

struct Noop;

impl Wake for Noop {
    fn wake(self: Arc<Self>) {}
}

/// Polls `fut` once with a waker that does nothing.
/// Input: the pinned future. Output: its poll result.
/// Lets a test move one future between threads between polls.
pub fn poll_once<F: Future + ?Sized>(fut: Pin<&mut F>) -> Poll<F::Output> {
    let waker = Waker::from(Arc::new(Noop));
    fut.poll(&mut Context::from_waker(&waker))
}

/// A future that is pending on its first poll and ready on the second.
/// Input: none. Output: `()` on the second poll.
#[derive(Default)]
pub struct YieldOnce {
    polled: bool,
}

impl Future for YieldOnce {
    type Output = ();

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        if self.polled {
            return Poll::Ready(());
        }
        self.polled = true;
        cx.waker().wake_by_ref();
        Poll::Pending
    }
}

/// Returns a JSON number field of `record` as u64, panicking when absent.
/// Input: the record and key. Output: the number.
pub fn num(record: &Json, key: &str) -> u64 {
    record[key]
        .as_u64()
        .unwrap_or_else(|| panic!("{key} missing in {record}"))
}

/// Returns a fresh path under the system temp folder that does not exist yet.
/// Input: a label for the test. Output: the path; the test removes it.
/// Unique per process, label and call, so parallel tests never share one.
pub fn scratch_dir(label: &str) -> std::path::PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let n = NEXT.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir()
        .join("troupekit-trace-tests")
        .join(format!("{label}-{}-{n}", std::process::id()));
    if dir.exists() {
        std::fs::remove_dir_all(&dir).unwrap();
    }
    dir
}
