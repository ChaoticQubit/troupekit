use crate::record;
use std::collections::VecDeque;
use std::fs::{File, OpenOptions};
use std::io::{self, BufWriter, Write};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard, PoisonError};

const MEMORY_CAPACITY_DEFAULT: usize = 100_000;
const MAX_BYTES_DEFAULT: u64 = 256 * 1024 * 1024;
const FLUSH_EVERY_DEFAULT: u32 = 64;
const PATH_NAME_BYTES_MAX: usize = 64;

/// A destination for finished trace record lines.
/// Input: one JSON line per call, without a trailing newline.
/// A sink never returns an error and never panics through the caller.
pub trait TraceSink: Send + Sync {
    /// Takes one record line. Errors are counted by the sink, not returned.
    fn write(&self, line: &str);

    /// Pushes buffered lines to their destination. The default does nothing.
    fn flush(&self) {}
}

fn locked<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Settings for `JsonlSink`. Defaults: refuse an existing file, stop past
/// 256 MiB with one `truncated` record, flush every 64 records.
/// Input: `with_*` setters. Output: passed to `JsonlSink::with_options`.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct JsonlOptions {
    append: bool,
    max_bytes: u64,
    flush_every: u32,
}

impl Default for JsonlOptions {
    fn default() -> JsonlOptions {
        JsonlOptions {
            append: false,
            max_bytes: MAX_BYTES_DEFAULT,
            flush_every: FLUSH_EVERY_DEFAULT,
        }
    }
}

impl JsonlOptions {
    /// Appends to an existing file instead of refusing it.
    /// Input: the flag. Output: the options.
    pub fn with_append(mut self, append: bool) -> JsonlOptions {
        self.append = append;
        self
    }

    /// Sets the byte limit of the file. Input: bytes. Output: the options.
    pub fn with_max_bytes(mut self, max_bytes: u64) -> JsonlOptions {
        self.max_bytes = max_bytes;
        self
    }

    /// Flushes after this many records, at least 1. Input: the count.
    /// Output: the options.
    pub fn with_flush_every(mut self, flush_every: u32) -> JsonlOptions {
        self.flush_every = flush_every.max(1);
        self
    }
}

struct JsonlState {
    out: BufWriter<File>,
    written: u64,
    pending: u32,
    stopped: bool,
}

/// Writes record lines to a private JSON lines file.
/// The file is created with mode 0600 and missing folders with 0700 on
/// Unix. I/O errors are counted, never returned to the traced run.
pub struct JsonlSink {
    state: Mutex<JsonlState>,
    options: JsonlOptions,
    errors: AtomicU64,
    path: PathBuf,
}

impl JsonlSink {
    /// Creates `path` with the default options; an existing file is refused.
    /// Input: the file path. Output: the sink or the I/O error.
    pub fn create(path: impl AsRef<Path>) -> io::Result<JsonlSink> {
        JsonlSink::with_options(path, JsonlOptions::default())
    }

    /// Creates or, with `with_append(true)`, opens `path` for appending.
    /// Input: the file path and options. Output: the sink or the I/O error.
    pub fn with_options(path: impl AsRef<Path>, options: JsonlOptions) -> io::Result<JsonlSink> {
        let path = path.as_ref().to_path_buf();
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            create_private_dirs(parent)?;
        }
        let file = open_private(&path, options.append)?;
        let written = file.metadata()?.len();
        let state = JsonlState {
            out: BufWriter::new(file),
            written,
            pending: 0,
            stopped: false,
        };
        Ok(JsonlSink {
            state: Mutex::new(state),
            options,
            errors: AtomicU64::new(0),
            path,
        })
    }

    /// Returns the file path. Input: none. Output: the path.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Returns the number of I/O errors so far. Input: none. Output: the count.
    pub fn errors(&self) -> u64 {
        self.errors.load(Ordering::Relaxed)
    }

    /// Returns the bytes in the file, buffered lines included.
    /// Input: none. Output: the count.
    pub fn bytes_written(&self) -> u64 {
        locked(&self.state).written
    }

    fn count(&self, result: io::Result<()>) {
        if result.is_err() {
            self.errors.fetch_add(1, Ordering::Relaxed);
        }
    }
}

impl TraceSink for JsonlSink {
    fn write(&self, line: &str) {
        let mut state = locked(&self.state);
        if state.stopped {
            return;
        }
        let size = line.len() as u64 + 1;
        if state.written.saturating_add(size) > self.options.max_bytes {
            state.stopped = true;
            let marker = record::truncated_from(line);
            let result = writeln!(state.out, "{marker}").and_then(|()| state.out.flush());
            self.count(result);
            return;
        }
        let result = writeln!(state.out, "{line}");
        self.count(result);
        state.written += size;
        state.pending += 1;
        if state.pending >= self.options.flush_every {
            state.pending = 0;
            let result = state.out.flush();
            self.count(result);
        }
        assert!(state.written <= self.options.max_bytes);
    }

    fn flush(&self) {
        let mut state = locked(&self.state);
        state.pending = 0;
        let result = state.out.flush();
        self.count(result);
    }
}

impl Drop for JsonlSink {
    fn drop(&mut self) {
        TraceSink::flush(self);
    }
}

#[cfg(unix)]
fn create_private_dirs(dir: &Path) -> io::Result<()> {
    use std::os::unix::fs::DirBuilderExt;
    std::fs::DirBuilder::new()
        .recursive(true)
        .mode(0o700)
        .create(dir)
}

#[cfg(not(unix))]
fn create_private_dirs(dir: &Path) -> io::Result<()> {
    std::fs::create_dir_all(dir)
}

fn open_private(path: &Path, append: bool) -> io::Result<File> {
    let mut options = OpenOptions::new();
    if append {
        options.append(true).create(true);
    } else {
        options.write(true).create_new(true);
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    options.open(path)
}

/// Keeps the newest record lines in memory, dropping the oldest past its
/// capacity (default 100 000 lines) and counting what it dropped.
/// Input: lines. Output: `lines()` copies and `dropped()`.
pub struct MemorySink {
    lines: Mutex<VecDeque<String>>,
    capacity: usize,
    dropped: AtomicU64,
}

impl Default for MemorySink {
    fn default() -> MemorySink {
        MemorySink::new(MEMORY_CAPACITY_DEFAULT)
    }
}

impl MemorySink {
    /// Returns an empty sink holding at most `capacity` lines, at least 1.
    /// Input: the capacity. Output: the sink.
    pub fn new(capacity: usize) -> MemorySink {
        let capacity = capacity.max(1);
        MemorySink {
            lines: Mutex::new(VecDeque::new()),
            capacity,
            dropped: AtomicU64::new(0),
        }
    }

    /// Returns a copy of the kept lines, oldest first. Input: none.
    pub fn lines(&self) -> Vec<String> {
        locked(&self.lines).iter().cloned().collect()
    }

    /// Returns how many lines were dropped. Input: none. Output: the count.
    pub fn dropped(&self) -> u64 {
        self.dropped.load(Ordering::Relaxed)
    }

    /// Returns the line capacity. Input: none. Output: the capacity.
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

impl TraceSink for MemorySink {
    fn write(&self, line: &str) {
        let mut lines = locked(&self.lines);
        if lines.len() == self.capacity {
            lines.pop_front();
            self.dropped.fetch_add(1, Ordering::Relaxed);
        }
        lines.push_back(line.to_owned());
        assert!(lines.len() <= self.capacity);
    }
}

type Callback = Box<dyn Fn(&str) + Send + Sync>;

/// Hands every record line to a callback. A panic in the callback is
/// caught, counted and disables the sink, so it never fails the run.
/// Input: lines. Output: calls to the callback.
pub struct CallbackSink {
    callback: Callback,
    disabled: AtomicBool,
    panics: AtomicU64,
}

impl CallbackSink {
    /// Returns a sink that calls `callback` with each line.
    /// Input: the callback. Output: the sink.
    pub fn new(callback: impl Fn(&str) + Send + Sync + 'static) -> CallbackSink {
        CallbackSink {
            callback: Box::new(callback),
            disabled: AtomicBool::new(false),
            panics: AtomicU64::new(0),
        }
    }

    /// Returns how many times the callback panicked. Input: none.
    pub fn panics(&self) -> u64 {
        self.panics.load(Ordering::Relaxed)
    }

    /// Returns whether a panic disabled the sink. Input: none.
    pub fn is_disabled(&self) -> bool {
        self.disabled.load(Ordering::Relaxed)
    }
}

impl TraceSink for CallbackSink {
    fn write(&self, line: &str) {
        if self.is_disabled() {
            return;
        }
        if catch_unwind(AssertUnwindSafe(|| (self.callback)(line))).is_err() {
            self.disabled.store(true, Ordering::Relaxed);
            self.panics.fetch_add(1, Ordering::Relaxed);
        }
    }
}

/// Returns the default trace file path: `$TROUPEKIT_TRACE_DIR` or
/// `./traces`, file `<name>-<trace id>.jsonl`. Bytes of `name` outside
/// `[a-z0-9_-]` become `_`; the name is cut to 64 bytes, empty is `trace`.
pub fn default_trace_path(name: &str, trace_id: u64) -> PathBuf {
    let dir = std::env::var_os("TROUPEKIT_TRACE_DIR")
        .map_or_else(|| PathBuf::from("traces"), PathBuf::from);
    dir.join(format!("{}-{trace_id:016x}.jsonl", file_stem(name)))
}

fn file_stem(name: &str) -> String {
    if name.is_empty() {
        return "trace".to_owned();
    }
    let stem: String = name
        .bytes()
        .take(PATH_NAME_BYTES_MAX)
        .map(|b| match b {
            b'a'..=b'z' | b'0'..=b'9' | b'_' | b'-' => char::from(b),
            _ => '_',
        })
        .collect();
    assert!(stem.len() <= PATH_NAME_BYTES_MAX);
    assert!(!stem.contains(['/', '\\', '.']));
    stem
}
