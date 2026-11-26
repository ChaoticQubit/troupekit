use crate::category::{Trace, TraceSet};
use crate::error::TraceError;
use crate::record;
use crate::sink::TraceSink;
use crate::switch;
use crate::value::{cut_to, Value};
use std::cell::{Cell, RefCell};
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::{SystemTime, UNIX_EPOCH};

const SECRETS_MAX: usize = 1024;
const NAME_BYTES_MAX: usize = 128;

thread_local! {
    static CURRENT: RefCell<Option<Ctx>> = const { RefCell::new(None) };
    static EMITTING: Cell<bool> = const { Cell::new(false) };
}

pub(crate) struct Shared {
    pub(crate) trace_id: u64,
    pub(crate) name: String,
    pub(crate) sink: Arc<dyn TraceSink>,
    next_span: AtomicU64,
    pub(crate) dropped: AtomicU64,
    pub(crate) truncated: AtomicU64,
}

impl Shared {
    fn alloc_span(&self) -> u64 {
        let id = self.next_span.fetch_add(1, Ordering::Relaxed);
        assert!(id > 0, "span ids start at 1");
        assert!(id < u64::MAX, "span ids exhausted");
        id
    }
}

impl Drop for Shared {
    fn drop(&mut self) {
        let line = record::footer(self);
        write_line(&*self.sink, &line);
        self.sink.flush();
    }
}

pub(crate) struct Inner {
    pub(crate) shared: Arc<Shared>,
    pub(crate) categories: TraceSet,
    pub(crate) content: bool,
    content_set: bool,
    pub(crate) scrub: Vec<String>,
}

impl Inner {
    fn new(
        shared: Arc<Shared>,
        categories: TraceSet,
        content: Option<bool>,
        scrub: Vec<String>,
    ) -> Inner {
        switch::acquire(categories);
        let content_set = content.is_some();
        Inner {
            shared,
            categories,
            content: content.unwrap_or(false),
            content_set,
            scrub,
        }
    }
}

impl Drop for Inner {
    fn drop(&mut self) {
        switch::release(self.categories);
    }
}

/// A live trace: categories, content flag, scrubbed secrets and one sink.
/// Built with `Tracer::builder()`. While any clone is alive its categories
/// are switched on, but points record only inside `enter` or `in_scope`.
#[derive(Clone)]
pub struct Tracer {
    inner: Arc<Inner>,
}

/// Settings for a new `Tracer`. The trace id is chosen when the builder is
/// made, so a caller can name a file after it before picking the sink.
/// Defaults: every category, content withheld, name `trace`.
pub struct TracerBuilder {
    trace_id: u64,
    categories: TraceSet,
    content: Option<bool>,
    sink: Option<Arc<dyn TraceSink>>,
    scrub: Vec<String>,
    name: String,
}

impl Tracer {
    /// Starts a tracer with the default settings and a fresh trace id.
    /// Input: none. Output: a builder; `.sink(..)` is required.
    pub fn builder() -> TracerBuilder {
        TracerBuilder {
            trace_id: new_trace_id(),
            categories: TraceSet::all(),
            content: None,
            sink: None,
            scrub: Vec::new(),
            name: "trace".to_owned(),
        }
    }

    /// Returns the 64-bit trace id; records show it as 16 hex characters.
    /// Input: none. Output: the id.
    pub fn trace_id(&self) -> u64 {
        self.inner.shared.trace_id
    }

    /// Returns the categories this tracer switches on.
    /// Input: none. Output: the set.
    pub fn categories(&self) -> TraceSet {
        self.inner.categories
    }

    /// Runs `f` with this tracer as the current scope on this thread.
    /// Inside an active scope of another tracer it nests (see `Ctx::nest`).
    /// Input: the closure. Output: its result.
    pub fn enter<R>(&self, f: impl FnOnce() -> R) -> R {
        self.scope_ctx().enter(f)
    }

    /// Wraps `fut` so this tracer is the current scope on every poll,
    /// on whatever thread polls it. The scope is fixed when this is called.
    /// Input: the future. Output: the wrapped future.
    pub fn in_scope<F: Future>(&self, fut: F) -> Instrumented<F> {
        self.scope_ctx().in_scope(fut)
    }

    fn scope_ctx(&self) -> Ctx {
        match current() {
            Some(outer) if Arc::ptr_eq(&outer.inner.shared, &self.inner.shared) => outer,
            Some(outer) => outer.merge(&self.inner),
            None => Ctx {
                inner: self.inner.clone(),
                span: 0,
            },
        }
    }
}

impl TracerBuilder {
    /// Sets the categories. Input: the set. Output: the builder.
    pub fn categories(mut self, categories: TraceSet) -> TracerBuilder {
        self.categories = categories;
        self
    }

    /// Keeps (`true`) or withholds content values. Input: the flag.
    /// Output: the builder. Unset means withheld, or the outer scope's value.
    pub fn content(mut self, content: bool) -> TracerBuilder {
        self.content = Some(content);
        self
    }

    /// Sets the sink that receives every record line. Required.
    /// Input: the sink. Output: the builder.
    pub fn sink(mut self, sink: Arc<dyn TraceSink>) -> TracerBuilder {
        self.sink = Some(sink);
        self
    }

    /// Adds a secret value, such as an API key, replaced by `[redacted]`
    /// in every record. An empty value is ignored. At most 1024 secrets.
    /// Input: the secret. Output: the builder.
    pub fn scrub(mut self, secret: &str) -> TracerBuilder {
        if !secret.is_empty() {
            self.scrub.push(secret.to_owned());
        }
        self
    }

    /// Sets the trace name written in the header, cut to 128 bytes.
    /// Input: the name. Output: the builder.
    pub fn name(mut self, name: &str) -> TracerBuilder {
        self.name = cut_to(name, NAME_BYTES_MAX).to_owned();
        self
    }

    /// Returns the trace id the tracer will have. Input: none. Output: the id.
    pub fn trace_id(&self) -> u64 {
        self.trace_id
    }

    /// Builds the tracer and writes the header record to its sink.
    /// Input: none. Output: the tracer, or `MissingSink` or `TooManySecrets`.
    pub fn build(self) -> Result<Tracer, TraceError> {
        let sink = self.sink.ok_or(TraceError::MissingSink)?;
        if self.scrub.len() > SECRETS_MAX {
            return Err(TraceError::TooManySecrets(self.scrub.len()));
        }
        assert_ne!(self.trace_id, 0);
        let shared = Arc::new(Shared {
            trace_id: self.trace_id,
            name: self.name,
            sink,
            next_span: AtomicU64::new(1),
            dropped: AtomicU64::new(0),
            truncated: AtomicU64::new(0),
        });
        let inner = Arc::new(Inner::new(
            shared,
            self.categories,
            self.content,
            self.scrub,
        ));
        write_line(&*inner.shared.sink, &record::header(&inner));
        Ok(Tracer { inner })
    }
}

/// A captured trace scope: a tracer plus the current span id.
/// Get one with `current()` and re-enter it in a spawned task or a
/// blocking closure with `enter` or `in_scope`, so parents stay right.
#[derive(Clone)]
pub struct Ctx {
    pub(crate) inner: Arc<Inner>,
    pub(crate) span: u64,
}

impl Ctx {
    /// Runs `f` with this scope current on this thread, restoring the
    /// previous scope afterwards, also when `f` panics.
    /// Input: the closure. Output: its result.
    pub fn enter<R>(&self, f: impl FnOnce() -> R) -> R {
        let _guard = ScopeGuard::set(Some(self.clone()));
        f()
    }

    /// Wraps `fut` so this scope is current on every poll.
    /// Input: the future. Output: the wrapped future.
    pub fn in_scope<F: Future>(&self, fut: F) -> Instrumented<F> {
        Instrumented {
            fut: Box::pin(fut),
            ctx: Some(self.clone()),
        }
    }

    /// Returns a scope in the same trace and sink with `categories` added.
    /// `content` replaces this scope's content flag when given.
    /// Input: extra categories and an optional flag. Output: the nested scope.
    pub fn nest(&self, categories: TraceSet, content: Option<bool>) -> Ctx {
        let inner = Inner::new(
            self.inner.shared.clone(),
            self.inner.categories.union(categories),
            Some(content.unwrap_or(self.inner.content)),
            self.inner.scrub.clone(),
        );
        Ctx {
            inner: Arc::new(inner),
            span: self.span,
        }
    }

    fn merge(&self, other: &Inner) -> Ctx {
        let content = if other.content_set {
            other.content
        } else {
            self.inner.content
        };
        let mut scrub = self.inner.scrub.clone();
        scrub.extend(other.scrub.iter().cloned());
        let inner = Inner::new(
            self.inner.shared.clone(),
            self.inner.categories.union(other.categories),
            Some(content),
            scrub,
        );
        Ctx {
            inner: Arc::new(inner),
            span: self.span,
        }
    }

    /// Returns the trace id. Input: none. Output: the id.
    pub fn trace_id(&self) -> u64 {
        self.inner.shared.trace_id
    }

    /// Returns the current span id, 0 at the root. Input: none. Output: the id.
    pub fn span_id(&self) -> u64 {
        self.span
    }

    /// Returns the categories of this scope. Input: none. Output: the set.
    pub fn categories(&self) -> TraceSet {
        self.inner.categories
    }

    /// Returns whether this scope keeps content. Input: none. Output: the flag.
    pub fn content(&self) -> bool {
        self.inner.content
    }

    pub(crate) fn child(&self, span: u64) -> Ctx {
        Ctx {
            inner: self.inner.clone(),
            span,
        }
    }

    pub(crate) fn alloc_span(&self) -> u64 {
        let id = self.inner.shared.alloc_span();
        check_parent(self.span, id);
        id
    }

    pub(crate) fn emit(&self, line: &str) {
        write_line(&*self.inner.shared.sink, line);
    }

    pub(crate) fn flush(&self) {
        self.inner.shared.sink.flush();
    }
}

/// Returns the trace scope current on this thread, if any.
/// Input: none. Output: a clone of the scope, to pass into a spawned task
/// or a blocking closure and re-enter there.
pub fn current() -> Option<Ctx> {
    CURRENT
        .try_with(|cell| cell.try_borrow().ok().and_then(|ctx| ctx.clone()))
        .ok()
        .flatten()
}

pub(crate) fn scope_contains(t: Trace) -> bool {
    if EMITTING.try_with(Cell::get).unwrap_or(true) {
        return false;
    }
    CURRENT
        .try_with(|cell| {
            cell.try_borrow()
                .ok()
                .is_some_and(|ctx| ctx.as_ref().is_some_and(|c| c.inner.categories.contains(t)))
        })
        .unwrap_or(false)
}

/// Writes a point event of category `cat` in the current scope.
/// Called by `trace!` after `enabled`; not meant to be called directly.
#[doc(hidden)]
pub fn point(cat: Trace, name: &'static str, fields: Vec<(&'static str, Value)>) {
    let Some(ctx) = current() else { return };
    let line = record::event(&ctx, cat, name, fields);
    ctx.emit(&line);
}

fn write_line(sink: &dyn TraceSink, line: &str) {
    if EMITTING.try_with(|flag| flag.replace(true)).unwrap_or(true) {
        return;
    }
    let _guard = EmitGuard;
    sink.write(line);
}

struct EmitGuard;

impl Drop for EmitGuard {
    fn drop(&mut self) {
        let _ = EMITTING.try_with(|flag| flag.set(false));
    }
}

fn check_parent(parent: u64, id: u64) {
    assert!(id > 0, "span id 0 is reserved");
    assert!(
        parent < id,
        "parent span {parent} not opened before span {id}"
    );
}

struct ScopeGuard {
    prev: Option<Ctx>,
}

impl ScopeGuard {
    fn set(ctx: Option<Ctx>) -> ScopeGuard {
        let prev = CURRENT.try_with(|cell| cell.replace(ctx)).ok().flatten();
        ScopeGuard { prev }
    }
}

impl Drop for ScopeGuard {
    fn drop(&mut self) {
        let prev = self.prev.take();
        let _ = CURRENT.try_with(|cell| cell.replace(prev));
    }
}

/// A future that installs a trace scope on every poll and restores the
/// previous one after, so the scope follows the task across threads.
/// Made by `Tracer::in_scope`, `Ctx::in_scope` and `Span::instrument`.
pub struct Instrumented<F> {
    fut: Pin<Box<F>>,
    ctx: Option<Ctx>,
}

impl<F> Instrumented<F> {
    pub(crate) fn passthrough(fut: F) -> Instrumented<F> {
        Instrumented {
            fut: Box::pin(fut),
            ctx: None,
        }
    }
}

impl<F: Future> Future for Instrumented<F> {
    type Output = F::Output;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<F::Output> {
        let this = self.get_mut();
        let _guard = this.ctx.clone().map(|ctx| ScopeGuard::set(Some(ctx)));
        this.fut.as_mut().poll(cx)
    }
}

fn new_trace_id() -> u64 {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    let count = COUNTER.fetch_add(1, Ordering::Relaxed);
    let id = splitmix64((nanos as u64) ^ count.rotate_left(32) ^ u64::from(std::process::id()));
    if id == 0 {
        1
    } else {
        id
    }
}

fn splitmix64(seed: u64) -> u64 {
    let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[should_panic(expected = "not opened before")]
    fn parent_after_child_panics() {
        check_parent(5, 3);
    }

    #[test]
    fn trace_ids_differ() {
        assert_ne!(new_trace_id(), new_trace_id());
    }
}
