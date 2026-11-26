use crate::category::Trace;
use crate::record;
use crate::tracer::{current, Ctx, Instrumented};
use crate::value::Value;
use std::future::Future;
use std::time::Instant;

/// A span guard: writes a `start` record when opened and an `end` record
/// when dropped. A span opened outside any scope, or for a category that
/// is off, is disabled and every method on it does nothing.
#[must_use = "a span ends when it is dropped"]
pub struct Span {
    open: Option<Open>,
}

pub(crate) struct Open {
    pub(crate) ctx: Ctx,
    pub(crate) parent: u64,
    pub(crate) started: Instant,
    pub(crate) fields: Vec<(String, Value)>,
    pub(crate) err: Option<String>,
}

impl Span {
    /// Opens a span of category `cat` under the current span.
    /// Called by `trace_span!` after `enabled`; not meant to be called directly.
    #[doc(hidden)]
    pub fn open(cat: Trace, name: &'static str, fields: Vec<(&'static str, Value)>) -> Span {
        let Some(scope) = current() else {
            return Span::none();
        };
        let id = scope.alloc_span();
        let line = record::start(&scope, id, cat, name, fields);
        scope.emit(&line);
        let ctx = scope.child(id);
        assert_eq!(ctx.span_id(), id);
        let open = Open {
            ctx,
            parent: scope.span_id(),
            started: Instant::now(),
            fields: Vec::new(),
            err: None,
        };
        Span { open: Some(open) }
    }

    /// Returns a disabled span. Input: none. Output: a span that records nothing.
    pub fn none() -> Span {
        Span { open: None }
    }

    /// Returns whether this span records. Input: none. Output: the flag.
    pub fn is_enabled(&self) -> bool {
        self.open.is_some()
    }

    /// Returns the span id, or 0 when disabled. Input: none. Output: the id.
    pub fn id(&self) -> u64 {
        self.open.as_ref().map_or(0, |open| open.ctx.span_id())
    }

    /// Adds a field written in the `end` record.
    /// Input: the key and value. Output: none.
    pub fn record(&mut self, key: &str, value: impl Into<Value>) {
        if let Some(open) = self.open.as_mut() {
            open.fields.push((key.to_owned(), value.into()));
        }
    }

    /// Marks the span failed with `msg`; the `end` record has `ok: false`.
    /// Input: the error text. Output: none.
    pub fn fail(&mut self, msg: impl Into<String>) {
        if let Some(open) = self.open.as_mut() {
            open.err = Some(msg.into());
        }
    }

    /// Runs `f` with this span as the current parent on this thread.
    /// Input: the closure. Output: its result.
    pub fn enter<R>(&self, f: impl FnOnce() -> R) -> R {
        match &self.open {
            Some(open) => open.ctx.enter(f),
            None => f(),
        }
    }

    /// Wraps `fut` so this span is the current parent on every poll.
    /// A disabled span leaves the caller's scope as it is.
    /// Input: the future. Output: the wrapped future.
    pub fn instrument<F: Future>(&self, fut: F) -> Instrumented<F> {
        match &self.open {
            Some(open) => open.ctx.in_scope(fut),
            None => Instrumented::passthrough(fut),
        }
    }
}

impl Drop for Span {
    fn drop(&mut self) {
        let Some(open) = self.open.take() else { return };
        let line = record::end(&open);
        open.ctx.emit(&line);
        if open.parent == 0 {
            open.ctx.flush();
        }
    }
}
