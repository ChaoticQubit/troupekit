use crate::category::Trace;
use crate::json::{push_f64, push_i64, push_key, push_str, push_u64};
use crate::redact::{self, Clean, Policy, FIELDS_MAX};
use crate::span::Open;
use crate::tracer::{Ctx, Inner, Shared};
use crate::value::Value;
use std::sync::atomic::Ordering;
use std::time::{SystemTime, UNIX_EPOCH};

const VERSION: u64 = 1;

/// Returns wall-clock microseconds since the Unix epoch, 0 before it.
/// Input: none. Output: the timestamp.
pub(crate) fn now_us() -> u64 {
    let micros = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_micros());
    u64::try_from(micros).unwrap_or(u64::MAX)
}

fn open_record(out: &mut String, kind: &str, trace_id: u64) {
    out.push_str("{\"v\":");
    push_u64(out, VERSION);
    out.push_str(",\"t\":");
    push_str(out, kind);
    out.push_str(",\"trace\":\"");
    out.push_str(&format!("{trace_id:016x}"));
    out.push('"');
}

fn push_num(out: &mut String, key: &str, value: u64) {
    push_key(out, key, false);
    push_u64(out, value);
}

/// Returns the `header` record line of a new tracer.
/// Input: the tracer's inner state. Output: one JSON line.
pub(crate) fn header(inner: &Inner) -> String {
    let mut out = String::with_capacity(256);
    open_record(&mut out, "header", inner.shared.trace_id);
    push_key(&mut out, "name", false);
    push_str(&mut out, &scrub(inner, &inner.shared.name));
    push_num(&mut out, "started_us", now_us());
    push_key(&mut out, "categories", false);
    out.push('[');
    for (i, t) in inner.categories.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        push_str(&mut out, t.as_str());
    }
    out.push(']');
    push_key(&mut out, "content", false);
    out.push_str(if inner.content { "true" } else { "false" });
    push_key(&mut out, "troupekit", false);
    push_str(&mut out, env!("CARGO_PKG_VERSION"));
    out.push('}');
    out
}

/// Returns the `start` record line of span `span` under the scope's span.
/// Input: scope, new span id, category, name and fields. Output: one line.
pub(crate) fn start(
    scope: &Ctx,
    span: u64,
    cat: Trace,
    name: &str,
    fields: Vec<(&'static str, Value)>,
) -> String {
    assert!(span > scope.span_id());
    let mut out = String::with_capacity(160);
    open_record(&mut out, "start", scope.trace_id());
    push_num(&mut out, "span", span);
    point_tail(
        &mut out,
        scope,
        cat,
        name,
        fields.into_iter().map(|(k, v)| (k.to_owned(), v)),
    );
    out
}

/// Returns an `event` record line under the scope's current span.
/// Input: scope, category, name and fields. Output: one line.
pub(crate) fn event(
    scope: &Ctx,
    cat: Trace,
    name: &str,
    fields: Vec<(&'static str, Value)>,
) -> String {
    let mut out = String::with_capacity(160);
    open_record(&mut out, "event", scope.trace_id());
    push_num(&mut out, "span", 0);
    point_tail(
        &mut out,
        scope,
        cat,
        name,
        fields.into_iter().map(|(k, v)| (k.to_owned(), v)),
    );
    out
}

fn point_tail(
    out: &mut String,
    scope: &Ctx,
    cat: Trace,
    name: &str,
    fields: impl Iterator<Item = (String, Value)>,
) {
    push_num(out, "parent", scope.span_id());
    push_key(out, "cat", false);
    push_str(out, cat.as_str());
    push_key(out, "name", false);
    push_str(out, &scrub(&scope.inner, name));
    push_num(out, "ts_us", now_us());
    push_fields(out, &scope.inner, fields);
    out.push('}');
}

/// Returns the `end` record line of an open span.
/// Input: the span state. Output: one line.
pub(crate) fn end(open: &Open) -> String {
    let ctx = &open.ctx;
    let mut out = String::with_capacity(128);
    open_record(&mut out, "end", ctx.trace_id());
    push_num(&mut out, "span", ctx.span_id());
    push_num(&mut out, "ts_us", now_us());
    let dur_us = u64::try_from(open.started.elapsed().as_micros()).unwrap_or(u64::MAX);
    push_num(&mut out, "dur_us", dur_us);
    push_key(&mut out, "ok", false);
    out.push_str(if open.err.is_none() { "true" } else { "false" });
    if let Some(err) = &open.err {
        push_key(&mut out, "err", false);
        push_str(&mut out, &scrub(&ctx.inner, err));
    }
    push_fields(&mut out, &ctx.inner, open.fields.iter().cloned());
    out.push('}');
    out
}

/// Returns the `footer` record line written when a trace ends.
/// `dropped` counts fields dropped past the per-record cap and `truncated`
/// counts string values cut at the size cap. Input: the trace. Output: a line.
pub(crate) fn footer(shared: &Shared) -> String {
    let mut out = String::with_capacity(96);
    open_record(&mut out, "footer", shared.trace_id);
    push_num(&mut out, "dropped", shared.dropped.load(Ordering::Relaxed));
    push_num(
        &mut out,
        "truncated",
        shared.truncated.load(Ordering::Relaxed),
    );
    out.push('}');
    out
}

fn push_fields(out: &mut String, inner: &Inner, fields: impl Iterator<Item = (String, Value)>) {
    let policy = Policy {
        secrets: &inner.scrub,
        content: inner.content,
    };
    let mut written = 0usize;
    let mut dropped = 0u64;
    for (key, value) in fields {
        if written == FIELDS_MAX {
            dropped += 1;
            continue;
        }
        out.push_str(if written == 0 { ",\"fields\":{" } else { "," });
        push_str(out, &redact::name(&key, policy.secrets));
        out.push(':');
        let (clean, was_cut) = redact::value(&key, value, &policy);
        push_clean(out, clean);
        if was_cut {
            inner.shared.truncated.fetch_add(1, Ordering::Relaxed);
        }
        written += 1;
    }
    if written > 0 {
        out.push('}');
    }
    assert!(written <= FIELDS_MAX);
    inner.shared.dropped.fetch_add(dropped, Ordering::Relaxed);
}

fn push_clean(out: &mut String, clean: Clean) {
    match clean {
        Clean::Str(text) => push_str(out, &text),
        Clean::Int(n) => push_i64(out, n),
        Clean::UInt(n) => push_u64(out, n),
        Clean::Float(n) => push_f64(out, n),
        Clean::Bool(b) => out.push_str(if b { "true" } else { "false" }),
        Clean::Withheld(bytes) => {
            out.push_str("{\"bytes\":");
            push_u64(out, bytes);
            out.push('}');
        }
    }
}

fn scrub(inner: &Inner, text: &str) -> String {
    let (clean, was_cut) = redact::cut(redact::scrub(text, &inner.scrub));
    if was_cut {
        inner.shared.truncated.fetch_add(1, Ordering::Relaxed);
    }
    clean
}
