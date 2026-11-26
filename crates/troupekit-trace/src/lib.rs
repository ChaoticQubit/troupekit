//! Trace spans for every Troupekit crate, with no dependencies.
//! A disabled trace point costs one atomic load and a branch.
//! Records go to sinks as JSON lines; the `view` module reads them back.

#![forbid(unsafe_code)]

mod category;
mod error;
mod json;
mod record;
mod sink;
mod span;
mod spec;
mod switch;
mod tracer;
mod value;

pub use category::{Trace, TraceSet};
pub use error::TraceError;
pub use sink::TraceSink;
pub use span::Span;
pub use spec::TraceSpec;
pub use switch::{enabled, live_categories};
pub use tracer::{current, point, Ctx, Instrumented, Tracer, TracerBuilder};
pub use value::{content, Value};

/// Writes one point event: `trace!(Trace::Laws, "law.check", held = true)`.
/// Field expressions run only when the category is on in the current scope.
/// Keys are identifiers; values are anything that converts into `Value`.
#[macro_export]
macro_rules! trace {
    ($cat:expr, $name:expr $(, $key:ident = $value:expr)* $(,)?) => {
        if $crate::enabled($cat) {
            $crate::point(
                $cat,
                $name,
                ::std::vec![$((::core::stringify!($key), $crate::Value::from($value))),*],
            );
        }
    };
}

/// Opens a span: `let span = trace_span!(Trace::Models, "model.call", model = id);`.
/// Field expressions run only when the category is on in the current scope;
/// otherwise the macro returns a disabled `Span`.
#[macro_export]
macro_rules! trace_span {
    ($cat:expr, $name:expr $(, $key:ident = $value:expr)* $(,)?) => {
        if $crate::enabled($cat) {
            $crate::Span::open(
                $cat,
                $name,
                ::std::vec![$((::core::stringify!($key), $crate::Value::from($value))),*],
            )
        } else {
            $crate::Span::none()
        }
    };
}
