use serde::Serialize;
use serde_json::Value;
use std::fmt::Display;
use troupekit_tools::ToolOutput;

/// Converts a tool fn's return value into what the model sees.
/// Text, numbers, `bool`, `()`, JSON values, `Json<T>`, `Option<T>` and
/// `Result<T, E: Display>` convert; an `Err` becomes an error output so the
/// model can react to the failure instead of the run stopping.
#[diagnostic::on_unimplemented(
    message = "`{Self}` cannot be a #[tool] return type",
    note = "return String, a number, bool, (), serde_json::Value, Json<T>, Option<T> or Result<T, E: Display>"
)]
pub trait IntoToolOutput {
    /// Consumes the value and returns the model-facing output.
    /// Never panics; a value that cannot be rendered becomes an error output.
    /// The output content is the text the model reads.
    fn into_output(self) -> ToolOutput;
}

/// Wraps any `Serialize` value so a tool can return it as JSON text.
/// A serialize failure becomes an error output naming the failure.
/// Use it for structs: `Json(Point { x, y })`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Json<T>(pub T);

impl IntoToolOutput for ToolOutput {
    fn into_output(self) -> ToolOutput {
        self
    }
}

impl IntoToolOutput for String {
    fn into_output(self) -> ToolOutput {
        ToolOutput::text(self)
    }
}

impl IntoToolOutput for &str {
    fn into_output(self) -> ToolOutput {
        ToolOutput::text(self)
    }
}

impl IntoToolOutput for () {
    fn into_output(self) -> ToolOutput {
        ToolOutput::text("ok")
    }
}

impl IntoToolOutput for Value {
    fn into_output(self) -> ToolOutput {
        ToolOutput::text(self.to_string())
    }
}

impl<T: Serialize> IntoToolOutput for Json<T> {
    fn into_output(self) -> ToolOutput {
        match serde_json::to_string(&self.0) {
            Ok(text) => ToolOutput::text(text),
            Err(e) => ToolOutput::error(format!("the tool result does not serialize: {e}")),
        }
    }
}

impl<T: IntoToolOutput> IntoToolOutput for Option<T> {
    fn into_output(self) -> ToolOutput {
        match self {
            Some(value) => value.into_output(),
            None => ToolOutput::text("null"),
        }
    }
}

impl<T: IntoToolOutput, E: Display> IntoToolOutput for Result<T, E> {
    fn into_output(self) -> ToolOutput {
        match self {
            Ok(value) => value.into_output(),
            Err(e) => ToolOutput::error(e.to_string()),
        }
    }
}

macro_rules! display_output {
    ($($t:ty),* $(,)?) => {
        $(
            impl IntoToolOutput for $t {
                fn into_output(self) -> ToolOutput {
                    ToolOutput::text(self.to_string())
                }
            }
        )*
    };
}

display_output!(bool, i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize, f32, f64);
