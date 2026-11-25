use crate::output::IntoToolOutput;
use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use serde_json::Value;
use std::future::Future;
use std::sync::Arc;
use troupekit_tools::{typed_tool, Tool, ToolContext, ToolSpec};

/// Anything an agent or graph accepts as a tool: every `Tool + 'static`.
/// A `#[tool]` fn and the `tool(..)` function form both qualify; an
/// `Arc<dyn Tool>` goes through `.tool_arc(..)` instead.
#[diagnostic::on_unimplemented(
    message = "`{Self}` is not a tool",
    note = "mark its fn with #[tool], or pass an Arc<dyn Tool> with .tool_arc(..)"
)]
pub trait AsTool: Tool + 'static {
    /// Moves the tool behind an `Arc<dyn Tool>` for a registry.
    /// Returns the shared tool; its spec is unchanged.
    /// Called once per registration.
    fn into_arc(self) -> Arc<dyn Tool>
    where
        Self: Sized;
}

impl<T: Tool + 'static> AsTool for T {
    fn into_arc(self) -> Arc<dyn Tool>
    where
        Self: Sized,
    {
        Arc::new(self)
    }
}

/// Builds a tool from argument type `A` and async fn `f`, without a macro.
/// `f` takes the parsed arguments and returns any `IntoToolOutput` value;
/// `name` and `description` are what the model sees. The schema is the
/// same one `typed_tool` derives from `A`.
pub fn tool<A, F, Fut>(name: &str, description: &str, f: F) -> impl Tool + 'static
where
    A: DeserializeOwned + JsonSchema + Send + 'static,
    F: Fn(A) -> Fut + Send + Sync + 'static,
    Fut: Future + Send + 'static,
    Fut::Output: IntoToolOutput,
{
    typed_tool(name, description, move |args: A, _ctx: ToolContext| {
        let pending = f(args);
        async move { Ok(pending.await.into_output()) }
    })
}

/// Returns the `ToolSpec` for argument type `A` named `name`.
/// The schema is `schema_for!(A)` without `$schema` and `title`, exactly
/// what `typed_tool` produces, so both paths give the model one schema.
/// Used by the `#[tool]` expansion.
pub fn spec<A: JsonSchema>(name: &str, description: &str) -> ToolSpec {
    let mut schema = schemars::schema_for!(A).to_value();
    assert!(schema.is_object(), "a derived schema is a JSON object");
    if let Value::Object(map) = &mut schema {
        map.remove("$schema");
        map.remove("title");
    }
    ToolSpec {
        name: name.into(),
        description: description.into(),
        schema,
        hosted: None,
    }
}
