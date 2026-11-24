use crate::tool::{Tool, ToolContext, ToolError, ToolOutput};
use async_trait::async_trait;
use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use serde_json::Value;
use std::future::Future;
use std::marker::PhantomData;
use troupekit_core::tool::ToolSpec;

/// A tool whose arguments are the serde struct `A` and whose body is the
/// async closure `F`. Build it with `typed_tool`; the argument schema comes
/// from `A`'s `schemars::JsonSchema` derive, so it never drifts from the type.
pub struct TypedTool<A, F> {
    spec: ToolSpec,
    f: F,
    args: PhantomData<fn() -> A>,
}

/// Builds a tool called `name` from the argument type `A` and closure `f`.
/// `f` receives the parsed arguments and an owned `ToolContext`, so it may
/// hold the context across awaits. Returns the tool; wrap it in an `Arc`.
pub fn typed_tool<A, F, Fut>(name: &str, description: &str, f: F) -> TypedTool<A, F>
where
    A: DeserializeOwned + JsonSchema + Send + 'static,
    F: Fn(A, ToolContext) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<ToolOutput, ToolError>> + Send + 'static,
{
    let mut schema = serde_json::to_value(schemars::schema_for!(A))
        .expect("a derived schema serializes to JSON");
    if let Value::Object(map) = &mut schema {
        map.remove("$schema");
        map.remove("title");
    }
    TypedTool {
        spec: ToolSpec {
            name: name.into(),
            description: description.into(),
            schema,
            hosted: None,
        },
        f,
        args: PhantomData,
    }
}

#[async_trait]
impl<A, F, Fut> Tool for TypedTool<A, F>
where
    A: DeserializeOwned + JsonSchema + Send + 'static,
    F: Fn(A, ToolContext) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<ToolOutput, ToolError>> + Send + 'static,
{
    fn spec(&self) -> &ToolSpec {
        &self.spec
    }

    async fn call(&self, args: Value, ctx: &ToolContext) -> Result<ToolOutput, ToolError> {
        let parsed: A =
            serde_json::from_value(args).map_err(|e| ToolError::InvalidArgs(e.to_string()))?;
        (self.f)(parsed, ctx.clone()).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::ToolRegistry;
    use crate::tool::{Tool, ToolContext, ToolError, ToolOutput};
    use serde::Deserialize;
    use serde_json::json;
    use std::sync::Arc;
    use troupekit_core::ids::RunId;

    #[derive(Deserialize, schemars::JsonSchema)]
    struct Args {
        city: String,
        days: Option<u8>,
    }

    fn weather() -> Arc<dyn Tool> {
        Arc::new(typed_tool(
            "weather",
            "Forecast for a city",
            |a: Args, _ctx: ToolContext| async move {
                Ok(ToolOutput::text(format!(
                    "{}:{}",
                    a.city,
                    a.days.map_or("none".into(), |d| d.to_string())
                )))
            },
        ))
    }

    #[test]
    fn schema_comes_from_the_struct() {
        let tool = weather();
        let s = &tool.spec().schema;
        assert_eq!(tool.spec().name, "weather");
        assert_eq!(s["type"], "object");
        assert_eq!(s["properties"]["city"]["type"], "string");
        assert_eq!(s["required"], json!(["city"]));
        assert!(s.get("$schema").is_none() && s.get("title").is_none());
    }

    #[tokio::test]
    async fn typed_tool_runs_through_registry() {
        let mut r = ToolRegistry::new();
        r.register(weather()).unwrap();
        let ctx = ToolContext::default();
        assert_eq!(
            r.call("weather", json!({"city":"Oslo"}), &ctx)
                .await
                .unwrap()
                .content,
            "Oslo:none"
        );
        assert_eq!(
            r.call("weather", json!({"city":"Oslo","days":3}), &ctx)
                .await
                .unwrap()
                .content,
            "Oslo:3"
        );
        assert!(matches!(
            r.call("weather", json!({"city":3}), &ctx).await,
            Err(ToolError::InvalidArgs(_))
        ));
    }

    #[tokio::test]
    async fn direct_call_with_bad_args_is_invalid_args() {
        let err = weather()
            .call(json!({"days":1}), &ToolContext::default())
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::InvalidArgs(_)), "{err}");
    }

    #[tokio::test]
    async fn closure_uses_context_across_await() {
        let tool = typed_tool("who", "Run id", |_a: Args, ctx: ToolContext| async move {
            tokio::task::yield_now().await;
            Ok(ToolOutput::text(format!("{:?}", ctx.run_id.is_some())))
        });
        let ctx = ToolContext {
            run_id: Some(RunId::new()),
            idempotency_key: None,
        };
        assert_eq!(
            tool.call(json!({"city":"x"}), &ctx).await.unwrap().content,
            "true"
        );
    }
}
