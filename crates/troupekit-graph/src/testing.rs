use async_trait::async_trait;
use futures_util::future::BoxFuture;
use serde_json::{json, Value};
use std::future::Future;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use troupekit_core::error::ProviderError;
use troupekit_core::event::Usage;
use troupekit_core::message::{Message, Role};
use troupekit_core::tool::ToolSpec;
use troupekit_providers::{CompletionRequest, CompletionResponse, Delta, Model, StopReason};
use troupekit_tools::{Tool, ToolContext, ToolError, ToolOutput};

/// Builds an assistant reply holding `text`, charged `input_tokens` input
/// and 5 output tokens at the model's price, for `FnModel` closures.
/// Returns the reply.
pub fn text_reply(text: &str, input_tokens: u64) -> Result<CompletionResponse, ProviderError> {
    Ok(CompletionResponse {
        message: Message::text(Role::Assistant, text),
        usage: Usage {
            input_tokens,
            output_tokens: 5,
            cost_usd_micros: 0,
        },
        stop: StopReason::EndTurn,
    })
}

type CallFn =
    dyn Fn(Value, ToolContext) -> BoxFuture<'static, Result<ToolOutput, ToolError>> + Send + Sync;

/// A `Tool` from an async closure over the arguments and context, for
/// tests. Its schema accepts any object, so a tool node's reads always
/// pass the registry check.
pub struct FnTool {
    spec: ToolSpec,
    call: Box<CallFn>,
}

impl FnTool {
    /// Builds tool `name` that answers every call with `f(args, ctx)`.
    /// Returns it ready for `ToolRegistry::register`.
    /// Panics on an invalid tool name, as tests pass literals.
    pub fn new<F, Fut>(name: &str, f: F) -> Arc<dyn Tool>
    where
        F: Fn(Value, ToolContext) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<ToolOutput, ToolError>> + Send + 'static,
    {
        assert!(troupekit_tools::valid_tool_name(name), "test tool name");
        Arc::new(FnTool {
            spec: ToolSpec {
                name: name.to_string(),
                description: format!("test tool {name}"),
                schema: json!({"type": "object"}),
                hosted: None,
            },
            call: Box::new(move |args, ctx| Box::pin(f(args, ctx))),
        })
    }
}

#[async_trait]
impl Tool for FnTool {
    fn spec(&self) -> &ToolSpec {
        &self.spec
    }

    async fn call(&self, args: Value, ctx: &ToolContext) -> Result<ToolOutput, ToolError> {
        (self.call)(args, ctx.clone()).await
    }
}

type ReplyFn =
    dyn Fn(&CompletionRequest) -> Result<CompletionResponse, ProviderError> + Send + Sync;

/// A `Model` whose reply is a pure function of the request, for tests: a
/// replayed call gets the same reply however often the model was called.
/// `stream` sends the reply text as one delta; `calls` counts every call.
pub struct FnModel {
    id: String,
    reply: Box<ReplyFn>,
    calls: AtomicUsize,
}

impl FnModel {
    /// Builds a model called `id` answering every request with `reply`.
    /// The closure must not depend on call order.
    /// Returns the model.
    pub fn new(
        id: &str,
        reply: impl Fn(&CompletionRequest) -> Result<CompletionResponse, ProviderError>
            + Send
            + Sync
            + 'static,
    ) -> Self {
        Self {
            id: id.to_string(),
            reply: Box::new(reply),
            calls: AtomicUsize::new(0),
        }
    }

    /// Returns how many times the model was called, streamed or not.
    /// Crash and concurrency tests compare it across runners.
    /// Counts failed replies too.
    pub fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }
}

#[async_trait]
impl Model for FnModel {
    fn id(&self) -> &str {
        &self.id
    }

    async fn complete(&self, req: &CompletionRequest) -> Result<CompletionResponse, ProviderError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        (self.reply)(req)
    }

    async fn stream(
        &self,
        req: &CompletionRequest,
        on_delta: &mut (dyn for<'d> FnMut(&'d Delta) + Send),
    ) -> Result<CompletionResponse, ProviderError> {
        let reply = self.complete(req).await?;
        let text = reply.message.text_content();
        if !text.is_empty() {
            on_delta(&Delta::TextDelta { text });
        }
        Ok(reply)
    }
}
