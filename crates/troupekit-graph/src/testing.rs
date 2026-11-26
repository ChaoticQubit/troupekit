use async_trait::async_trait;
use futures_util::future::BoxFuture;
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::future::Future;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Poll, Waker};
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

#[derive(Default)]
struct Gates {
    open: BTreeSet<String>,
    waiting: Vec<(String, Waker)>,
    in_flight: BTreeMap<String, usize>,
    calls: BTreeMap<String, Vec<Option<String>>>,
    total: usize,
    peak: usize,
    finished: Vec<String>,
}

/// Gates for test tools, so a test fixes which parallel branch finishes
/// first: each tool made by `tool` waits until its name is `open`ed.
/// Counts calls in flight per name and the peak across names, and keeps
/// the finish order. Clones share the same gates.
#[derive(Clone, Default)]
pub struct GatedTool {
    gates: Arc<Mutex<Gates>>,
}

impl GatedTool {
    /// Builds a set of gates with every gate closed.
    /// Tools made from it wait until their gate opens.
    /// Returns the set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Builds tool `name` that waits for gate `name`, then answers with
    /// `reply(args)`. Returns it ready for `ToolRegistry::register`.
    /// Panics on an invalid tool name, as tests pass literals.
    pub fn tool(
        &self,
        name: &str,
        reply: impl Fn(Value) -> ToolOutput + Send + Sync + 'static,
    ) -> Arc<dyn Tool> {
        let (gates, reply, gate) = (self.clone(), Arc::new(reply), name.to_string());
        FnTool::new(name, move |args, ctx| {
            let (gates, reply, gate) = (gates.clone(), reply.clone(), gate.clone());
            async move {
                gates.enter(&gate, ctx.idempotency_key);
                gates.wait(&gate).await;
                gates.leave(&gate);
                Ok(reply(args))
            }
        })
    }

    /// Opens gate `name`: every waiting and later call of that tool goes on.
    /// Opening an open gate changes nothing.
    /// Wakes the waiting calls.
    pub fn open(&self, name: &str) {
        let wakers: Vec<Waker> = {
            let mut g = self.gates.lock().expect("gates lock");
            g.open.insert(name.to_string());
            let (wake, keep) = std::mem::take(&mut g.waiting)
                .into_iter()
                .partition(|(n, _)| n == name);
            g.waiting = keep;
            wake.into_iter().map(|(_, w)| w).collect()
        };
        wakers.into_iter().for_each(Waker::wake);
    }

    /// Returns how many calls of tool `name` are in flight now.
    /// A call is in flight from its start until it answers.
    /// Zero for a tool never called.
    pub fn in_flight(&self, name: &str) -> usize {
        let g = self.gates.lock().expect("gates lock");
        g.in_flight.get(name).copied().unwrap_or(0)
    }

    /// Returns the most calls of all gated tools that were in flight at
    /// once since the set was built. A concurrency cap holds when this
    /// never exceeds it.
    pub fn max_in_flight(&self) -> usize {
        self.gates.lock().expect("gates lock").peak
    }

    /// Returns the names of finished calls in finish order.
    /// A name appears once per finished call.
    /// Empty before any call finishes.
    pub fn finished(&self) -> Vec<String> {
        self.gates.lock().expect("gates lock").finished.clone()
    }

    /// Returns the idempotency key of every call of tool `name` in start
    /// order, finished or not. Its length is how many times the tool was
    /// called; empty for a tool never called.
    pub fn calls(&self, name: &str) -> Vec<Option<String>> {
        let g = self.gates.lock().expect("gates lock");
        g.calls.get(name).cloned().unwrap_or_default()
    }

    fn enter(&self, name: &str, key: Option<String>) {
        let mut g = self.gates.lock().expect("gates lock");
        g.calls.entry(name.to_string()).or_default().push(key);
        *g.in_flight.entry(name.to_string()).or_default() += 1;
        g.total += 1;
        g.peak = g.peak.max(g.total);
    }

    fn leave(&self, name: &str) {
        let mut g = self.gates.lock().expect("gates lock");
        let count = g.in_flight.get_mut(name).expect("a leaving call entered");
        assert!(*count > 0, "in-flight count never goes below zero");
        *count -= 1;
        assert!(g.total > 0, "total in flight never goes below zero");
        g.total -= 1;
        g.finished.push(name.to_string());
    }

    async fn wait(&self, name: &str) {
        std::future::poll_fn(|cx| {
            let mut g = self.gates.lock().expect("gates lock");
            if g.open.contains(name) {
                return Poll::Ready(());
            }
            g.waiting.push((name.to_string(), cx.waker().clone()));
            Poll::Pending
        })
        .await
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

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn gated_tools_finish_in_the_order_their_gates_open() {
        let gates = GatedTool::new();
        let a = gates.tool("a", |_| ToolOutput::text("A"));
        let b = gates.tool("b", |_| ToolOutput::text("B"));
        let ctx = ToolContext::default();
        let opener = async {
            while gates.in_flight("a") + gates.in_flight("b") < 2 {
                tokio::task::yield_now().await;
            }
            gates.open("b");
            while gates.finished().is_empty() {
                tokio::task::yield_now().await;
            }
            gates.open("a");
        };
        let (ra, rb, ()) = tokio::join!(a.call(json!({}), &ctx), b.call(json!({}), &ctx), opener);
        assert_eq!(ra.unwrap().content, "A");
        assert_eq!(rb.unwrap().content, "B");
        assert_eq!(gates.finished(), vec!["b".to_string(), "a".to_string()]);
        assert_eq!(gates.max_in_flight(), 2);
        assert_eq!(gates.in_flight("a"), 0);
        assert_eq!(gates.calls("a"), vec![None]);
    }
}
