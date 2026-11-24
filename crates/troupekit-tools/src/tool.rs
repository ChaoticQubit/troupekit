use async_trait::async_trait;
use serde_json::Value;
use std::sync::Arc;
use troupekit_core::http::{HttpError, HttpErrorKind};
use troupekit_core::ids::RunId;
use troupekit_core::tool::{HostedTool, ToolSpec};

/// Per-call context handed to every tool.
/// Cheap to clone, so typed tool closures can own a copy across awaits.
/// `run_id` names the run that made the call; `idempotency_key` is the same
/// for a replayed call and new for every other, for tools that dedupe.
#[derive(Debug, Clone, Default)]
pub struct ToolContext {
    pub run_id: Option<RunId>,
    pub idempotency_key: Option<String>,
}

impl ToolContext {
    /// Builds the context for call `index` of step `step` of run `run_id`.
    /// The key is `"{run_id}:{step}:{index}"`, so replaying a step from its
    /// checkpoint gives the same keys and the next step new ones.
    pub fn for_call(run_id: RunId, step: u64, index: usize) -> ToolContext {
        ToolContext {
            run_id: Some(run_id),
            idempotency_key: Some(format!("{run_id}:{step}:{index}")),
        }
    }
}

/// What a tool returns to the model.
/// `is_error` marks content that describes a failure the model may react to,
/// such as an HTTP error status from a remote API.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolOutput {
    pub content: String,
    pub is_error: bool,
}

impl ToolOutput {
    /// Builds a successful output holding `content`.
    /// Returns the output with `is_error` false.
    pub fn text(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            is_error: false,
        }
    }

    /// Builds an error output holding `content`.
    /// Returns the output with `is_error` true.
    pub fn error(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            is_error: true,
        }
    }

    /// Folds a tool result into what the model sees.
    /// `Ok` passes through; `Err` becomes an error output whose content is
    /// the error's display text, which never holds secret values.
    pub fn from_result(result: Result<ToolOutput, ToolError>) -> Self {
        result.unwrap_or_else(|e| Self::error(e.to_string()))
    }
}

/// Why a tool call or tool setup failed.
/// Unknown tools, schema-invalid arguments, blocked addresses, transport
/// and decode failures and bad configuration each have their own variant.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ToolError {
    #[error("invalid arguments: {0}")]
    InvalidArgs(String),
    #[error("unknown tool: {0}")]
    UnknownTool(String),
    #[error("duplicate tool: {0}")]
    DuplicateTool(String),
    #[error("invalid tool spec: {0}")]
    InvalidSpec(String),
    #[error("config: {0}")]
    Config(String),
    #[error("blocked: {0}")]
    Blocked(String),
    #[error("http: {0}")]
    Http(String),
    #[error("decode: {0}")]
    Decode(String),
    #[error("failed: {0}")]
    Failed(String),
}

impl From<HttpError> for ToolError {
    fn from(e: HttpError) -> Self {
        match e.kind {
            HttpErrorKind::Invalid => ToolError::InvalidArgs(e.message),
            HttpErrorKind::Blocked => ToolError::Blocked(e.message),
            _ => ToolError::Http(e.message),
        }
    }
}

/// A capability an agent can call: a spec the model sees and an async call.
/// Object safe, so registries hold `Arc<dyn Tool>`. Arguments arrive as JSON
/// already checked against `spec().schema` when called through a registry.
#[async_trait]
pub trait Tool: Send + Sync {
    /// Returns the name, description and argument schema the model sees.
    fn spec(&self) -> &ToolSpec;

    /// Runs the tool with JSON `args` and the per-call `ctx`.
    /// Returns the model-facing output or a `ToolError`.
    async fn call(&self, args: Value, ctx: &ToolContext) -> Result<ToolOutput, ToolError>;
}

struct Hosted {
    inner: Arc<dyn Tool>,
    spec: ToolSpec,
}

#[async_trait]
impl Tool for Hosted {
    fn spec(&self) -> &ToolSpec {
        &self.spec
    }

    async fn call(&self, args: Value, ctx: &ToolContext) -> Result<ToolOutput, ToolError> {
        self.inner.call(args, ctx).await
    }
}

/// Marks local tool `inner` as standing in for server-side tool `kind`:
/// vendors that host it run theirs, every other vendor calls `inner`.
/// Panics when `inner` is not named as `kind` requires.
pub fn hosted(inner: Arc<dyn Tool>, kind: HostedTool) -> Arc<dyn Tool> {
    assert_eq!(
        inner.spec().name,
        kind.name(),
        "a hosted stand-in carries the hosted tool's name"
    );
    let spec = ToolSpec {
        hosted: Some(kind),
        ..inner.spec().clone()
    };
    Arc::new(Hosted { inner, spec })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn for_call_builds_stable_keys() {
        let run = RunId::new();
        let a = ToolContext::for_call(run, 3, 1);
        assert_eq!(a.run_id, Some(run));
        assert_eq!(a.idempotency_key, Some(format!("{run}:3:1")));
        assert_eq!(
            ToolContext::for_call(run, 3, 1).idempotency_key,
            a.idempotency_key
        );
        assert_ne!(
            ToolContext::for_call(run, 4, 1).idempotency_key,
            a.idempotency_key
        );
        assert_eq!(ToolContext::default().idempotency_key, None);
    }
}
