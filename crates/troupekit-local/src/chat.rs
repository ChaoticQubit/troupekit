use crate::error::LocalError;
use crate::model::{ChatMessage, ChatRole, Flow, GenParams, GenStop, Generation, LocalModel};
use crate::reply::parse_reply;
use crate::tools::{request_grammar, turn_prompt, ToolMode};
use async_trait::async_trait;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use troupekit_core::error::{ProviderError, ProviderErrorKind};
use troupekit_core::event::Usage;
use troupekit_core::message::{ContentPart, Message, Role};
use troupekit_providers::request::{CompletionRequest, CompletionResponse, Delta, StopReason};
use troupekit_providers::Model;

/// Settings of a `LocalChat`: tool mode, the temperature used when a
/// request has none, the sampling seed, whether end of text is banned
/// and an optional minimum confidence. Defaults are plan rows S18 to S22.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct LocalChatSettings {
    tool_mode: ToolMode,
    default_temperature: f64,
    seed: u32,
    ban_eos: bool,
    min_confidence: Option<f64>,
}

impl Default for LocalChatSettings {
    fn default() -> Self {
        Self {
            tool_mode: ToolMode::Auto,
            default_temperature: 0.7,
            seed: 0x5452_4B54,
            ban_eos: false,
            min_confidence: None,
        }
    }
}

impl LocalChatSettings {
    /// Sets whether the model may answer in text when tools are offered.
    /// `Required` asks for exactly one tool call.
    /// Default `Auto`.
    pub fn with_tool_mode(mut self, tool_mode: ToolMode) -> Self {
        self.tool_mode = tool_mode;
        self
    }

    /// Sets the temperature used when a request has none (0 to 2).
    /// Values outside the range are clamped.
    /// Default 0.7.
    pub fn with_default_temperature(mut self, temperature: f64) -> Self {
        self.default_temperature = if temperature.is_finite() {
            temperature.clamp(0.0, 2.0)
        } else {
            0.7
        };
        self
    }

    /// Sets the sampling seed, so runs repeat for the same input.
    /// Input: any u32.
    /// Default `0x5452_4B54`.
    pub fn with_seed(mut self, seed: u32) -> Self {
        self.seed = seed;
        self
    }

    /// Bans end of text, so generation runs to `max_tokens`.
    /// Only for fixed-length runs such as benchmarks and kill tests.
    /// Default false.
    pub fn with_ban_eos(mut self, ban_eos: bool) -> Self {
        self.ban_eos = ban_eos;
        self
    }

    /// Sets the minimum confidence (geometric mean token probability,
    /// 0 to 1) below which a call fails with a `Server` error a router
    /// may fall back on. While set, a streamed reply is held and sent only
    /// after it passes. Default `None` (off: one softmax per token).
    pub fn with_min_confidence(mut self, min_confidence: Option<f64>) -> Self {
        self.min_confidence = min_confidence
            .filter(|c| c.is_finite())
            .map(|c| c.clamp(0.0, 1.0));
        self
    }

    /// Returns the tool mode.
    /// Input: none.
    /// Default `Auto`.
    pub fn tool_mode(&self) -> ToolMode {
        self.tool_mode
    }

    /// Returns the temperature used when a request has none.
    /// Input: none.
    /// Default 0.7.
    pub fn default_temperature(&self) -> f64 {
        self.default_temperature
    }

    /// Returns the sampling seed.
    /// Input: none.
    /// Default `0x5452_4B54`.
    pub fn seed(&self) -> u32 {
        self.seed
    }

    /// Returns whether end of text is banned.
    /// Input: none.
    /// Default false.
    pub fn ban_eos(&self) -> bool {
        self.ban_eos
    }

    /// Returns the minimum confidence, or `None` when off.
    /// Input: none.
    /// Default `None`.
    pub fn min_confidence(&self) -> Option<f64> {
        self.min_confidence
    }
}

/// A local model behind the `Model` interface, so the same agent runs
/// on device. Renders requests through the model's chat template,
/// generates on a blocking thread and cancels when the future is dropped.
pub struct LocalChat {
    model: Arc<dyn LocalModel>,
    id: String,
    settings: LocalChatSettings,
}

impl std::fmt::Debug for LocalChat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LocalChat")
            .field("id", &self.id)
            .field("settings", &self.settings)
            .finish()
    }
}

struct CancelOnDrop(Arc<AtomicBool>);

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

impl LocalChat {
    /// Wraps a loaded local model with default settings.
    /// Input: the model and the id `Model::id` returns.
    /// Output: the chat model.
    pub fn new(model: Arc<dyn LocalModel>, id: impl Into<String>) -> Self {
        Self {
            model,
            id: id.into(),
            settings: LocalChatSettings::default(),
        }
    }

    /// Replaces the settings.
    /// Input: the settings.
    /// Output: the chat model.
    pub fn with_settings(mut self, settings: LocalChatSettings) -> Self {
        self.settings = settings;
        self
    }

    /// Returns the settings.
    /// Input: none.
    /// Output: the settings.
    pub fn settings(&self) -> &LocalChatSettings {
        &self.settings
    }

    fn render(&self, req: &CompletionRequest) -> Result<Vec<ChatMessage>, ProviderError> {
        let mut system = req.system_text().unwrap_or_default();
        let tools = turn_prompt(
            &req.tools,
            self.settings.tool_mode,
            req.output_schema.as_ref(),
        );
        if !tools.is_empty() {
            if !system.is_empty() {
                system.push_str("\n\n");
            }
            system.push_str(&tools);
        }
        let mut out = Vec::new();
        if !system.is_empty() {
            out.push(ChatMessage::new(ChatRole::System, system));
        }
        for message in &req.messages {
            let role = match message.role {
                Role::System => continue,
                Role::User => ChatRole::User,
                Role::Assistant => ChatRole::Assistant,
                Role::Tool => ChatRole::User,
            };
            let mut lines = Vec::new();
            for part in &message.parts {
                match part {
                    ContentPart::Text { text } => lines.push(text.clone()),
                    ContentPart::ToolCall { name, arguments, .. } => {
                        lines.push(serde_json::json!({ "tool": name, "arguments": arguments }).to_string())
                    }
                    ContentPart::ToolResult { call_id, content, is_error } => lines.push(format!(
                        "Tool output, JSON encoded: {}",
                        serde_json::json!({ "call_id": call_id, "content": content, "is_error": is_error })
                    )),
                    ContentPart::Image { .. } => {
                        return Err(ProviderError::new(
                            ProviderErrorKind::BadRequest,
                            "local models do not take images",
                        ))
                    }
                    ContentPart::Hosted { .. } => {}
                }
            }
            if !lines.is_empty() {
                out.push(ChatMessage::new(role, lines.join("\n")));
            }
        }
        Ok(out)
    }

    async fn run(
        &self,
        req: &CompletionRequest,
        on_delta: &mut (dyn for<'d> FnMut(&'d Delta) + Send),
    ) -> Result<CompletionResponse, ProviderError> {
        req.validate()?;
        let req = req.sendable(false)?;
        let grammar = request_grammar(
            &req.tools,
            self.settings.tool_mode,
            req.output_schema.as_ref(),
        )
        .map_err(server)?;
        let messages = self.render(&req)?;
        for message in &messages {
            self.refuse_control_tokens(message.content())?;
        }
        let rendered = self.model.render_chat(&messages, true).map_err(server)?;
        let prompt = self.model.tokenize(&rendered, true).map_err(server)?;
        let limit = self.model.info().context_length();
        if prompt.len() >= limit {
            return Err(server(LocalError::ContextFull {
                needed: prompt.len() + 1,
                limit,
            }));
        }
        let max_tokens = (req.max_tokens as usize).min(limit - prompt.len());
        assert!(max_tokens > 0 && prompt.len() + max_tokens <= limit);
        let cancel = Arc::new(AtomicBool::new(false));
        let _guard = CancelOnDrop(cancel.clone());
        let temperature = req.temperature.unwrap_or(self.settings.default_temperature);
        let params = GenParams::default()
            .with_max_tokens(max_tokens)
            .with_temperature(temperature as f32)
            .with_seed(self.settings.seed)
            .with_ban_eos(self.settings.ban_eos)
            .with_grammar(grammar)
            .with_confidence(self.settings.min_confidence.is_some())
            .with_cancel(Some(cancel));
        let prompt_tokens = prompt.len();
        let require_call = self.settings.tool_mode == ToolMode::Required && !req.tools.is_empty();
        let hold_all = require_call || self.settings.min_confidence.is_some();
        let mut gate = TextGate::new(!req.tools.is_empty(), hold_all);
        let generation = self.generate(prompt, params, &mut gate, on_delta).await?;
        self.check_confidence(&generation)?;
        let earlier_calls = req
            .messages
            .iter()
            .flat_map(|m| &m.parts)
            .filter(|p| matches!(p, ContentPart::ToolCall { .. }))
            .count();
        let parts: Vec<ContentPart> = parse_reply(generation.text(), &req.tools)
            .map_err(server)?
            .into_iter()
            .map(|p| match p {
                ContentPart::ToolCall {
                    name, arguments, ..
                } => ContentPart::ToolCall {
                    id: format!("local-{}", earlier_calls + 1),
                    name,
                    arguments,
                },
                other => other,
            })
            .collect();
        let call = parts.iter().find_map(|p| match p {
            ContentPart::ToolCall {
                id,
                name,
                arguments,
            } => Some((id.clone(), name.clone(), arguments.to_string())),
            _ => None,
        });
        if require_call && call.is_none() {
            return Err(ProviderError::new(
                ProviderErrorKind::Server,
                "local model answered in text, but tool_mode Required needs a tool call",
            ));
        }
        match &call {
            Some((id, name, json)) => {
                on_delta(&Delta::ToolCallStarted {
                    index: 0,
                    id: id.clone(),
                    name: name.clone(),
                });
                on_delta(&Delta::ToolCallArgs {
                    index: 0,
                    json: json.clone(),
                });
            }
            None => {
                if let Some(text) = gate.flush() {
                    on_delta(&Delta::TextDelta { text });
                }
            }
        }
        let stop = match (call.is_some(), generation.stop()) {
            (true, _) => StopReason::ToolUse,
            (false, GenStop::MaxTokens) => StopReason::MaxTokens,
            (false, GenStop::Cancelled) => StopReason::Other("cancelled".into()),
            (false, _) => StopReason::EndTurn,
        };
        let usage = Usage {
            input_tokens: prompt_tokens as u64,
            output_tokens: generation.tokens() as u64,
            cost_usd_micros: 0,
        };
        on_delta(&Delta::Usage {
            input_tokens: Some(usage.input_tokens),
            output_tokens: Some(usage.output_tokens),
        });
        on_delta(&Delta::Stop {
            reason: stop.clone(),
        });
        let message = Message::new(Role::Assistant, parts)
            .map_err(|e| ProviderError::new(ProviderErrorKind::Server, e.to_string()))?;
        Ok(CompletionResponse {
            message,
            usage,
            stop,
        })
    }

    fn refuse_control_tokens(&self, content: &str) -> Result<(), ProviderError> {
        let plain = self.model.tokenize(content, false).map_err(server)?;
        let special = self.model.tokenize(content, true).map_err(server)?;
        if plain != special {
            return Err(ProviderError::new(
                ProviderErrorKind::BadRequest,
                "message text holds chat template control tokens",
            ));
        }
        Ok(())
    }

    fn check_confidence(&self, generation: &Generation) -> Result<(), ProviderError> {
        let (Some(min), Some(lp)) = (self.settings.min_confidence, generation.mean_logprob())
        else {
            return Ok(());
        };
        let confidence = lp.exp();
        assert!((0.0..=1.0).contains(&confidence));
        if confidence < min {
            return Err(ProviderError::new(
                ProviderErrorKind::Server,
                format!("local confidence {confidence:.3} below {min}"),
            ));
        }
        Ok(())
    }

    async fn generate(
        &self,
        prompt: Vec<u32>,
        params: GenParams,
        gate: &mut TextGate,
        on_delta: &mut (dyn for<'d> FnMut(&'d Delta) + Send),
    ) -> Result<Generation, ProviderError> {
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<String>();
        let model = self.model.clone();
        let ctx = troupekit_trace::current();
        let handle = tokio::task::spawn_blocking(move || {
            let mut on_token = |piece: &str| {
                if tx.send(piece.to_string()).is_err() {
                    Flow::Stop
                } else {
                    Flow::Continue
                }
            };
            match ctx {
                Some(ctx) => ctx.enter(|| model.generate(&prompt, &params, &mut on_token)),
                None => model.generate(&prompt, &params, &mut on_token),
            }
        });
        while let Some(piece) = rx.recv().await {
            if let Some(text) = gate.push(&piece) {
                on_delta(&Delta::TextDelta { text });
            }
        }
        handle
            .await
            .map_err(|e| {
                ProviderError::new(
                    ProviderErrorKind::Server,
                    format!("local generation stopped: {e}"),
                )
            })?
            .map_err(server)
    }
}

fn server(e: LocalError) -> ProviderError {
    ProviderError::new(ProviderErrorKind::Server, e.to_string())
}

struct TextGate {
    buffer: String,
    state: GateState,
}

#[derive(PartialEq)]
enum GateState {
    Deciding,
    Text,
    Held,
}

impl TextGate {
    fn new(tools: bool, hold_all: bool) -> Self {
        Self {
            buffer: String::new(),
            state: match (tools, hold_all) {
                (_, true) => GateState::Held,
                (true, false) => GateState::Deciding,
                (false, false) => GateState::Text,
            },
        }
    }

    fn push(&mut self, piece: &str) -> Option<String> {
        match self.state {
            GateState::Text => (!piece.is_empty()).then(|| piece.to_string()),
            GateState::Held => {
                self.buffer.push_str(piece);
                None
            }
            GateState::Deciding => {
                self.buffer.push_str(piece);
                match self.buffer.trim_start().chars().next() {
                    None => None,
                    Some('{') => {
                        self.state = GateState::Held;
                        None
                    }
                    Some(_) => {
                        self.state = GateState::Text;
                        Some(std::mem::take(&mut self.buffer))
                    }
                }
            }
        }
    }

    fn flush(&mut self) -> Option<String> {
        let held = self.state != GateState::Text;
        self.state = GateState::Text;
        let text = std::mem::take(&mut self.buffer);
        (held && !text.is_empty()).then_some(text)
    }
}

#[async_trait]
impl Model for LocalChat {
    fn id(&self) -> &str {
        &self.id
    }

    async fn complete(&self, req: &CompletionRequest) -> Result<CompletionResponse, ProviderError> {
        self.run(req, &mut |_| {}).await
    }

    async fn stream(
        &self,
        req: &CompletionRequest,
        on_delta: &mut (dyn for<'d> FnMut(&'d Delta) + Send),
    ) -> Result<CompletionResponse, ProviderError> {
        self.run(req, on_delta).await
    }
}
