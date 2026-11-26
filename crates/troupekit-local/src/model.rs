use crate::error::LocalError;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

const MAX_TOKENS_DEFAULT: usize = 512;
const TEMPERATURE_DEFAULT: f32 = 0.7;
const TOP_P_DEFAULT: f32 = 0.95;
const SEED_DEFAULT: u32 = 0x5452_4B54;
const ROOT_MAX_BYTES: usize = 64;

/// One in-process model: tokenizer, chat template, generation, scoring
/// and embeddings. Calls block; `LocalChat` runs them off the async
/// runtime. llama.cpp implements it behind `llama`; 0.4 adds the
/// platform models. Object safe, so `Arc<dyn LocalModel>` works.
pub trait LocalModel: Send + Sync {
    /// Returns the fixed facts about the loaded model.
    /// Input: none. Output: context length, vocabulary size,
    /// embedding size (0 when it cannot embed) and template presence.
    fn info(&self) -> &LocalInfo;

    /// Turns text into token ids, adding the model's start token when
    /// its vocabulary asks for one. `special` lets control tokens in the
    /// text (such as `<|im_start|>`) parse as themselves.
    /// Output: the ids, or `Tokenize` on a refused input.
    fn tokenize(&self, text: &str, special: bool) -> Result<Vec<u32>, LocalError>;

    /// Turns token ids back into text.
    /// Input: ids from this model's vocabulary.
    /// Output: the text, or `Tokenize` for an id outside the vocabulary.
    fn detokenize(&self, tokens: &[u32]) -> Result<String, LocalError>;

    /// Renders messages with the model's embedded chat template.
    /// `add_assistant` ends the text with the assistant turn opener.
    /// Output: the prompt text, or `Unsupported("chat template")`.
    fn render_chat(&self, msgs: &[ChatMessage], add_assistant: bool) -> Result<String, LocalError>;

    /// Generates from a token prompt, calling `on_token` with each piece
    /// of text; `Flow::Stop` ends the run. Stops at end of text, the
    /// token limit, a stop string or the cancel flag in `params`.
    /// Output: the `Generation`, or `ContextFull` before decoding.
    fn generate(
        &self,
        prompt: &[u32],
        params: &GenParams,
        on_token: &mut dyn FnMut(&str) -> Flow,
    ) -> Result<Generation, LocalError>;

    /// Scores a token sequence as the sum of each token's log-probability
    /// given the tokens before it.
    /// Output: the sum (at most 0), or an error from decoding.
    fn score(&self, tokens: &[u32]) -> Result<f64, LocalError>;

    /// Embeds each text into one mean-pooled, L2-normalized vector of
    /// `info().embedding_dims()` floats.
    /// Output: one vector per text, or `Unsupported("embeddings")`.
    fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, LocalError>;

    /// Drops any cached prompt state, so the next call reuses no prefix.
    /// Never blocks: called while a call runs (also from its `on_token`
    /// callback), it takes effect before the next call. Input: none.
    fn reset(&self);

    /// Counts the tokens of `text` with `tokenize`, control tokens not
    /// parsed. Input: any text.
    /// Output: the count, or the `tokenize` error.
    fn count_tokens(&self, text: &str) -> Result<usize, LocalError> {
        Ok(self.tokenize(text, false)?.len())
    }
}

/// Fixed facts about a loaded model: context length in tokens,
/// vocabulary size, embedding size (0 when the model cannot embed) and
/// whether it carries a chat template. Read through the getters.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LocalInfo {
    context_length: usize,
    vocab_size: usize,
    embedding_dims: usize,
    chat_template: bool,
}

impl LocalInfo {
    /// Builds the facts for a `LocalModel` implementation.
    /// Inputs: context length, vocabulary size, embedding size (0 for
    /// none), template presence. Panics when context or vocabulary is 0.
    pub fn new(
        context_length: usize,
        vocab_size: usize,
        embedding_dims: usize,
        chat_template: bool,
    ) -> Self {
        assert!(
            context_length > 0,
            "a model has a context of at least one token"
        );
        assert!(
            vocab_size > 0,
            "a model has at least one token in its vocabulary"
        );
        Self {
            context_length,
            vocab_size,
            embedding_dims,
            chat_template,
        }
    }

    /// Returns the context length in tokens.
    /// Prompt plus generated tokens never exceed it.
    /// Always above 0.
    pub fn context_length(&self) -> usize {
        self.context_length
    }

    /// Returns the number of tokens in the vocabulary.
    /// Every id the model returns is below it.
    /// Always above 0.
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Returns the length of each embedding vector.
    /// 0 means `embed` returns `Unsupported`.
    /// `LocalEmbedder` refuses a model with 0.
    pub fn embedding_dims(&self) -> usize {
        self.embedding_dims
    }

    /// Returns whether the model carries a chat template.
    /// Without one `render_chat` returns `Unsupported`.
    /// `LocalChat` needs it.
    pub fn chat_template(&self) -> bool {
        self.chat_template
    }
}

/// The speaker of one chat message. `as_str` gives the role name that
/// chat templates expect: `system`, `user`, `assistant`, `tool`.
/// Default `User`.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ChatRole {
    System,
    #[default]
    User,
    Assistant,
    Tool,
}

impl ChatRole {
    /// Returns the role name passed to the chat template.
    /// Input: none. Output: a lowercase static name.
    /// The names match the common template conventions.
    pub fn as_str(&self) -> &'static str {
        match self {
            ChatRole::System => "system",
            ChatRole::User => "user",
            ChatRole::Assistant => "assistant",
            ChatRole::Tool => "tool",
        }
    }
}

/// One message given to `render_chat`: a role and its text.
/// Built with `new`; read through `role` and `content`.
/// The text is passed to the template unchanged.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatMessage {
    role: ChatRole,
    content: String,
}

impl ChatMessage {
    /// Builds a message.
    /// Inputs: the speaker and the text.
    /// Output: the message.
    pub fn new(role: ChatRole, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
        }
    }

    /// Returns the speaker.
    /// Input: none.
    /// Output: the role.
    pub fn role(&self) -> ChatRole {
        self.role
    }

    /// Returns the text.
    /// Input: none.
    /// Output: the text as given to `new`.
    pub fn content(&self) -> &str {
        &self.content
    }
}

/// A GBNF grammar that constrains generation, with the name of its start
/// rule. Built by `Grammar::new`, which refuses empty text, text over
/// `MAX_BYTES`, an interior NUL, and a root that is not a rule name.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Grammar {
    gbnf: String,
    root: String,
}

impl Grammar {
    /// The longest grammar text accepted, in bytes.
    /// Setting S27 of the 0.3 plan: 256 KiB.
    /// `new` refuses longer text.
    pub const MAX_BYTES: usize = 256 * 1024;

    /// Checks and builds a grammar.
    /// Inputs: GBNF text and the start rule name (letters, digits, `-`,
    /// `_`, at most 64 bytes). Output: the grammar, or `LocalError::Grammar`.
    pub fn new(gbnf: &str, root: &str) -> Result<Self, LocalError> {
        if gbnf.trim().is_empty() {
            return Err(LocalError::Grammar("grammar text is empty".into()));
        }
        if gbnf.len() > Self::MAX_BYTES {
            return Err(LocalError::Grammar(format!(
                "grammar is {} bytes, limit {}",
                gbnf.len(),
                Self::MAX_BYTES
            )));
        }
        if gbnf.contains('\0') {
            return Err(LocalError::Grammar("grammar text holds a NUL byte".into()));
        }
        let root_ok = !root.is_empty()
            && root.len() <= ROOT_MAX_BYTES
            && root
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_');
        if !root_ok {
            return Err(LocalError::Grammar(format!(
                "root rule name {root:?} is not a rule name"
            )));
        }
        Ok(Self {
            gbnf: gbnf.to_string(),
            root: root.to_string(),
        })
    }

    /// Returns the GBNF text.
    /// Never empty, never over `MAX_BYTES`, no NUL.
    /// Passed to the model unchanged.
    pub fn gbnf(&self) -> &str {
        &self.gbnf
    }

    /// Returns the start rule name.
    /// ASCII letters, digits, `-` and `_` only.
    /// At most 64 bytes.
    pub fn root(&self) -> &str {
        &self.root
    }
}

/// Settings for one `generate` call. Defaults: 512 tokens, temperature
/// 0.7, top-p 0.95, seed `0x5452_4B54`, no stop strings, no grammar, no
/// confidence, end of text allowed, no cancel flag. `with_*` setters.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct GenParams {
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
    seed: u32,
    stop: Vec<String>,
    grammar: Option<Grammar>,
    confidence: bool,
    ban_eos: bool,
    cancel: Option<Arc<AtomicBool>>,
}

impl Default for GenParams {
    fn default() -> Self {
        Self {
            max_tokens: MAX_TOKENS_DEFAULT,
            temperature: TEMPERATURE_DEFAULT,
            top_p: TOP_P_DEFAULT,
            seed: SEED_DEFAULT,
            stop: Vec::new(),
            grammar: None,
            confidence: false,
            ban_eos: false,
            cancel: None,
        }
    }
}

impl GenParams {
    /// Sets the most tokens generated in this call.
    /// The model also stops at its context limit.
    /// Default 512.
    pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Sets the sampling temperature; 0 picks the most likely token.
    /// A negative or non-finite value is refused by the model.
    /// Default 0.7.
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }

    /// Sets the nucleus sampling share, from 0 (exclusive) to 1.
    /// 1 keeps every token.
    /// Default 0.95.
    pub fn with_top_p(mut self, top_p: f32) -> Self {
        self.top_p = top_p;
        self
    }

    /// Sets the sampler seed.
    /// The same seed, prompt and model give the same text.
    /// Default `0x5452_4B54`.
    pub fn with_seed(mut self, seed: u32) -> Self {
        self.seed = seed;
        self
    }

    /// Sets strings that end generation when they appear.
    /// The stop string is not part of the returned text.
    /// Default none.
    pub fn with_stop(mut self, stop: Vec<String>) -> Self {
        self.stop = stop;
        self
    }

    /// Sets the grammar every generated token must follow.
    /// Tool calls and structured output use it.
    /// Default none.
    pub fn with_grammar(mut self, grammar: Option<Grammar>) -> Self {
        self.grammar = grammar;
        self
    }

    /// Asks for `Generation::mean_logprob`.
    /// Costs one softmax over the vocabulary per token.
    /// Default false.
    pub fn with_confidence(mut self, confidence: bool) -> Self {
        self.confidence = confidence;
        self
    }

    /// Bans end-of-text tokens, so the run reaches `max_tokens`.
    /// For fixed-length runs: the bench and the kill test.
    /// Default false.
    pub fn with_ban_eos(mut self, ban_eos: bool) -> Self {
        self.ban_eos = ban_eos;
        self
    }

    /// Sets a flag that stops generation within one token once true.
    /// `LocalChat` sets it when its future is dropped.
    /// Default none.
    pub fn with_cancel(mut self, cancel: Option<Arc<AtomicBool>>) -> Self {
        self.cancel = cancel;
        self
    }

    /// Returns the token limit of the call.
    /// Input: none.
    /// Default 512.
    pub fn max_tokens(&self) -> usize {
        self.max_tokens
    }

    /// Returns the sampling temperature.
    /// Input: none.
    /// Default 0.7.
    pub fn temperature(&self) -> f32 {
        self.temperature
    }

    /// Returns the nucleus sampling share.
    /// Input: none.
    /// Default 0.95.
    pub fn top_p(&self) -> f32 {
        self.top_p
    }

    /// Returns the sampler seed.
    /// Input: none.
    /// Default `0x5452_4B54`.
    pub fn seed(&self) -> u32 {
        self.seed
    }

    /// Returns the stop strings.
    /// Input: none.
    /// Default empty.
    pub fn stop(&self) -> &[String] {
        &self.stop
    }

    /// Returns the grammar, if any.
    /// Input: none.
    /// Default none.
    pub fn grammar(&self) -> Option<&Grammar> {
        self.grammar.as_ref()
    }

    /// Returns whether `mean_logprob` is computed.
    /// Input: none.
    /// Default false.
    pub fn confidence(&self) -> bool {
        self.confidence
    }

    /// Returns whether end-of-text tokens are banned.
    /// Input: none.
    /// Default false.
    pub fn ban_eos(&self) -> bool {
        self.ban_eos
    }

    /// Returns the cancel flag, if any.
    /// Input: none.
    /// Default none.
    pub fn cancel(&self) -> Option<&Arc<AtomicBool>> {
        self.cancel.as_ref()
    }
}

/// Why a generation ended: the model's end of text, the token limit, a
/// stop string or `Flow::Stop`, or the cancel flag. A full context is an
/// error (`LocalError::ContextFull`), not a stop.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GenStop {
    EndOfText,
    MaxTokens,
    Stop,
    Cancelled,
}

/// Speed and memory of one call: prefill and decode tokens per second,
/// time to first token, peak resident memory when the platform reports
/// it, and prompt tokens reused from the cache. All zero by default.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Default)]
pub struct LocalMetrics {
    prefill_tokens_per_second: f64,
    decode_tokens_per_second: f64,
    time_to_first_token_ms: f64,
    peak_ram_bytes: Option<u64>,
    reused_prefix_tokens: usize,
}

impl LocalMetrics {
    /// Sets the prompt processing speed in tokens per second.
    /// Input: a finite value at least 0.
    /// Default 0.
    pub fn with_prefill_tokens_per_second(mut self, value: f64) -> Self {
        self.prefill_tokens_per_second = value;
        self
    }

    /// Sets the generation speed in tokens per second.
    /// Input: a finite value at least 0.
    /// Default 0.
    pub fn with_decode_tokens_per_second(mut self, value: f64) -> Self {
        self.decode_tokens_per_second = value;
        self
    }

    /// Sets the time from the call to the first token, in milliseconds.
    /// Input: a finite value at least 0.
    /// Default 0.
    pub fn with_time_to_first_token_ms(mut self, value: f64) -> Self {
        self.time_to_first_token_ms = value;
        self
    }

    /// Sets the peak resident memory of the process in bytes.
    /// None where the platform has no query.
    /// Default none.
    pub fn with_peak_ram_bytes(mut self, value: Option<u64>) -> Self {
        self.peak_ram_bytes = value;
        self
    }

    /// Sets how many prompt tokens came from the cache, not decoding.
    /// Input: at most the prompt length.
    /// Default 0.
    pub fn with_reused_prefix_tokens(mut self, value: usize) -> Self {
        self.reused_prefix_tokens = value;
        self
    }

    /// Returns the prompt processing speed in tokens per second.
    /// Input: none.
    /// Default 0.
    pub fn prefill_tokens_per_second(&self) -> f64 {
        self.prefill_tokens_per_second
    }

    /// Returns the generation speed in tokens per second.
    /// Input: none.
    /// Default 0.
    pub fn decode_tokens_per_second(&self) -> f64 {
        self.decode_tokens_per_second
    }

    /// Returns the time to the first token in milliseconds.
    /// Input: none.
    /// Default 0.
    pub fn time_to_first_token_ms(&self) -> f64 {
        self.time_to_first_token_ms
    }

    /// Returns the peak resident memory in bytes, when known.
    /// Input: none.
    /// Default none.
    pub fn peak_ram_bytes(&self) -> Option<u64> {
        self.peak_ram_bytes
    }

    /// Returns how many prompt tokens were reused from the cache.
    /// Input: none.
    /// Default 0.
    pub fn reused_prefix_tokens(&self) -> usize {
        self.reused_prefix_tokens
    }
}

/// The result of one `generate` call: text, generated token count, why
/// it stopped, the mean token log-probability when asked for, and the
/// call's metrics. Built by `LocalModel` implementations with `new`.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct Generation {
    text: String,
    tokens: usize,
    stop: GenStop,
    mean_logprob: Option<f64>,
    metrics: LocalMetrics,
}

impl Generation {
    /// Builds a result from text, token count, stop reason, mean
    /// log-prob (`None` unless asked for) and metrics.
    /// Panics when the mean log-prob is above 0 or NaN.
    pub fn new(
        text: String,
        tokens: usize,
        stop: GenStop,
        mean_logprob: Option<f64>,
        metrics: LocalMetrics,
    ) -> Self {
        if let Some(lp) = mean_logprob {
            assert!(lp <= 0.0, "a mean log-probability is at most 0");
        }
        Self {
            text,
            tokens,
            stop,
            mean_logprob,
            metrics,
        }
    }

    /// Returns the generated text, stop string excluded.
    /// Input: none.
    /// Output: the text.
    pub fn text(&self) -> &str {
        &self.text
    }

    /// Returns how many tokens were generated.
    /// Input: none.
    /// Output: the count.
    pub fn tokens(&self) -> usize {
        self.tokens
    }

    /// Returns why generation ended.
    /// Input: none.
    /// Output: the stop reason.
    pub fn stop(&self) -> GenStop {
        self.stop
    }

    /// Returns the mean token log-probability.
    /// Only set when `GenParams::confidence` was true.
    /// Output: at most 0.
    pub fn mean_logprob(&self) -> Option<f64> {
        self.mean_logprob
    }

    /// Returns the call's speed and memory figures.
    /// Input: none.
    /// Output: the metrics.
    pub fn metrics(&self) -> &LocalMetrics {
        &self.metrics
    }
}

/// What a token callback tells `generate`: keep going or stop now.
/// `Stop` ends the run with `GenStop::Stop`.
/// Used by streaming and by stop checks in the caller.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Flow {
    Continue,
    Stop,
}
