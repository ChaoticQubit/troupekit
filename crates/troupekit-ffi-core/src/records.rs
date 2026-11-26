/// Token counts and cost of a run or step, as plain numbers.
/// `cost_usd_micros` is millionths of a US dollar from the model price;
/// host models are free unless priced with `host_model_priced`.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FfiUsage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cost_usd_micros: u64,
}

/// Result of one finished agent run: the answer `text`, the typed
/// output as JSON text (`output_json`), `usage`, the number of model
/// calls and the run id text.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RunResult {
    pub text: String,
    pub output_json: String,
    pub usage: FfiUsage,
    pub model_calls: u32,
    pub run_id: String,
}

impl RunResult {
    /// Copies a kit run outcome into a plain record: the typed output
    /// becomes JSON text and the run id its text form.
    /// Output: the record.
    pub(crate) fn from_outcome(outcome: &troupekit_kit::RunOutcome) -> RunResult {
        let usage = outcome.usage();
        RunResult {
            text: outcome.text().to_string(),
            output_json: outcome.output().to_string(),
            usage: FfiUsage {
                input_tokens: usage.input_tokens,
                output_tokens: usage.output_tokens,
                cost_usd_micros: usage.cost_usd_micros,
            },
            model_calls: outcome.model_calls(),
            run_id: outcome.run_id().to_string(),
        }
    }
}

/// Result of one finished graph run: the last node's output and the
/// whole state as JSON text, `usage`, the steps taken and the run id
/// text.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct GraphResult {
    pub output_json: String,
    pub state_json: String,
    pub usage: FfiUsage,
    pub steps: u64,
    pub run_id: String,
}

/// Where a run stands after one `resume_for` window: `Done` with the
/// answer text, or `Pending` when the budget ran out first; the run
/// then continues from its last checkpoint on the next call.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Progress {
    Done { text: String },
    Pending { run_id: String },
}

impl Progress {
    /// Copies a kit `Progress` into the neutral one.
    /// Output: `Done` or `Pending`, or `Internal` for a kit state added
    /// later that has no neutral form yet.
    pub(crate) fn from_kit(progress: troupekit_kit::Progress) -> Result<Progress, crate::FfiError> {
        match progress {
            troupekit_kit::Progress::Done(text) => Ok(Progress::Done { text }),
            troupekit_kit::Progress::Pending { run_id } => Ok(Progress::Pending {
                run_id: run_id.to_string(),
            }),
            #[allow(unreachable_patterns)]
            _ => Err(crate::FfiError::new(
                crate::ErrorCode::Internal,
                "resume_for returned a progress state this version does not know",
            )),
        }
    }
}

/// Why a run paused: a graph node waits for an approval or an answer,
/// or a budget `limit` (`call`, `run`, a period) would be passed.
/// Resume after the host has dealt with it.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FfiPause {
    Approval { node: String },
    Question { node: String },
    Budget { limit: String },
}

/// What happened in one event, mirroring the engine's event bodies:
/// run and step lifecycle, text deltas, tool calls, pauses, the end.
/// Hosts match it with a fallback arm, since new kinds may be added.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FfiEventKind {
    RunStarted {
        workflow: String,
    },
    StepStarted {
        step: String,
        persona: Option<String>,
    },
    TextDelta {
        text: String,
    },
    ToolCallStarted {
        call_id: String,
        name: String,
    },
    ToolCallFinished {
        call_id: String,
        is_error: bool,
    },
    StepFinished {
        step: String,
        usage: FfiUsage,
    },
    RunPaused {
        reason: FfiPause,
    },
    RunFinished {
        usage: FfiUsage,
    },
    RunFailed {
        error: String,
    },
}

/// One event of a run: its run id text, its `seq` (ordered within one
/// run id), the clock time in milliseconds since the Unix epoch, and
/// the `kind`. Delivered by `RunStream::next` or an `EventListener`.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FfiEvent {
    pub run_id: String,
    pub seq: u64,
    pub at_ms: i64,
    pub kind: FfiEventKind,
}

/// What a host model can do: native tool calls (`tools`), JSON Schema
/// constrained output (`json_schema`), token streaming (`streaming`)
/// and its context window in tokens. Default: none, 4096 tokens.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HostModelCaps {
    pub tools: bool,
    pub json_schema: bool,
    pub streaming: bool,
    pub context_tokens: u32,
}

impl Default for HostModelCaps {
    /// Returns the most limited model: no native tools, no schema, no
    /// streaming, a 4096-token context. Hosts switch on what their model
    /// supports, field by field.
    fn default() -> Self {
        HostModelCaps {
            tools: false,
            json_schema: false,
            streaming: false,
            context_tokens: 4096,
        }
    }
}

/// One message of a host model request: `role` (`user`, `assistant`,
/// `tool`), its `text`, the assistant's tool calls as JSON text, and the
/// tool call id a `tool` message answers.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct HostMessage {
    pub role: String,
    pub text: String,
    pub tool_calls_json: Option<String>,
    pub tool_call_id: Option<String>,
}

/// One request to a host model: the system prompt, the conversation,
/// the tools as a JSON array, the output JSON Schema, and the reply's
/// token cap. Built by troupekit, read by the host.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct HostRequest {
    pub system: Option<String>,
    pub messages: Vec<HostMessage>,
    pub tools_json: Option<String>,
    pub schema_json: Option<String>,
    pub max_tokens: u32,
}

/// A host model's reply: the whole `text` (also when it was streamed
/// through `TokenSink`) and the token counts the model reported.
/// Start from `HostReply::default()` and set the fields.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct HostReply {
    pub text: String,
    pub input_tokens: u64,
    pub output_tokens: u64,
}

const SECRET_HEADERS: [&str; 4] = [
    "authorization",
    "proxy-authorization",
    "x-api-key",
    "x-goog-api-key",
];

/// One HTTP header: name, value, and `sensitive`, set when the value is
/// a credential such as the vendor API key. `Debug` shows a sensitive
/// value as `[redacted]`; adapters must hide it in their text forms too.
#[non_exhaustive]
#[derive(Clone, Default, PartialEq, Eq)]
pub struct HostHeader {
    pub name: String,
    pub value: String,
    pub sensitive: bool,
}

impl HostHeader {
    /// Builds a header from `name` and `value`. `sensitive` is set for
    /// the credential headers `authorization`, `proxy-authorization`,
    /// `x-api-key` and `x-goog-api-key`, in any case. Output: the header.
    pub fn new(name: impl Into<String>, value: impl Into<String>) -> HostHeader {
        let name = name.into();
        let sensitive = is_secret_name(&name);
        HostHeader {
            name,
            value: value.into(),
            sensitive,
        }
    }

    fn hidden(&self) -> bool {
        self.sensitive || is_secret_name(&self.name)
    }
}

impl std::fmt::Debug for HostHeader {
    /// Writes the header with its value replaced by `[redacted]` when it
    /// is sensitive or named like a credential header, so logging a
    /// request never prints the API key.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let value: &dyn std::fmt::Debug = if self.hidden() {
            &"[redacted]"
        } else {
            &self.value
        };
        f.debug_struct("HostHeader")
            .field("name", &self.name)
            .field("value", value)
            .field("sensitive", &self.sensitive)
            .finish()
    }
}

fn is_secret_name(name: &str) -> bool {
    SECRET_HEADERS
        .iter()
        .any(|secret| name.eq_ignore_ascii_case(secret))
}

/// One HTTP request a host transport sends: method, absolute URL,
/// headers and body bytes, only for model and embedding calls to the
/// resolved vendor. The key header is `sensitive`; `Debug` redacts it.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct HostHttpRequest {
    pub method: String,
    pub url: String,
    pub headers: Vec<HostHeader>,
    pub body: Vec<u8>,
}

/// A failure inside host code, as a message. Foreign exceptions arrive
/// as this type from the adapters. A host tool's error reaches the model
/// as an error result; other host errors fail the call with `HostCallback`.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("{message}")]
pub struct HostError {
    pub message: String,
}

impl HostError {
    /// Builds a host error from `message`.
    /// Input: any text; it is shown to the developer and may reach the
    /// model for a tool error. Output: the error.
    pub fn new(message: impl Into<String>) -> HostError {
        HostError {
            message: message.into(),
        }
    }
}

/// Knowledge settings for `AgentBuilder::knowledge`: the document
/// `dir` to ingest at build, an optional index and store path, and
/// whether files that fail to load are allowed.
#[cfg(feature = "knowledge")]
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct KnowledgeConfig {
    pub dir: String,
    pub index: Option<String>,
    pub store: Option<String>,
    pub allow_failed: bool,
}

#[cfg(feature = "knowledge")]
impl KnowledgeConfig {
    /// Builds knowledge settings for the document directory `dir`.
    /// Other fields keep their defaults: no index or store path, and a
    /// file that fails to load fails the build.
    pub fn new(dir: impl Into<String>) -> KnowledgeConfig {
        KnowledgeConfig {
            dir: dir.into(),
            ..KnowledgeConfig::default()
        }
    }
}

/// Memory settings for `AgentBuilder::memory`: the `user` and `session`
/// the facts belong to, an optional SQLite path, and whether known facts
/// are put in the system prompt (`preamble`).
#[cfg(feature = "knowledge")]
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemoryConfig {
    pub user: String,
    pub session: String,
    pub sqlite: Option<String>,
    pub preamble: bool,
}

#[cfg(feature = "knowledge")]
impl MemoryConfig {
    /// Builds memory settings for `user` and `session`, both set by the
    /// host. Other fields keep the kit's defaults: the agent's own store
    /// and the preamble on.
    pub fn new(user: impl Into<String>, session: impl Into<String>) -> MemoryConfig {
        MemoryConfig {
            user: user.into(),
            session: session.into(),
            sqlite: None,
            preamble: true,
        }
    }
}
