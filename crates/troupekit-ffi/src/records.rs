use troupekit_ffi_core as neutral;

const UNMIRRORED: &str = "troupekit-ffi has no mirror for a new neutral variant";

/// Process-wide settings for `initialize`, mirrored from `RuntimeSettings`.
/// The neutral layer checks the bounds of plan section 8 (S1 to S3).
/// Input: worker threads, max caller string bytes, event buffer size.
#[derive(Clone, Debug, PartialEq, Eq, uniffi::Record)]
pub struct RuntimeSettings {
    pub worker_threads: u32,
    pub max_input_bytes: u64,
    pub event_buffer: u32,
}

impl From<RuntimeSettings> for neutral::RuntimeSettings {
    fn from(value: RuntimeSettings) -> Self {
        let mut out = neutral::RuntimeSettings::default();
        out.worker_threads = usize::try_from(value.worker_threads).unwrap_or(usize::MAX);
        out.max_input_bytes = usize::try_from(value.max_input_bytes).unwrap_or(usize::MAX);
        out.event_buffer = usize::try_from(value.event_buffer).unwrap_or(usize::MAX);
        out
    }
}

/// Token counts and cost of a step or a run, mirrored from `FfiUsage`.
/// Output only: built from the neutral value, never read back.
/// Cost is in millionths of a US dollar.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, uniffi::Record)]
pub struct Usage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cost_usd_micros: u64,
}

impl From<neutral::FfiUsage> for Usage {
    fn from(value: neutral::FfiUsage) -> Self {
        Usage {
            input_tokens: value.input_tokens,
            output_tokens: value.output_tokens,
            cost_usd_micros: value.cost_usd_micros,
        }
    }
}

/// Result of one agent run, mirrored from `RunResult`.
/// `output_json` holds the structured output when the agent has a schema.
/// Output only.
#[derive(Clone, Debug, PartialEq, Eq, uniffi::Record)]
pub struct RunResult {
    pub text: String,
    pub output_json: String,
    pub usage: Usage,
    pub model_calls: u32,
    pub run_id: String,
}

impl From<neutral::RunResult> for RunResult {
    fn from(value: neutral::RunResult) -> Self {
        RunResult {
            text: value.text,
            output_json: value.output_json,
            usage: value.usage.into(),
            model_calls: value.model_calls,
            run_id: value.run_id,
        }
    }
}

/// Result of one graph run, mirrored from `GraphResult`.
/// Output and state are JSON text, as the neutral layer renders them.
/// Output only.
#[derive(Clone, Debug, PartialEq, Eq, uniffi::Record)]
pub struct GraphResult {
    pub output_json: String,
    pub state_json: String,
    pub usage: Usage,
    pub steps: u64,
    pub run_id: String,
}

impl From<neutral::GraphResult> for GraphResult {
    fn from(value: neutral::GraphResult) -> Self {
        GraphResult {
            output_json: value.output_json,
            state_json: value.state_json,
            usage: value.usage.into(),
            steps: value.steps,
            run_id: value.run_id,
        }
    }
}

/// Outcome of one background window, mirrored from `Progress`.
/// `Done` carries the answer; `Pending` the run id to resume next window.
/// Output only.
#[derive(Clone, Debug, PartialEq, Eq, uniffi::Enum)]
pub enum Progress {
    Done { text: String },
    Pending { run_id: String },
}

impl From<neutral::Progress> for Progress {
    fn from(value: neutral::Progress) -> Self {
        match value {
            neutral::Progress::Done { text } => Progress::Done { text },
            neutral::Progress::Pending { run_id } => Progress::Pending { run_id },
            _ => unreachable!("{UNMIRRORED}: Progress"),
        }
    }
}

/// Why a run paused, mirrored from `FfiPause`.
/// `node` names the graph node; `limit` names the budget that ran out.
/// Output only.
#[derive(Clone, Debug, PartialEq, Eq, uniffi::Enum)]
pub enum Pause {
    Approval { node: String },
    Question { node: String },
    Budget { limit: String },
}

impl From<neutral::FfiPause> for Pause {
    fn from(value: neutral::FfiPause) -> Self {
        match value {
            neutral::FfiPause::Approval { node } => Pause::Approval { node },
            neutral::FfiPause::Question { node } => Pause::Question { node },
            neutral::FfiPause::Budget { limit } => Pause::Budget { limit },
            _ => unreachable!("{UNMIRRORED}: FfiPause"),
        }
    }
}

/// What happened in one run event, mirrored from `FfiEventKind`.
/// Lifecycle kinds are never dropped; `TextDelta` may be when the buffer
/// is full (`RunStream::dropped_deltas`). Output only.
#[derive(Clone, Debug, PartialEq, Eq, uniffi::Enum)]
pub enum EventKind {
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
        usage: Usage,
    },
    RunPaused {
        reason: Pause,
    },
    RunFinished {
        usage: Usage,
    },
    RunFailed {
        error: String,
    },
}

impl From<neutral::FfiEventKind> for EventKind {
    fn from(value: neutral::FfiEventKind) -> Self {
        use neutral::FfiEventKind as K;
        match value {
            K::RunStarted { workflow } => EventKind::RunStarted { workflow },
            K::StepStarted { step, persona } => EventKind::StepStarted { step, persona },
            K::TextDelta { text } => EventKind::TextDelta { text },
            K::ToolCallStarted { call_id, name } => EventKind::ToolCallStarted { call_id, name },
            K::ToolCallFinished { call_id, is_error } => {
                EventKind::ToolCallFinished { call_id, is_error }
            }
            K::StepFinished { step, usage } => EventKind::StepFinished {
                step,
                usage: usage.into(),
            },
            K::RunPaused { reason } => EventKind::RunPaused {
                reason: reason.into(),
            },
            K::RunFinished { usage } => EventKind::RunFinished {
                usage: usage.into(),
            },
            K::RunFailed { error } => EventKind::RunFailed { error },
            _ => unreachable!("{UNMIRRORED}: FfiEventKind"),
        }
    }
}

/// One run event, mirrored from `FfiEvent`.
/// `seq` grows by one per event of a run; `at_ms` is Unix time in ms.
/// Output only.
#[derive(Clone, Debug, PartialEq, Eq, uniffi::Record)]
pub struct Event {
    pub run_id: String,
    pub seq: u64,
    pub at_ms: i64,
    pub kind: EventKind,
}

impl From<neutral::FfiEvent> for Event {
    fn from(value: neutral::FfiEvent) -> Self {
        Event {
            run_id: value.run_id,
            seq: value.seq,
            at_ms: value.at_ms,
            kind: value.kind.into(),
        }
    }
}

/// What a host model can do, mirrored from `HostModelCaps`.
/// Returned by the foreign `HostModel::capabilities` and read by Rust.
/// `context_tokens` is the model's context window.
#[derive(Clone, Copy, Debug, PartialEq, Eq, uniffi::Record)]
pub struct HostModelCaps {
    pub tools: bool,
    pub json_schema: bool,
    pub streaming: bool,
    pub context_tokens: u32,
}

impl From<HostModelCaps> for neutral::HostModelCaps {
    fn from(value: HostModelCaps) -> Self {
        let mut out = neutral::HostModelCaps::default();
        out.tools = value.tools;
        out.json_schema = value.json_schema;
        out.streaming = value.streaming;
        out.context_tokens = value.context_tokens;
        out
    }
}

/// One message of a host model request, mirrored from `HostMessage`.
/// Tool calls and the tool call id are JSON and text the Rust side rendered.
/// Input to the foreign `HostModel::generate`.
#[derive(Clone, Debug, PartialEq, Eq, uniffi::Record)]
pub struct HostMessage {
    pub role: String,
    pub text: String,
    pub tool_calls_json: Option<String>,
    pub tool_call_id: Option<String>,
}

impl From<neutral::HostMessage> for HostMessage {
    fn from(value: neutral::HostMessage) -> Self {
        HostMessage {
            role: value.role,
            text: value.text,
            tool_calls_json: value.tool_calls_json,
            tool_call_id: value.tool_call_id,
        }
    }
}

/// A request to a host model, mirrored from `HostRequest`.
/// Rust renders every field; the host passes them to its model as given.
/// Input to the foreign `HostModel::generate`.
#[derive(Clone, Debug, PartialEq, Eq, uniffi::Record)]
pub struct HostRequest {
    pub system: Option<String>,
    pub messages: Vec<HostMessage>,
    pub tools_json: Option<String>,
    pub schema_json: Option<String>,
    pub max_tokens: u32,
}

impl From<neutral::HostRequest> for HostRequest {
    fn from(value: neutral::HostRequest) -> Self {
        HostRequest {
            system: value.system,
            messages: value.messages.into_iter().map(Into::into).collect(),
            tools_json: value.tools_json,
            schema_json: value.schema_json,
            max_tokens: value.max_tokens,
        }
    }
}

/// A host model's whole reply, mirrored from `HostReply`.
/// Rust parses and validates `text`; token counts feed the run's usage.
/// Returned by the foreign `HostModel::generate`.
#[derive(Clone, Debug, PartialEq, Eq, uniffi::Record)]
pub struct HostReply {
    pub text: String,
    pub input_tokens: u64,
    pub output_tokens: u64,
}

impl From<HostReply> for neutral::HostReply {
    fn from(value: HostReply) -> Self {
        let mut out = neutral::HostReply::default();
        out.text = value.text;
        out.input_tokens = value.input_tokens;
        out.output_tokens = value.output_tokens;
        out
    }
}

/// One HTTP header, mirrored from `HostHeader`.
/// Used in both directions: request headers out, reply headers back.
/// Names and values are passed through unchanged.
#[derive(Clone, Debug, PartialEq, Eq, uniffi::Record)]
pub struct HostHeader {
    pub name: String,
    pub value: String,
}

impl From<neutral::HostHeader> for HostHeader {
    fn from(value: neutral::HostHeader) -> Self {
        HostHeader {
            name: value.name,
            value: value.value,
        }
    }
}

impl From<HostHeader> for neutral::HostHeader {
    fn from(value: HostHeader) -> Self {
        neutral::HostHeader::new(value.name, value.value)
    }
}

/// An HTTP request for a host transport, mirrored from `HostHttpRequest`.
/// The neutral layer has already checked the URL against the vendor.
/// Input to the foreign `HostTransport::send`.
#[derive(Clone, Debug, PartialEq, Eq, uniffi::Record)]
pub struct HostHttpRequest {
    pub method: String,
    pub url: String,
    pub headers: Vec<HostHeader>,
    pub body: Vec<u8>,
}

impl From<neutral::HostHttpRequest> for HostHttpRequest {
    fn from(value: neutral::HostHttpRequest) -> Self {
        HostHttpRequest {
            method: value.method,
            url: value.url,
            headers: value.headers.into_iter().map(Into::into).collect(),
            body: value.body,
        }
    }
}

/// Knowledge settings for an agent, mirrored from `KnowledgeConfig`.
/// `dir` is the folder to ingest; `index` and `store` are optional paths.
/// Input to `AgentBuilder::knowledge`.
#[cfg(feature = "knowledge")]
#[derive(Clone, Debug, PartialEq, Eq, uniffi::Record)]
pub struct KnowledgeConfig {
    pub dir: String,
    pub index: Option<String>,
    pub store: Option<String>,
    pub allow_failed: bool,
}

#[cfg(feature = "knowledge")]
impl From<KnowledgeConfig> for neutral::KnowledgeConfig {
    fn from(value: KnowledgeConfig) -> Self {
        let mut out = neutral::KnowledgeConfig::new(value.dir);
        out.index = value.index;
        out.store = value.store;
        out.allow_failed = value.allow_failed;
        out
    }
}

/// Memory settings for an agent, mirrored from `MemoryConfig`.
/// `user` and `session` are set by the host and keep users apart.
/// Input to `AgentBuilder::memory`.
#[cfg(feature = "knowledge")]
#[derive(Clone, Debug, PartialEq, Eq, uniffi::Record)]
pub struct MemoryConfig {
    pub user: String,
    pub session: String,
    pub sqlite: Option<String>,
    pub preamble: bool,
}

#[cfg(feature = "knowledge")]
impl From<MemoryConfig> for neutral::MemoryConfig {
    fn from(value: MemoryConfig) -> Self {
        let mut out = neutral::MemoryConfig::new(value.user, value.session);
        out.sqlite = value.sqlite;
        out.preamble = value.preamble;
        out
    }
}
