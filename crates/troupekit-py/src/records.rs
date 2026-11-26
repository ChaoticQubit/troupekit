use pyo3::prelude::*;
use troupekit_ffi_core as neutral;

const UNMIRRORED: &str = "troupekit-py has no mirror for a new neutral variant";

/// Process-wide settings for `init`, mirrored from `RuntimeSettings`.
/// An argument left out keeps the neutral default; the neutral layer checks
/// the bounds of plan section 8 (S1 to S3).
#[pyclass(frozen, get_all, from_py_object, module = "troupekit")]
#[derive(Clone, Debug)]
pub struct RuntimeSettings {
    pub worker_threads: usize,
    pub max_input_bytes: usize,
    pub event_buffer: usize,
}

#[pymethods]
impl RuntimeSettings {
    #[new]
    #[pyo3(signature = (worker_threads=None, max_input_bytes=None, event_buffer=None))]
    fn new(
        worker_threads: Option<usize>,
        max_input_bytes: Option<usize>,
        event_buffer: Option<usize>,
    ) -> Self {
        let base = neutral::RuntimeSettings::default();
        RuntimeSettings {
            worker_threads: worker_threads.unwrap_or(base.worker_threads),
            max_input_bytes: max_input_bytes.unwrap_or(base.max_input_bytes),
            event_buffer: event_buffer.unwrap_or(base.event_buffer),
        }
    }
}

impl From<RuntimeSettings> for neutral::RuntimeSettings {
    fn from(value: RuntimeSettings) -> Self {
        let mut out = neutral::RuntimeSettings::default();
        out.worker_threads = value.worker_threads;
        out.max_input_bytes = value.max_input_bytes;
        out.event_buffer = value.event_buffer;
        out
    }
}

/// Token counts and cost of a step or a run, mirrored from `FfiUsage`.
/// Output only. Cost is in millionths of a US dollar.
/// Exported as `Usage` (freeze open question Q3).
#[pyclass(frozen, get_all, module = "troupekit")]
#[derive(Clone, Copy, Debug)]
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
#[pyclass(frozen, get_all, module = "troupekit")]
#[derive(Clone, Debug)]
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
#[pyclass(frozen, get_all, module = "troupekit")]
#[derive(Clone, Debug)]
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
/// `kind` is `"done"` with `text`, or `"pending"` with `run_id`, as the
/// freeze row names it for Python. Output only.
#[pyclass(frozen, get_all, module = "troupekit")]
#[derive(Clone, Debug)]
pub struct Progress {
    pub kind: &'static str,
    pub text: Option<String>,
    pub run_id: Option<String>,
}

impl From<neutral::Progress> for Progress {
    fn from(value: neutral::Progress) -> Self {
        match value {
            neutral::Progress::Done { text } => Progress {
                kind: "done",
                text: Some(text),
                run_id: None,
            },
            neutral::Progress::Pending { run_id } => Progress {
                kind: "pending",
                text: None,
                run_id: Some(run_id),
            },
            _ => unreachable!("{UNMIRRORED}: Progress"),
        }
    }
}

/// Why a run paused, mirrored from `FfiPause`.
/// `kind` is `"approval"` or `"question"` with `node`, or `"budget"` with
/// `limit`. Output only.
#[pyclass(frozen, get_all, module = "troupekit")]
#[derive(Clone, Debug)]
pub struct Pause {
    pub kind: &'static str,
    pub node: Option<String>,
    pub limit: Option<String>,
}

impl From<neutral::FfiPause> for Pause {
    fn from(value: neutral::FfiPause) -> Self {
        let (kind, node, limit) = match value {
            neutral::FfiPause::Approval { node } => ("approval", Some(node), None),
            neutral::FfiPause::Question { node } => ("question", Some(node), None),
            neutral::FfiPause::Budget { limit } => ("budget", None, Some(limit)),
            _ => unreachable!("{UNMIRRORED}: FfiPause"),
        };
        Pause { kind, node, limit }
    }
}

/// The values `Event.kind` takes, one class attribute per neutral kind.
/// Input: none. Output: string constants, for example `EventKind.TEXT_DELTA`
/// is `"text_delta"`. Exported as `EventKind` (freeze open question Q3).
#[pyclass(frozen, module = "troupekit")]
pub struct EventKind;

#[pymethods]
impl EventKind {
    #[classattr]
    const RUN_STARTED: &'static str = "run_started";
    #[classattr]
    const STEP_STARTED: &'static str = "step_started";
    #[classattr]
    const TEXT_DELTA: &'static str = "text_delta";
    #[classattr]
    const TOOL_CALL_STARTED: &'static str = "tool_call_started";
    #[classattr]
    const TOOL_CALL_FINISHED: &'static str = "tool_call_finished";
    #[classattr]
    const STEP_FINISHED: &'static str = "step_finished";
    #[classattr]
    const RUN_PAUSED: &'static str = "run_paused";
    #[classattr]
    const RUN_FINISHED: &'static str = "run_finished";
    #[classattr]
    const RUN_FAILED: &'static str = "run_failed";
}

/// One run event, mirrored from `FfiEvent` with its kind flattened.
/// `kind` is an `EventKind` value; only the fields of that kind are set,
/// the others are `None`. Output only.
#[pyclass(frozen, get_all, module = "troupekit")]
#[derive(Clone, Debug, Default)]
pub struct Event {
    pub run_id: String,
    pub seq: u64,
    pub at_ms: i64,
    pub kind: &'static str,
    pub workflow: Option<String>,
    pub step: Option<String>,
    pub persona: Option<String>,
    pub text: Option<String>,
    pub call_id: Option<String>,
    pub name: Option<String>,
    pub is_error: Option<bool>,
    pub usage: Option<Usage>,
    pub reason: Option<Pause>,
    pub error: Option<String>,
}

impl From<neutral::FfiEvent> for Event {
    fn from(value: neutral::FfiEvent) -> Self {
        let base = Event {
            run_id: value.run_id,
            seq: value.seq,
            at_ms: value.at_ms,
            ..Event::default()
        };
        flatten(base, value.kind)
    }
}

/// Fills the kind and the kind's own fields of one flattened event.
/// Input: the event with its common fields set, and the neutral kind.
/// Output: the event with `kind` and that kind's fields set.
fn flatten(base: Event, kind: neutral::FfiEventKind) -> Event {
    use neutral::FfiEventKind as K;
    match kind {
        K::RunStarted { workflow } => Event {
            kind: EventKind::RUN_STARTED,
            workflow: Some(workflow),
            ..base
        },
        K::StepStarted { step, persona } => Event {
            kind: EventKind::STEP_STARTED,
            step: Some(step),
            persona,
            ..base
        },
        K::TextDelta { text } => Event {
            kind: EventKind::TEXT_DELTA,
            text: Some(text),
            ..base
        },
        K::ToolCallStarted { call_id, name } => Event {
            kind: EventKind::TOOL_CALL_STARTED,
            call_id: Some(call_id),
            name: Some(name),
            ..base
        },
        K::ToolCallFinished { call_id, is_error } => Event {
            kind: EventKind::TOOL_CALL_FINISHED,
            call_id: Some(call_id),
            is_error: Some(is_error),
            ..base
        },
        K::StepFinished { step, usage } => Event {
            kind: EventKind::STEP_FINISHED,
            step: Some(step),
            usage: Some(usage.into()),
            ..base
        },
        K::RunPaused { reason } => Event {
            kind: EventKind::RUN_PAUSED,
            reason: Some(reason.into()),
            ..base
        },
        K::RunFinished { usage } => Event {
            kind: EventKind::RUN_FINISHED,
            usage: Some(usage.into()),
            ..base
        },
        K::RunFailed { error } => Event {
            kind: EventKind::RUN_FAILED,
            error: Some(error),
            ..base
        },
        _ => unreachable!("{UNMIRRORED}: FfiEventKind"),
    }
}

/// What a host model can do, mirrored from `HostModelCaps`.
/// An argument left out keeps the neutral default. Returned by a Python
/// `HostModel.capabilities()` and read by Rust once, at registration.
#[pyclass(frozen, get_all, from_py_object, module = "troupekit")]
#[derive(Clone, Copy, Debug)]
pub struct HostModelCaps {
    pub tools: bool,
    pub json_schema: bool,
    pub streaming: bool,
    pub context_tokens: u32,
}

#[pymethods]
impl HostModelCaps {
    #[new]
    #[pyo3(signature = (tools=None, json_schema=None, streaming=None, context_tokens=None))]
    fn new(
        tools: Option<bool>,
        json_schema: Option<bool>,
        streaming: Option<bool>,
        context_tokens: Option<u32>,
    ) -> Self {
        let base = neutral::HostModelCaps::default();
        HostModelCaps {
            tools: tools.unwrap_or(base.tools),
            json_schema: json_schema.unwrap_or(base.json_schema),
            streaming: streaming.unwrap_or(base.streaming),
            context_tokens: context_tokens.unwrap_or(base.context_tokens),
        }
    }
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
/// Passed to a Python `HostModel.generate`.
#[pyclass(frozen, get_all, module = "troupekit")]
#[derive(Clone, Debug)]
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
/// Passed to a Python `HostModel.generate`.
#[pyclass(frozen, get_all, module = "troupekit")]
#[derive(Clone, Debug)]
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
/// Rust parses and validates `text` and `tool_calls_json` (native tool
/// calls, for a model whose caps set `tools`); token counts feed usage.
/// Returned by a Python `HostModel.generate`.
#[pyclass(frozen, get_all, from_py_object, module = "troupekit")]
#[derive(Clone, Debug)]
pub struct HostReply {
    pub text: String,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub tool_calls_json: Option<String>,
}

#[pymethods]
impl HostReply {
    #[new]
    #[pyo3(signature = (text, input_tokens=0, output_tokens=0, tool_calls_json=None))]
    fn new(
        text: String,
        input_tokens: u64,
        output_tokens: u64,
        tool_calls_json: Option<String>,
    ) -> Self {
        HostReply {
            text,
            input_tokens,
            output_tokens,
            tool_calls_json,
        }
    }
}

impl From<HostReply> for neutral::HostReply {
    fn from(value: HostReply) -> Self {
        let mut out = neutral::HostReply::default();
        out.text = value.text;
        out.input_tokens = value.input_tokens;
        out.output_tokens = value.output_tokens;
        out.tool_calls_json = value.tool_calls_json;
        out
    }
}

/// One HTTP header, mirrored from `HostHeader`.
/// Used in both directions: request headers out, reply headers back.
/// `sensitive` marks a credential; `repr` and `str` show it as `[redacted]`.
#[pyclass(frozen, get_all, from_py_object, module = "troupekit")]
#[derive(Clone)]
pub struct HostHeader {
    pub name: String,
    pub value: String,
    pub sensitive: bool,
}

#[pymethods]
impl HostHeader {
    #[new]
    #[pyo3(signature = (name, value, sensitive = false))]
    fn new(name: String, value: String, sensitive: bool) -> Self {
        HostHeader {
            name,
            value,
            sensitive,
        }
    }

    /// The neutral layer's redacting text of this header.
    /// Input: none. Output: the text; a credential value is `[redacted]`.
    fn __repr__(&self) -> String {
        format!("{:?}", neutral::HostHeader::from(self.clone()))
    }

    /// Same text as `repr`, so `print(header)` never shows the key.
    /// Input: none. Output: the redacted text.
    fn __str__(&self) -> String {
        self.__repr__()
    }
}

impl From<neutral::HostHeader> for HostHeader {
    fn from(value: neutral::HostHeader) -> Self {
        HostHeader {
            name: value.name,
            value: value.value,
            sensitive: value.sensitive,
        }
    }
}

impl From<HostHeader> for neutral::HostHeader {
    /// Builds the neutral header; a header the host marked sensitive stays
    /// sensitive, and the neutral layer also flags credential header names.
    /// Input: the mirror. Output: the neutral header.
    fn from(value: HostHeader) -> Self {
        let mut header = neutral::HostHeader::new(value.name, value.value);
        header.sensitive = header.sensitive || value.sensitive;
        header
    }
}

/// An HTTP request for a host transport, mirrored from `HostHttpRequest`.
/// The neutral layer has already checked the URL against the vendor.
/// Passed to a Python `HostTransport.send`; `body` is `bytes`.
/// `repr` and `str` redact credential headers.
#[pyclass(frozen, get_all, module = "troupekit")]
#[derive(Clone)]
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

#[pymethods]
impl HostHttpRequest {
    /// The neutral layer's redacting text of this request.
    /// Input: none. Output: the text; credential header values are
    /// `[redacted]`.
    fn __repr__(&self) -> String {
        let mut request = neutral::HostHttpRequest::default();
        request.method = self.method.clone();
        request.url = self.url.clone();
        request.headers = self.headers.iter().cloned().map(Into::into).collect();
        request.body = self.body.clone();
        format!("{request:?}")
    }

    /// Same text as `repr`, so `print(request)` never shows the key.
    /// Input: none. Output: the redacted text.
    fn __str__(&self) -> String {
        self.__repr__()
    }
}

/// Knowledge settings for an agent, mirrored from `KnowledgeConfig`.
/// `dir` is the folder to ingest; `index` and `store` are optional paths.
/// Input to `AgentBuilder.knowledge`.
#[cfg(feature = "knowledge")]
#[pyclass(frozen, get_all, from_py_object, module = "troupekit")]
#[derive(Clone, Debug)]
pub struct KnowledgeConfig {
    pub dir: String,
    pub index: Option<String>,
    pub store: Option<String>,
    pub allow_failed: bool,
}

#[cfg(feature = "knowledge")]
#[pymethods]
impl KnowledgeConfig {
    #[new]
    #[pyo3(signature = (dir, index=None, store=None, allow_failed=None))]
    fn new(
        dir: String,
        index: Option<String>,
        store: Option<String>,
        allow_failed: Option<bool>,
    ) -> Self {
        let base = neutral::KnowledgeConfig::new(dir);
        KnowledgeConfig {
            dir: base.dir,
            index: index.or(base.index),
            store: store.or(base.store),
            allow_failed: allow_failed.unwrap_or(base.allow_failed),
        }
    }
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
/// `user` and `session` are set by the host and keep users apart; an
/// argument left out keeps the neutral default. Input to `AgentBuilder.memory`.
#[cfg(feature = "knowledge")]
#[pyclass(frozen, get_all, from_py_object, module = "troupekit")]
#[derive(Clone, Debug)]
pub struct MemoryConfig {
    pub user: String,
    pub session: String,
    pub sqlite: Option<String>,
    pub preamble: bool,
}

#[cfg(feature = "knowledge")]
#[pymethods]
impl MemoryConfig {
    #[new]
    #[pyo3(signature = (user, session, sqlite=None, preamble=None))]
    fn new(user: String, session: String, sqlite: Option<String>, preamble: Option<bool>) -> Self {
        let base = neutral::MemoryConfig::new(user, session);
        MemoryConfig {
            user: base.user,
            session: base.session,
            sqlite: sqlite.or(base.sqlite),
            preamble: preamble.unwrap_or(base.preamble),
        }
    }
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
