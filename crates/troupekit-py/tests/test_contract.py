import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys
import textwrap
import threading

import pytest

import troupekit
from troupekit import _troupekit

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import pending
import stub_server

CRATES = pathlib.Path(__file__).resolve().parents[2]
CASES = CRATES / "troupekit-ffi-core" / "contract" / "cases.txt"
FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "agent.toml"
SELECTOR = f"{stub_server.VENDOR_ID}/{stub_server.MODEL_ID}"
ANSWER = ["Hello", " from", " the stub"]
RUN_SECONDS_MAX = 30
EVENTS_MAX = 10_000
EMBED_DIMS = 64
CONCURRENT_RUNS = 8
SECRET = "sk-contract-secret-value-0001"
MISSING_ENV = "TROUPEKIT_CONTRACT_KEY_THAT_IS_NOT_SET"
CODES = {
    "Config": 1,
    "MissingKey": 2,
    "Catalog": 3,
    "Persona": 4,
    "Agent": 5,
    "Tool": 6,
    "Workflow": 7,
    "Graph": 8,
    "Store": 9,
    "Context": 10,
    "Failed": 11,
    "Paused": 12,
    "Retries": 13,
    "Cancelled": 14,
    "HostCallback": 15,
    "InvalidArgument": 16,
    "Unsupported": 17,
    "Panic": 18,
    "Internal": 99,
}


class ScriptedModel(troupekit.HostModel):
    """A host model that answers from a script and records every request.

    Input: the replies, one per `generate` call; the last one repeats.
    Output: `requests` holds the rendered text of each request.
    """

    def __init__(self, replies):
        super().__init__()
        if not replies:
            raise ValueError("the script needs at least one reply")
        self.replies = list(replies)
        self.requests = []
        self.lock = threading.Lock()

    def id(self):
        """Returns the model id.

        Input: none. Output: a fixed id.
        """
        return "scripted"

    def capabilities(self):
        """Declares no native tools, so Rust emulates tool calls in text.

        Input: none. Output: `HostModelCaps` with every flag off.
        """
        return troupekit.HostModelCaps(tools=False, json_schema=False, streaming=True)

    def generate(self, request, sink):
        """Records the request, streams the scripted reply and returns it.

        Input: the `HostRequest` and its `TokenSink`. Output: a `HostReply`.
        """
        with self.lock:
            turn = len(self.requests)
            self.requests.append(render(request))
            text = self.replies[min(turn, len(self.replies) - 1)]
        sink.delta(text)
        return troupekit.HostReply(text, input_tokens=3, output_tokens=1)


class EmbedderCounter(troupekit.HostEmbedder):
    """A deterministic host embedder: a hashed bag of lowercase words.

    Input: none. Output: `calls` counts `embed` calls; equal word sets give
    equal vectors, so a query finds the document that holds its words.
    """

    def __init__(self):
        super().__init__()
        self.calls = 0

    def dims(self):
        """Returns the vector size.

        Input: none. Output: EMBED_DIMS.
        """
        return EMBED_DIMS

    def embed(self, texts):
        """Embeds each text.

        Input: a list of str. Output: one vector per text.
        """
        self.calls += 1
        return [bag_of_words(text) for text in texts]


class RecordingTool(troupekit.HostTool):
    """A host tool that records its calls and the threads they ran on.

    Input: what `call` does: echo `text`, raise `error`, or return `output`.
    Output: `calls` holds `(args, thread_id)` per call.
    """

    def __init__(self, name="echo", error=None, output=None):
        super().__init__()
        self.tool_name = name
        self.error = error
        self.output = output
        self.calls = []

    def name(self):
        """Returns the tool name.

        Input: none. Output: the name.
        """
        return self.tool_name

    def description(self):
        """Returns the description.

        Input: none. Output: one sentence.
        """
        return "Repeats the text it is given."

    def schema_json(self):
        """Returns the argument schema.

        Input: none. Output: JSON object text.
        """
        return json.dumps({"type": "object", "properties": {"text": {"type": "string"}}})

    def call(self, args_json):
        """Records the call, then echoes, raises or returns the set output.

        Input: the arguments as JSON text. Output: the tool output text.
        """
        args = json.loads(args_json)
        self.calls.append((args, threading.get_ident()))
        if self.error is not None:
            raise self.error
        if self.output is not None:
            return self.output
        return args.get("text", "")


class ChunkTransport(troupekit.HostTransport):
    """A host transport that answers every request from a fixed script.

    Input: the head URL to report (None: the request URL) and the body
    chunks. Output: `requests` holds each `HostHttpRequest` it got.
    """

    def __init__(self, chunks, head_url=None):
        super().__init__()
        self.chunks = list(chunks)
        self.head_url = head_url
        self.requests = []

    def send(self, request, sink):
        """Sends the head, then each chunk.

        Input: request and sink. Output: None.
        """
        self.requests.append(request)
        url = self.head_url if self.head_url is not None else request.url
        sink.head(200, url, [troupekit.HostHeader("content-type", "text/event-stream")])
        for chunk in self.chunks:
            sink.chunk(chunk)


class Collector(troupekit.EventListener):
    """An event listener that stores the kind of every event it gets.

    Input: none. Output: `kinds` in arrival order; `done` is set once a run
    ends with `run_finished`, `run_failed` or `run_paused`.
    """

    def __init__(self):
        super().__init__()
        self.kinds = []
        self.done = threading.Event()

    def on_event(self, event):
        """Stores one event kind.

        Input: the event. Output: None.
        """
        self.kinds.append(event.kind)
        if event.kind in ("run_finished", "run_failed", "run_paused"):
            self.done.set()


def render(request):
    """Joins every text field of a host request for substring checks.

    Input: a `HostRequest`. Output: one str holding the system text, the
    tool list and every message text, tool call and tool call id.
    """
    parts = [request.system or "", request.tools_json or ""]
    for message in request.messages:
        parts += [message.role, message.text]
        parts += [message.tool_calls_json or "", message.tool_call_id or ""]
    return "\n".join(parts)


def bag_of_words(text):
    """Hashes the lowercase words of `text` into a unit vector.

    Input: a str. Output: a list of EMBED_DIMS floats; all zeros but one
    when the text has no word, so the vector is never all zero.
    """
    vector = [0.0] * EMBED_DIMS
    for word in re.findall(r"[a-z0-9]+", text.lower()):
        digest = hashlib.sha256(word.encode()).digest()
        vector[digest[0] % EMBED_DIMS] += 1.0
    vector[0] += 1e-3
    norm = sum(v * v for v in vector) ** 0.5
    return [v / norm for v in vector]


def tool_reply(name, arguments):
    """Renders an emulated tool call as the host model writes it (K8).

    Input: the tool name and its arguments dict. Output: the JSON text.
    """
    return json.dumps({"tool": name, "arguments": arguments})


def answer_reply(text):
    """Renders an emulated final answer as the host model writes it (K8).

    Input: the answer text. Output: the JSON text.
    """
    return json.dumps({"answer": text})


def host_agent(model, *tools, sqlite=None):
    """Builds an agent on a host model with the given host tools.

    Input: the model, tools, and an optional SQLite path. Output: the
    unbuilt `AgentBuilder`.
    """
    builder = troupekit.AgentBuilder("helper").instructions("Help the user.")
    builder = builder.host_model(model)
    for tool in tools:
        builder = builder.tool(tool)
    if sqlite is not None:
        builder = builder.sqlite(str(sqlite))
    return builder


def cloud_agent(base_url, sqlite=None):
    """Builds an agent on the stub vendor through the cloud path.

    Input: the stub base URL and an optional SQLite path. Output: the
    unbuilt `AgentBuilder`.
    """
    builder = (
        troupekit.AgentBuilder("helper")
        .instructions("Help the user.")
        .model(SELECTOR)
        .catalog_toml(stub_server.catalog_for(base_url))
    )
    if sqlite is not None:
        builder = builder.sqlite(str(sqlite))
    return builder


def with_deadline(call):
    """Runs `call()` on a daemon thread and waits RUN_SECONDS_MAX for it.

    Input: a function of no arguments. Output: its result, or its exception
    re-raised; fails the test when it does not return in time.
    """
    outcome = {}

    def target():
        """Stores the result or the exception.

        Input: none. Output: None.
        """
        try:
            outcome["value"] = call()
        except BaseException as error:
            outcome["error"] = error

    worker = threading.Thread(target=target, daemon=True)
    worker.start()
    worker.join(RUN_SECONDS_MAX)
    if worker.is_alive():
        pytest.fail(f"call did not return within {RUN_SECONDS_MAX} s")
    if "error" in outcome:
        raise outcome["error"]
    return outcome["value"]


def pull_all(stream):
    """Pulls every event of a stream until it ends.

    Input: a `RunStream`. Output: the list of events; fails the test past
    EVENTS_MAX events.
    """
    events = []
    for _ in range(EVENTS_MAX):
        event = stream.next()
        if event is None:
            return events
        events.append(event)
    pytest.fail("stream did not end within EVENTS_MAX events")


def code_of(excinfo):
    """Returns the numeric code of a raised `TroupekitError`.

    Input: the pytest `ExceptionInfo`. Output: the int code. A code of
    `Unsupported` skips the case: the neutral call it reached is a stub,
    and `pytest.raises` has already caught what `needs_neutral` would.
    """
    code = excinfo.value.code
    if code == pending.UNSUPPORTED:
        pytest.skip(f"not built in troupekit-ffi-core yet: {excinfo.value.message}")
    return code


@pytest.fixture
def stub():
    """Runs one stub vendor that answers ANSWER; stops it afterwards.

    Input: none. Output: (state, base_url).
    """
    server, state, base_url = stub_server.start(ANSWER)
    yield state, base_url
    server.shutdown()


def test_version_matches_core():
    """The package and the native module report the same core version.

    Input: none. Output: passes when both are equal semantic versions.
    """
    assert re.fullmatch(r"\d+\.\d+\.\d+", troupekit.version())
    assert troupekit.version() == _troupekit.version()


@pending.needs_neutral
def test_missing_model_is_config_error():
    """An agent with no model and no host model is refused at build.

    Input: a builder with instructions only. Output: `TroupekitError` with
    code `Config`.
    """
    with pytest.raises(troupekit.TroupekitError) as excinfo:
        troupekit.AgentBuilder("helper").instructions("Help.").build()
    assert code_of(excinfo) == CODES["Config"]


@pending.needs_neutral
def test_missing_key_names_env_not_value(stub):
    """A missing key names the environment variable, never a key value.

    Input: a keyed vendor and an unset variable, then a set key that fails.
    Output: `MissingKey` naming the variable; no error text holds the key.
    """
    _, base_url = stub
    keyed = stub_server.catalog_for(base_url).replace('auth = "none"', 'auth = "bearer"')
    keyed = keyed.replace(base_url, "https://127.0.0.1:9/v1")
    os.environ.pop(MISSING_ENV, None)

    def keyed_agent():
        """Starts a fresh keyed builder; `build` consumes a builder (K6).

        Input: none. Output: an unbuilt `AgentBuilder` on the keyed vendor.
        """
        builder = troupekit.AgentBuilder("helper").instructions("Help.").model(SELECTOR)
        return builder.catalog_toml(keyed)

    with pytest.raises(troupekit.TroupekitError) as excinfo:
        keyed_agent().api_key_env(MISSING_ENV).build().run("hi")
    assert code_of(excinfo) == CODES["MissingKey"]
    assert MISSING_ENV in excinfo.value.message
    with pytest.raises(troupekit.TroupekitError) as excinfo:
        keyed_agent().api_key(SECRET).build().run("hi")
    assert SECRET not in excinfo.value.message
    assert SECRET not in str(excinfo.value)


@pending.needs_neutral
def test_host_model_run_returns_text():
    """A run on a host model returns the model's text.

    Input: a host model scripted to say one sentence. Output: passes when
    the answer is that sentence and the model was called once.
    """
    model = ScriptedModel(["Hello from the host model"])
    assert host_agent(model).build().run("hi") == "Hello from the host model"
    assert len(model.requests) == 1


@pending.needs_neutral
def test_host_tool_called_with_json_args():
    """A host tool receives the arguments the model asked for, as JSON.

    Input: a model that calls `echo` once, then answers. Output: passes when
    the tool ran once with those arguments and the answer arrives.
    """
    model = ScriptedModel([tool_reply("echo", {"text": "hi"}), answer_reply("done")])
    tool = RecordingTool()
    assert host_agent(model, tool).build().run("echo hi") == "done"
    assert [args for args, _ in tool.calls] == [{"text": "hi"}]
    assert len(model.requests) == 2


@pending.needs_neutral
def test_host_tool_error_reaches_model():
    """A `HostError` from a tool reaches the model as the tool's output.

    Input: a tool that raises `HostError`. Output: passes when the second
    model request holds the error text and the run still answers.
    """
    model = ScriptedModel([tool_reply("echo", {"text": "hi"}), answer_reply("recovered")])
    tool = RecordingTool(error=troupekit.HostError("echo is offline"))
    assert host_agent(model, tool).build().run("echo hi") == "recovered"
    assert "echo is offline" in model.requests[1]


@pending.needs_neutral
def test_host_tool_throw_does_not_crash():
    """Any Python exception in a tool is contained and the run goes on.

    Input: a tool that raises `ZeroDivisionError`. Output: passes when the
    run answers and the model saw the exception type.
    """
    model = ScriptedModel([tool_reply("echo", {"text": "hi"}), answer_reply("still here")])
    tool = RecordingTool(error=ZeroDivisionError("division by zero"))
    assert host_agent(model, tool).build().run("echo hi") == "still here"
    assert "ZeroDivisionError" in model.requests[1]


@pending.needs_neutral
def test_stream_yields_deltas_then_finished(stub):
    """A stream yields the text deltas, then `run_finished` last.

    Input: the stub vendor streaming ANSWER. Output: passes when the deltas
    join to ANSWER, the first event is `run_started` and the last
    `run_finished`, and nothing was dropped.
    """
    _, base_url = stub
    stream = cloud_agent(base_url).build().stream("hi")
    events = pull_all(stream)
    deltas = [e.text for e in events if e.kind == troupekit.EventKind.TEXT_DELTA]
    assert "".join(deltas) == "".join(ANSWER)
    assert events[0].kind == troupekit.EventKind.RUN_STARTED
    assert events[-1].kind == troupekit.EventKind.RUN_FINISHED
    assert stream.dropped_deltas() == 0


@pending.needs_neutral
def test_listener_sees_same_events_as_pull():
    """A listener gets the same kinds, in order, as pulling the same run.

    Input: two runs of the same scripted host model. Output: passes when the
    pulled kinds equal the listener's kinds.
    """
    first = host_agent(ScriptedModel(["same"])).build().stream("hi")
    pulled = [event.kind for event in pull_all(first)]
    collector = Collector()
    stream = host_agent(ScriptedModel(["same"])).build().stream("hi")
    stream.set_listener(collector)
    stream.result()
    assert collector.done.wait(RUN_SECONDS_MAX)
    assert collector.kinds == pulled


@pending.needs_neutral
def test_start_returns_run_id_before_model_call(stub, tmp_path):
    """`start` returns a run id and makes no model call.

    Input: the stub vendor and a SQLite file. Output: passes when the id is
    not empty and the stub saw no request.
    """
    state, base_url = stub
    run_id = cloud_agent(base_url, tmp_path / "runs.db").build().start("hi")
    assert run_id
    assert state.count() == 0


@pending.needs_neutral
def test_resume_after_handles_dropped(stub, tmp_path):
    """A run started by one handle is finished by a fresh one.

    Input: `start` on one handle, which is then dropped; `resume` on a new
    handle on the same SQLite file. Output: passes on ANSWER and one request.
    """
    state, base_url = stub
    database = tmp_path / "runs.db"
    first = cloud_agent(base_url, database).build()
    run_id = first.start("hi")
    del first
    assert cloud_agent(base_url, database).build().resume(run_id) == "".join(ANSWER)
    assert state.count() == 1


@pending.needs_neutral
def test_resume_for_zero_budget_is_pending(stub, tmp_path):
    """A zero budget window does no step and reports the run as pending.

    Input: a started run and `resume_for(id, 0)`. Output: passes when the
    progress is `pending` with the same id and the stub saw no request.
    """
    state, base_url = stub
    agent = cloud_agent(base_url, tmp_path / "runs.db").build()
    run_id = agent.start("hi")
    progress = agent.resume_for(run_id, 0)
    assert progress.kind == "pending"
    assert progress.run_id == run_id
    assert state.count() == 0


@pending.needs_neutral
def test_cancel_keeps_checkpoint_and_resumes(stub, tmp_path):
    """A cancelled streamed run keeps its checkpoint and can be resumed.

    Input: a stream cancelled at once, then `resume` of its run id. Output:
    passes when the stream ends `Cancelled` and the resume answers ANSWER.
    """
    _, base_url = stub
    agent = cloud_agent(base_url, tmp_path / "runs.db").build()
    stream = agent.stream("hi")
    stream.cancel()
    with pytest.raises(troupekit.TroupekitError) as excinfo:
        stream.result()
    assert code_of(excinfo) == CODES["Cancelled"]
    assert agent.resume(stream.run_id()) == "".join(ANSWER)


@pending.needs_neutral
def test_toml_agent_runs(stub):
    """A persona TOML agent runs against the stub vendor.

    Input: `tests/fixtures/agent.toml`. Output: passes on ANSWER.
    """
    _, base_url = stub
    text = FIXTURE.read_text(encoding="utf-8")
    catalog = stub_server.catalog_for(base_url)
    agent = troupekit.AgentBuilder.from_toml(text).catalog_toml(catalog)
    assert agent.build().run("hi") == "".join(ANSWER)


@pending.needs_neutral
def test_toml_over_cap_refused():
    """A persona TOML over 1 MiB is refused before it is parsed.

    Input: the fixture padded past 1 MiB. Output: `Config` from the kit
    (`MAX_PERSONA_BYTES`) or `InvalidArgument` from the caller string cap.
    """
    text = FIXTURE.read_text(encoding="utf-8") + "\n" + "a" * (1024 * 1024)
    with pytest.raises(troupekit.TroupekitError) as excinfo:
        troupekit.AgentBuilder.from_toml(text).build()
    assert code_of(excinfo) in (CODES["Config"], CODES["InvalidArgument"])
    assert "MiB" in excinfo.value.message


@pending.needs_neutral
def test_graph_agent_then_step_runs():
    """A graph runs an agent node, then a host tool step after it.

    Input: agent `writer` on a host model, step `stamp`, edge writer to
    stamp. Output: passes when the step ran once and its output is in the
    graph output.
    """
    writer = host_agent(ScriptedModel(["draft"]))
    stamp = RecordingTool(name="stamp", output="stamped")
    graph = troupekit.GraphBuilder("flow").agent("writer", writer).step("stamp", stamp)
    result = graph.edge("writer", "stamp").build().run_result("go")
    assert len(stamp.calls) == 1
    assert "stamped" in result.output_json


@pending.needs_neutral
def test_knowledge_with_host_embedder_finds_doc(tmp_path):
    """Knowledge with a host embedder finds the document a query names.

    Input: one document, a hashed-words embedder, and a model that searches
    once. Output: passes when the embedder ran and the search result the
    model saw holds the document text.
    """
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "fruit.md").write_text("The secret fruit is pineapple.\n", encoding="utf-8")
    embedder = EmbedderCounter()
    search = tool_reply("search_knowledge", {"query": "secret fruit pineapple"})
    model = ScriptedModel([search, answer_reply("pineapple")])
    config = troupekit.KnowledgeConfig(str(docs))
    agent = host_agent(model, sqlite=tmp_path / "kb.db").knowledge(config, embedder)
    assert agent.build().run("What is the secret fruit?") == "pineapple"
    assert embedder.calls > 0
    assert "The secret fruit is pineapple" in model.requests[1]


@pending.needs_neutral
def test_memory_facts_persist_between_runs(tmp_path):
    """A fact stored in one run is in the next run's request.

    Input: run 1 calls `remember_fact`; run 2 on a new handle with the same
    user, session and SQLite file. Output: passes when run 2's first model
    request holds the fact.
    """
    database = str(tmp_path / "memory.db")
    config = troupekit.MemoryConfig("user-1", "session-1", sqlite=database)
    remember = tool_reply("remember_fact", {"key": "name", "value": "Ada Lovelace"})
    first = ScriptedModel([remember, answer_reply("noted")])
    assert host_agent(first).memory(config).build().run("My name is Ada Lovelace") == "noted"
    second = ScriptedModel(["Hello Ada"])
    assert host_agent(second).memory(config).build().run("Who am I?") == "Hello Ada"
    assert "Ada Lovelace" in second.requests[0]


def test_error_codes_are_stable():
    """Every error code keeps its fixed number.

    Input: the table of plan section 6.1. Output: passes when each
    `ErrorCode` member converts to its number.
    """
    for name, number in CODES.items():
        assert int(getattr(troupekit.ErrorCode, name)) == number


@pending.needs_neutral
def test_laws_path_with_dotdot_refused(tmp_path):
    """A laws path with `..` is refused at build.

    Input: a host model agent with `laws("../laws.bend")`. Output:
    `TroupekitError` with code `Config` or `InvalidArgument` naming `..`.
    """
    agent = host_agent(ScriptedModel(["x"])).laws("../laws.bend")
    with pytest.raises(troupekit.TroupekitError) as excinfo:
        agent.build()
    assert code_of(excinfo) in (CODES["Config"], CODES["InvalidArgument"])
    assert ".." in excinfo.value.message


@pending.needs_neutral
def test_host_transport_refuses_other_origin(stub):
    """A reply whose final URL is on another origin is refused.

    Input: a host transport reporting `https://elsewhere.invalid/`. Output:
    passes when the run fails and the transport was called once.
    """
    _, base_url = stub
    other = "https://elsewhere.invalid/v1/chat/completions"
    transport = ChunkTransport([b"data: [DONE]\n\n"], head_url=other)
    agent = cloud_agent(base_url).host_transport(transport).build()
    with pytest.raises(troupekit.TroupekitError):
        agent.run("hi")
    assert len(transport.requests) == 1


@pending.needs_neutral
def test_host_transport_streams_chunks(stub):
    """A host transport's body chunks become the run's text deltas.

    Input: an SSE body split into one chunk per event. Output: passes when
    the deltas join to the streamed text, through the vendor's base URL.
    """
    state, base_url = stub
    chunks = [sse({"choices": [{"index": 0, "delta": {"content": part}}]}) for part in ANSWER]
    chunks.append(sse({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}))
    chunks.append(b"data: [DONE]\n\n")
    transport = ChunkTransport(chunks)
    events = pull_all(cloud_agent(base_url).host_transport(transport).build().stream("hi"))
    deltas = [e.text for e in events if e.kind == troupekit.EventKind.TEXT_DELTA]
    assert "".join(deltas) == "".join(ANSWER)
    assert transport.requests[0].url.startswith(base_url)
    assert state.count() == 0


def sse(payload):
    """Encodes one SSE `data:` event.

    Input: a dict. Output: the bytes.
    """
    return b"data: " + json.dumps(payload).encode() + b"\n\n"


@pending.needs_neutral
def test_full_buffer_drops_deltas_not_lifecycle():
    """A full event buffer drops text deltas but keeps every lifecycle event.

    Input: a child process with a 16-event buffer and a 500-delta reply,
    read only after the run ends. Output: passes when deltas were dropped
    and `run_started` and `run_finished` both arrived.
    """
    script = textwrap.dedent(
        """
        import json, sys
        import troupekit

        class Chatty(troupekit.HostModel):
            def id(self):
                return "chatty"
            def capabilities(self):
                return troupekit.HostModelCaps()
            def generate(self, request, sink):
                for _ in range(500):
                    sink.delta("x")
                return troupekit.HostReply("x" * 500)

        troupekit.init(troupekit.RuntimeSettings(event_buffer=16))
        agent = troupekit.AgentBuilder("a").instructions("Talk.").host_model(Chatty()).build()
        stream = agent.stream("hi")
        stream.result()
        kinds = []
        while (event := stream.next()) is not None:
            kinds.append(event.kind)
        print(json.dumps({"dropped": stream.dropped_deltas(), "kinds": kinds}))
        """
    )
    child = subprocess.run(
        [sys.executable, "-I", "-c", script],
        capture_output=True,
        text=True,
        timeout=RUN_SECONDS_MAX,
        check=False,
    )
    if child.returncode != 0 and "is not built yet" in child.stderr:
        last = child.stderr.strip().splitlines()[-1]
        pytest.skip(f"not built in troupekit-ffi-core yet: {last}")
    assert child.returncode == 0, child.stderr
    report = json.loads(child.stdout.strip().splitlines()[-1])
    assert report["dropped"] > 0
    assert "run_started" in report["kinds"]
    assert report["kinds"][-1] == "run_finished"


@pending.needs_neutral
def test_callback_from_worker_thread_no_deadlock():
    """A Python tool called from a Rust worker thread does not deadlock.

    Input: a tool call on a run started from a Python thread. Output:
    passes when the run returns in time and the tool ran off that thread.
    """
    model = ScriptedModel([tool_reply("echo", {"text": "hi"}), answer_reply("done")])
    tool = RecordingTool()
    agent = host_agent(model, tool).build()
    runner = {}

    def run():
        """Runs the agent and stores the runner thread.

        Output: the answer.
        """
        runner["thread"] = threading.get_ident()
        return agent.run("echo hi")

    assert with_deadline(run) == "done"
    assert tool.calls[0][1] != runner["thread"]


@pending.needs_neutral
def test_concurrent_runs_on_one_handle():
    """One handle serves several runs at once from several Python threads.

    Input: CONCURRENT_RUNS threads calling `run` on the same handle. Output:
    passes when every run answers and the model saw every request.
    """
    model = ScriptedModel(["parallel"])
    agent = host_agent(model).build()
    answers = []
    errors = []

    def run():
        """Runs once and stores the answer or the error.

        Output: None.
        """
        try:
            answers.append(agent.run("hi"))
        except troupekit.TroupekitError as error:
            errors.append(error)

    threads = [threading.Thread(target=run, daemon=True) for _ in range(CONCURRENT_RUNS)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(RUN_SECONDS_MAX)
    assert not any(thread.is_alive() for thread in threads)
    if errors:
        raise errors[0]
    assert answers == ["parallel"] * CONCURRENT_RUNS
    assert len(model.requests) == CONCURRENT_RUNS


def test_host_request_debug_hides_key():
    """The text forms of a host header never show a credential value.

    Input: an `authorization` header, a header the host marked sensitive and
    a `content-type` header. Output: `repr` and `str` of each, and of the
    list, hide both secrets and keep `application/json`.
    """
    headers = [
        troupekit.HostHeader("Authorization", f"Bearer {SECRET}"),
        troupekit.HostHeader("x-trace", "sk-flagged-secret", sensitive=True),
        troupekit.HostHeader("content-type", "application/json"),
    ]
    texts = [repr(headers), str(headers[0]), repr(headers[1]), str(headers[1])]
    for text in texts:
        assert SECRET not in text, text
        assert "sk-flagged-secret" not in text, text
    assert "application/json" in texts[0]
    assert headers[1].sensitive is True
    assert headers[2].sensitive is False


def test_every_case_is_implemented():
    """Every contract case name has a test in this runner.

    Input: `troupekit-ffi-core/contract/cases.txt`. Output: passes when each
    name has a `test_<name>` function here.
    """
    lines = CASES.read_text(encoding="utf-8").splitlines()
    names = [line.strip() for line in lines if line.strip()]
    assert names
    missing = [name for name in names if not callable(globals().get(f"test_{name}"))]
    assert missing == []
