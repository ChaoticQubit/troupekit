import json
import pathlib
import sys
import threading

import pytest

import troupekit

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import stub_server

ANSWER = ["Done"]
RUN_SECONDS_MAX = 30
TOOL_NAME = "echo"
TOOL_ARGS = {"text": "hi"}

pytestmark = pytest.mark.xfail(
    not hasattr(troupekit, "AgentBuilder"),
    reason="B5: troupekit-py does not expose AgentBuilder and HostTool support yet",
    raises=AttributeError,
    strict=True,
)


class EchoTool:
    """A Python host tool that records each call and the thread it ran on.

    Input: an optional exception to raise from `call`. Output: `calls` holds
    `(args, thread_id)` per call; `call` returns the `text` argument.
    """

    def __init__(self, error=None):
        self.error = error
        self.calls = []

    def name(self):
        """Returns the tool name the model calls.

        Input: none. Output: TOOL_NAME.
        """
        return TOOL_NAME

    def description(self):
        """Returns the description shown to the model.

        Input: none. Output: one sentence.
        """
        return "Repeats the text it is given."

    def schema_json(self):
        """Returns the JSON Schema of the tool arguments.

        Input: none. Output: a JSON object with one required string `text`.
        """
        schema = {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        }
        return json.dumps(schema)

    def call(self, args_json):
        """Records the call, then echoes `text` or raises the set error.

        Input: the arguments as a JSON string. Output: the `text` value;
        raises `self.error` when one was given.
        """
        self.calls.append((json.loads(args_json), threading.get_ident()))
        if self.error is not None:
            raise self.error
        return json.loads(args_json)["text"]


@pytest.fixture
def stub():
    """Runs a stub whose first reply calls the echo tool, then answers.

    Input: none. Output: (state, base_url) of the running stub.
    """
    script = [(TOOL_NAME, json.dumps(TOOL_ARGS))]
    server, state, base_url = stub_server.start(ANSWER, script)
    yield state, base_url
    server.shutdown()


def run_with_deadline(agent, prompt):
    """Runs `agent.run(prompt)` on a daemon thread and waits RUN_SECONDS_MAX.

    Input: a built agent and the prompt. Output: (answer, runner thread id);
    fails the test when the run hangs (a GIL deadlock), re-raises its error.
    """
    outcome = {}

    def target():
        """Stores the answer or the exception of one run.

        Input: none. Output: none; fills `outcome`.
        """
        outcome["runner"] = threading.get_ident()
        try:
            outcome["answer"] = agent.run(prompt)
        except BaseException as error:
            outcome["error"] = error

    worker = threading.Thread(target=target, daemon=True)
    worker.start()
    worker.join(RUN_SECONDS_MAX)
    if worker.is_alive():
        pytest.fail(f"agent.run did not return within {RUN_SECONDS_MAX} s: GIL deadlock")
    if "error" in outcome:
        raise outcome["error"]
    return outcome["answer"], outcome["runner"]


def build_agent(base_url, tool):
    """Builds a code agent on the stub vendor with one Python host tool.

    Input: the stub base URL and the tool object. Output: a built `AgentHandle`.
    """
    builder = (
        troupekit.AgentBuilder("helper")
        .model(f"{stub_server.VENDOR_ID}/{stub_server.MODEL_ID}")
        .instructions("Use the echo tool, then answer.")
        .catalog_toml(stub_server.catalog_for(base_url))
        .tool(tool)
    )
    return builder.build()


def tool_messages(body):
    """Returns the `tool` role messages of one recorded chat request.

    Input: a request body dict. Output: the list of tool result messages.
    """
    return [m for m in body.get("messages", []) if m.get("role") == "tool"]


def test_python_tool_called_from_worker_thread_does_not_deadlock(stub):
    """A Python tool called from a tokio blocking thread runs and returns.

    Input: a stub that asks for one echo call. Output: passes when the run
    answers within RUN_SECONDS_MAX, the tool ran once off the calling thread
    with the scripted arguments, and its output reached the model.
    """
    state, base_url = stub
    tool = EchoTool()
    answer, runner = run_with_deadline(build_agent(base_url, tool), "Echo hi")
    assert answer == "".join(ANSWER)
    assert len(tool.calls) == 1
    args, thread_id = tool.calls[0]
    assert args == TOOL_ARGS
    assert thread_id != runner
    assert state.count() == 2
    results = tool_messages(state.body(1))
    assert len(results) == 1
    assert "hi" in json.dumps(results[0])


def test_python_exception_becomes_host_error(stub):
    """A Python exception in a tool reaches the model as an error output.

    Input: a stub that asks for one echo call; the tool raises. Output: passes
    when the run still answers, and the next request carries the exception
    text in the tool result instead of the process crashing.
    """
    state, base_url = stub
    tool = EchoTool(error=ValueError("echo broke: code 42"))
    answer, _ = run_with_deadline(build_agent(base_url, tool), "Echo hi")
    assert answer == "".join(ANSWER)
    assert len(tool.calls) == 1
    assert state.count() == 2
    results = tool_messages(state.body(1))
    assert len(results) == 1
    assert "echo broke: code 42" in json.dumps(results[0])
