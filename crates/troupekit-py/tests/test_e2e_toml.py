import pathlib
import sys

import pytest

import troupekit

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import stub_server

FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "agent.toml"
CHUNKS = ["Hello", " from", " the stub"]
EVENTS_MAX = 10_000

pytestmark = pytest.mark.xfail(
    not hasattr(troupekit, "AgentBuilder"),
    reason="B5: troupekit-py does not expose AgentBuilder, AgentHandle and RunStream yet",
    raises=AttributeError,
    strict=True,
)


def catalog_for(base_url):
    """Builds a catalog TOML naming one no-key vendor at the stub's URL.

    Input: the stub base URL ending in `/v1`. Output: catalog TOML text with
    vendor `stub` and model `stub-model`, the selector the fixture uses.
    """
    return f"""
[[vendor]]
id = "stub"
name = "Stub"
protocol = "openai_compatible"
base_url = "{base_url}"
auth = "none"

[[vendor.model]]
id = "stub-model"
context_window = 8192
input_usd_micros_per_mtok = 0
output_usd_micros_per_mtok = 0
price_verified = true
"""


@pytest.fixture
def stub():
    """Runs one stub server for a test and stops it afterwards.

    Input: none. Output: (state, base_url) of the running stub.
    """
    server, state, base_url = stub_server.start(CHUNKS)
    yield state, base_url
    server.shutdown()


def build_agent(base_url):
    """Loads the fixture persona and points it at the stub catalog.

    Input: the stub base URL. Output: a built `AgentHandle`.
    """
    text = FIXTURE.read_text(encoding="utf-8")
    builder = troupekit.AgentBuilder.from_toml(text).catalog_toml(catalog_for(base_url))
    return builder.build()


def test_pip_installed_toml_agent_answers(stub):
    """A pip-installed wheel loads a TOML agent and returns the stub's answer.

    Input: the fixture persona, a stub vendor. Output: passes when the answer
    is the stub's text and the stub saw exactly one chat request.
    """
    state, base_url = stub
    agent = build_agent(base_url)
    answer = agent.run("Say hello")
    assert answer == "".join(CHUNKS)
    assert state.count() == 1


def test_stream_from_python_yields_deltas(stub):
    """Streaming from Python yields every text delta, then the run's end.

    Input: the fixture persona, a stub vendor. Output: passes when the deltas
    join to the stub's text, the last event is `run_finished`, and the stub
    saw exactly one streaming request.
    """
    state, base_url = stub
    agent = build_agent(base_url)
    stream = agent.stream("Say hello")
    kinds = []
    deltas = []
    for _ in range(EVENTS_MAX):
        event = stream.next()
        if event is None:
            break
        kinds.append(event.kind)
        if event.kind == "text_delta":
            deltas.append(event.text)
    else:
        pytest.fail("stream did not end within EVENTS_MAX events")
    assert "".join(deltas) == "".join(CHUNKS)
    assert kinds[-1] == "run_finished"
    assert stream.dropped_deltas() == 0
    assert state.count() == 1
