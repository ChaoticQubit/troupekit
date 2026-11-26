"""B8: the cost of crossing the PyO3 boundary from Python.

Runs the same four workloads as `troupekit-ffi/examples/overhead.rs`, each
`TROUPEKIT_BENCH_N` times (default 10 000), and prints one line per bench
with p50 and p99 in nanoseconds. CI runs it twice: quiet and loaded.
"""

import os
import threading
import time

import troupekit

TOOL_OUTPUT = "noop-ok"
ITERATIONS_MAX = 1_000_000
TERMINAL = {"run_finished", "run_failed", "run_paused"}


class BenchModel(troupekit.HostModel):
    """A host model that calls the no-op tool once per run, then answers.

    Input: none. Output: emulated tool calls (K8); stateless across runs.
    """

    def id(self):
        """Names the model.

        Input: none. Output: `bench`.
        """
        return "bench"

    def capabilities(self):
        """Declares a text-only streaming model.

        Input: none. Output: `HostModelCaps` with emulated tools.
        """
        return troupekit.HostModelCaps(
            tools=False, json_schema=False, streaming=True, context_tokens=4096
        )

    def generate(self, request, sink):
        """Asks for the tool until its output is in the request, then answers.

        Input: the request and sink. Output: a `HostReply`.
        """
        called = any(TOOL_OUTPUT in message.text for message in request.messages)
        text = '{"answer": "done"}' if called else '{"tool": "noop", "arguments": {}}'
        sink.delta(text)
        return troupekit.HostReply(text=text, input_tokens=1, output_tokens=1)


class PlainModel(BenchModel):
    """A host model that answers at once with one short text.

    Input: none. Output: `x` for every request.
    """

    def generate(self, request, sink):
        """Answers `x`.

        Input: the request and sink. Output: a `HostReply`.
        """
        sink.delta("x")
        return troupekit.HostReply(text="x", input_tokens=1, output_tokens=1)


class NoopTool(troupekit.HostTool):
    """A host tool that does nothing and counts its calls.

    Input: none. Output: `TOOL_OUTPUT`; `calls` proves every iteration ran.
    """

    def __init__(self):
        """Starts the count at zero.

        Input: none. Output: none.
        """
        self.calls = 0
        self.lock = threading.Lock()

    def name(self):
        """Names the tool.

        Input: none. Output: `noop`.
        """
        return "noop"

    def description(self):
        """Describes the tool.

        Input: none. Output: one sentence.
        """
        return "Does nothing."

    def schema_json(self):
        """Declares no arguments.

        Input: none. Output: an empty object schema.
        """
        return '{"type":"object","properties":{}}'

    def call(self, args_json):
        """Counts the call.

        Input: the argument JSON. Output: `TOOL_OUTPUT`.
        """
        with self.lock:
            self.calls += 1
        return TOOL_OUTPUT


class Ender(troupekit.EventListener):
    """Signals the end of each listened run.

    Input: none. Output: `ended` gets one release per terminal event.
    """

    def __init__(self):
        """Creates the semaphore.

        Input: none. Output: none.
        """
        self.ended = threading.Semaphore(0)

    def on_event(self, event):
        """Releases once per terminal event.

        Input: the event. Output: none.
        """
        if event.kind in TERMINAL:
            self.ended.release()


def benchmark(name, count, label, body):
    """Times `body` for `count` runs and prints one result line.

    Input: the bench name, count, label and body. Output: the line, also
    appended to `TROUPEKIT_BENCH_OUT` when set.
    """
    samples = []
    for _ in range(count):
        start = time.perf_counter_ns()
        body()
        samples.append(time.perf_counter_ns() - start)
    if len(samples) != count:
        raise AssertionError(f"{name} ran {len(samples)} of {count}")
    samples.sort()
    p50 = samples[count // 2]
    p99 = samples[min(count * 99 // 100, count - 1)]
    line = (
        f"overhead lang=python bench={name} n={count}"
        f" p50_ns={p50} p99_ns={p99} label={label}"
    )
    print(line, flush=True)
    path = os.environ.get("TROUPEKIT_BENCH_OUT")
    if path:
        with open(path, "a", encoding="utf-8") as out:
            out.write(line + "\n")
    return line


def agent(model, tools=()):
    """Builds an agent handle on `model` with the given host tools.

    Input: the model and tools. Output: the built `AgentHandle`.
    """
    builder = troupekit.AgentBuilder("bench").instructions("Benchmark.").host_model(model)
    for tool in tools:
        builder = builder.tool(tool)
    return builder.build()


def require(condition, message):
    """Raises when a benchmark invariant does not hold.

    Input: the condition and message. Output: none; `-O` cannot strip it.
    """
    if not condition:
        raise AssertionError(message)


def main():
    """Runs the four benches.

    Input: `TROUPEKIT_BENCH_N`, `TROUPEKIT_BENCH_LABEL`. Output: one line each.
    """
    count = int(os.environ.get("TROUPEKIT_BENCH_N", "10000"))
    require(0 < count <= ITERATIONS_MAX, "TROUPEKIT_BENCH_N out of bounds")
    label = os.environ.get("TROUPEKIT_BENCH_LABEL", "quiet")
    counts = {"version": 0, "pull": 0, "listen": 0}

    def call_version():
        """Calls `version()` once and counts it.

        Input: none. Output: none.
        """
        require(troupekit.version(), "empty version")
        counts["version"] += 1

    benchmark("version", count, label, call_version)
    require(counts["version"] == count, "version count")

    tool = NoopTool()
    tooled = agent(BenchModel(), [tool])

    def run_tool():
        """Runs one agent turn that calls the no-op tool.

        Input: none. Output: none.
        """
        require(tooled.run("go") == "done", "answer")

    benchmark("tool_round_trip", count, label, run_tool)
    require(tool.calls == count, f"tool ran {tool.calls} of {count}")

    plain = agent(PlainModel())

    def pull():
        """Reads one streamed run by pull and counts it.

        Input: none. Output: none.
        """
        for _ in plain.stream("go"):
            pass
        counts["pull"] += 1

    benchmark("event_pull", count, label, pull)
    require(counts["pull"] == count, "pull count")

    ender = Ender()

    def listen():
        """Reads one streamed run with a listener and counts it.

        Input: none. Output: none.
        """
        stream = plain.stream("go")
        stream.set_listener(ender)
        stream.result()
        require(ender.ended.acquire(timeout=30), "no terminal event in 30 s")
        counts["listen"] += 1

    benchmark("event_listener", count, label, listen)
    require(counts["listen"] == count, "listen count")


if __name__ == "__main__":
    main()
