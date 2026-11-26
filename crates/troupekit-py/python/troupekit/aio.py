"""Async forms of the troupekit handles, for asyncio code.

Same names as `troupekit`; every blocking call runs in `asyncio.to_thread`,
so the event loop keeps running. Input: none. Output: the wrapper classes.
"""

import asyncio

from troupekit import _troupekit


class AgentBuilder:
    """Async-building form of `troupekit.AgentBuilder`.

    Input: an agent name, or `_inner`, a native builder to wrap. Output: a
    builder whose setters return wrapped builders and whose `build` is async.
    """

    def __init__(self, name=None, *, _inner=None):
        if (name is None) == (_inner is None):
            raise TypeError("pass exactly one of name or _inner")
        self._inner = _inner if _inner is not None else _troupekit.AgentBuilder(name)

    @staticmethod
    def from_toml(text):
        """Loads a persona TOML text. Input: the text. Output: a wrapped builder.

        The text is validated by `build`, as in the native builder.
        """
        return AgentBuilder(_inner=_troupekit.AgentBuilder.from_toml(text))

    @staticmethod
    def from_toml_file(path):
        """Loads a persona TOML file. Input: the path. Output: a wrapped builder.

        The file is read and validated by `build`, as in the native builder.
        """
        return AgentBuilder(_inner=_troupekit.AgentBuilder.from_toml_file(path))

    def __getattr__(self, name):
        """Forwards a setter and wraps the builder it returns.

        Input: a setter name of the native builder. Output: a function that
        calls it with the same arguments and returns a wrapped builder.
        """
        setter = getattr(self._inner, name)

        def call(*args, **kwargs):
            """Calls the native setter.

            Input: its arguments. Output: a wrapped builder.
            """
            return AgentBuilder(_inner=setter(*args, **kwargs))

        return call

    async def build(self):
        """Builds the agent off the event loop.

        Input: none. Output: an `AgentHandle`; raises `TroupekitError`.
        """
        return AgentHandle(await asyncio.to_thread(self._inner.build))


class AgentHandle:
    """Async form of `troupekit.AgentHandle`.

    Input: the native handle. Output: async `run`, `run_outcome`, `start`,
    `resume`, `resume_for`, and `stream` returning an async `RunStream`.
    """

    def __init__(self, inner):
        self._inner = inner

    async def run(self, input):
        """Runs the agent.

        Input: the user text. Output: the answer text.
        """
        return await asyncio.to_thread(self._inner.run, input)

    async def run_outcome(self, input):
        """Runs the agent.

        Input: the user text. Output: a `RunResult`.
        """
        return await asyncio.to_thread(self._inner.run_outcome, input)

    async def start(self, input):
        """Starts a run without a model call.

        Input: the text. Output: the run id.
        """
        return await asyncio.to_thread(self._inner.start, input)

    async def resume(self, run_id):
        """Resumes a run to its end.

        Input: the run id. Output: the answer text.
        """
        return await asyncio.to_thread(self._inner.resume, run_id)

    async def resume_for(self, run_id, budget_ms):
        """Resumes a run for one window.

        Input: run id, budget. Output: a `Progress`.
        """
        return await asyncio.to_thread(self._inner.resume_for, run_id, budget_ms)

    async def stream(self, input):
        """Starts a streamed run once its run id is known.

        Input: the user text. Output: an async `RunStream`; a setup error
        raises `TroupekitError` here, as `start` does.
        """
        return RunStream(await asyncio.to_thread(self._inner.stream, input))


class GraphBuilder:
    """Async-building form of `troupekit.GraphBuilder`.

    Input: a graph name, or `_inner`, a native builder to wrap. Output: a
    builder whose setters return wrapped builders and whose `build` is async.
    """

    def __init__(self, name=None, *, _inner=None):
        if (name is None) == (_inner is None):
            raise TypeError("pass exactly one of name or _inner")
        self._inner = _inner if _inner is not None else _troupekit.GraphBuilder(name)

    def agent(self, id, agent):
        """Adds an agent node. Input: the node id and an agent builder of
        either module. Output: a wrapped builder.
        """
        native = agent._inner if isinstance(agent, AgentBuilder) else agent
        return GraphBuilder(_inner=self._inner.agent(id, native))

    def __getattr__(self, name):
        """Forwards a setter and wraps the builder it returns.

        Input: a setter name of the native builder. Output: a function that
        calls it with the same arguments and returns a wrapped builder.
        """
        setter = getattr(self._inner, name)

        def call(*args, **kwargs):
            """Calls the native setter.

            Input: its arguments. Output: a wrapped builder.
            """
            return GraphBuilder(_inner=setter(*args, **kwargs))

        return call

    async def build(self):
        """Builds the graph off the event loop.

        Input: none. Output: a `GraphHandle`; raises `TroupekitError`.
        """
        return GraphHandle(await asyncio.to_thread(self._inner.build))


class GraphHandle:
    """Async form of `troupekit.GraphHandle`.

    Input: the native handle. Output: async `run`, `run_result`, `start`,
    `resume`, `resume_for`, and `stream` returning an async `RunStream`.
    """

    def __init__(self, inner):
        self._inner = inner

    async def run(self, input):
        """Runs the graph.

        Input: the input text. Output: the output text.
        """
        return await asyncio.to_thread(self._inner.run, input)

    async def run_result(self, input):
        """Runs the graph.

        Input: the input text. Output: a `GraphResult`.
        """
        return await asyncio.to_thread(self._inner.run_result, input)

    async def start(self, input):
        """Starts a run without running a step.

        Input: the text. Output: the run id.
        """
        return await asyncio.to_thread(self._inner.start, input)

    async def resume(self, run_id):
        """Resumes a run to its end.

        Input: the run id. Output: the output text.
        """
        return await asyncio.to_thread(self._inner.resume, run_id)

    async def resume_for(self, run_id, budget_ms):
        """Resumes a run for one window.

        Input: run id, budget. Output: a `Progress`.
        """
        return await asyncio.to_thread(self._inner.resume_for, run_id, budget_ms)

    async def stream(self, input):
        """Starts a streamed graph run once its run id is known.

        Input: the input text. Output: an async `RunStream`; a setup error
        raises `TroupekitError` here, as `start` does.
        """
        return RunStream(await asyncio.to_thread(self._inner.stream, input))


class RunStream:
    """Async form of `troupekit.RunStream`, also an async iterator.

    Input: the native stream. Output: `async for event in stream` yields each
    event; `next` and `result` are async, the rest forward unchanged.
    """

    def __init__(self, inner):
        self._inner = inner

    def __aiter__(self):
        """Returns the stream itself.

        Input: none. Output: this stream.
        """
        return self

    async def __anext__(self):
        """Waits for the next event. Input: none. Output: the event.

        Raises StopAsyncIteration when the stream has ended.
        """
        event = await self.next()
        if event is None:
            raise StopAsyncIteration
        return event

    async def next(self):
        """Waits for the next event.

        Input: none. Output: an `Event` or None at the end.
        """
        return await asyncio.to_thread(self._inner.next)

    async def result(self):
        """Waits for the run to end.

        Input: none. Output: its `RunResult`.
        """
        return await asyncio.to_thread(self._inner.result)

    def set_listener(self, listener):
        """Pushes events to `listener.on_event`.

        Input: the listener. Output: None.
        """
        return self._inner.set_listener(listener)

    def cancel(self):
        """Cancels the run; its checkpoint is kept.

        Input: none. Output: None.
        """
        return self._inner.cancel()

    def run_id(self):
        """Returns the run id.

        Input: none. Output: the id text.
        """
        return self._inner.run_id()

    def dropped_deltas(self):
        """Returns how many text deltas a full buffer dropped.

        Output: an int.
        """
        return self._inner.dropped_deltas()
