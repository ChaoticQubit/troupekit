import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

BODY_BYTES_MAX = 1024 * 1024
REQUESTS_MAX = 1000
CHAT_PATH = "/v1/chat/completions"
VENDOR_ID = "stub"
MODEL_ID = "stub-model"


class StubState:
    """Shared state of one stub server: the script and the requests seen.

    Input: the answer chunks, and the tool calls `(name, args_json)` that the
    first requests get, one per request, before the answer. Output: a
    thread-safe record of every request body the server received.
    """

    def __init__(self, chunks, tool_calls=()):
        if not chunks:
            raise ValueError("the stub needs at least one answer chunk")
        if len(tool_calls) >= REQUESTS_MAX:
            raise ValueError("the stub script is longer than REQUESTS_MAX")
        self.chunks = list(chunks)
        self.tool_calls = list(tool_calls)
        self.lock = threading.Lock()
        self.bodies = []

    def record(self, body):
        """Stores one request body; refuses more than REQUESTS_MAX requests.

        Input: the decoded JSON body. Output: the tool call `(id, name,
        args_json)` this request gets, or None when it gets the answer.
        """
        with self.lock:
            if len(self.bodies) >= REQUESTS_MAX:
                raise RuntimeError("stub saw more than REQUESTS_MAX requests")
            self.bodies.append(body)
            turn = len(self.bodies)
        if turn > len(self.tool_calls):
            return None
        name, args_json = self.tool_calls[turn - 1]
        return f"call_{turn}", name, args_json

    def count(self):
        """Returns how many chat requests the stub has answered.

        Input: none. Output: an int, 0 or more.
        """
        with self.lock:
            return len(self.bodies)

    def body(self, index):
        """Returns one recorded request body, oldest first.

        Input: a 0-based index below `count()`. Output: the decoded JSON dict;
        IndexError when that request never arrived.
        """
        with self.lock:
            return self.bodies[index]


def make_handler(state):
    """Builds the request handler class bound to one StubState.

    Input: the StubState to record into. Output: a BaseHTTPRequestHandler
    subclass that speaks the OpenAI-compatible chat API, plain or SSE.
    """

    class Handler(BaseHTTPRequestHandler):
        """Answers POST /v1/chat/completions with the scripted turn.

        Input: an HTTP request. Output: JSON, or SSE when the body asks for
        `stream`; 404 for other paths, 413 for bodies over BODY_BYTES_MAX.
        """

        def log_message(self, format, *args):
            """Drops the per-request access log so test output stays readable.

            Input: the log format and args. Output: none.
            """
            return

        def do_POST(self):
            """Handles one chat request on CHAT_PATH, recording its body first.

            Input: the request. Output: a JSON or SSE reply, or a 404 or 413 error.
            """
            if self.path != CHAT_PATH:
                self.send_error(404)
                return
            length = int(self.headers.get("Content-Length", "0"))
            if length < 0 or length > BODY_BYTES_MAX:
                self.send_error(413)
                return
            body = json.loads(self.rfile.read(length) or b"{}")
            call = state.record(body)
            if body.get("stream"):
                self.reply_stream(call)
            else:
                self.reply_json(call)

        def reply_json(self, call):
            """Sends the turn as one non-streaming chat completion.

            Input: the tool call of this turn, or None for the answer.
            Output: none; writes the JSON body with its length.
            """
            if call is None:
                message = {"role": "assistant", "content": "".join(state.chunks)}
                finish = "stop"
            else:
                message = {"role": "assistant", "content": None, "tool_calls": [tool_call(call)]}
                finish = "tool_calls"
            payload = {
                "choices": [{"index": 0, "message": message, "finish_reason": finish}],
                "usage": {"prompt_tokens": 3, "completion_tokens": len(state.chunks)},
            }
            data = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def reply_stream(self, call):
            """Sends the turn as SSE events, then finish, usage and [DONE].

            Input: the tool call of this turn, or None for the answer chunks.
            Output: none; the connection closes after [DONE].
            """
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            if call is None:
                for chunk in state.chunks:
                    self.send_event({"choices": [{"index": 0, "delta": {"content": chunk}}]})
                finish = "stop"
            else:
                delta = {"tool_calls": [dict(tool_call(call), index=0)]}
                self.send_event({"choices": [{"index": 0, "delta": delta}]})
                finish = "tool_calls"
            self.send_event({"choices": [{"index": 0, "delta": {}, "finish_reason": finish}]})
            usage = {"prompt_tokens": 3, "completion_tokens": len(state.chunks)}
            self.send_event({"choices": [], "usage": usage})
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

        def send_event(self, payload):
            """Writes one SSE `data:` event and flushes it to the client.

            Input: a JSON-able dict. Output: none.
            """
            self.wfile.write(b"data: " + json.dumps(payload).encode() + b"\n\n")
            self.wfile.flush()

    return Handler


def tool_call(call):
    """Renders one scripted tool call in the Chat Completions shape.

    Input: `(id, name, args_json)`. Output: the `tool_calls` list item, with
    `arguments` as the JSON string the API sends.
    """
    call_id, name, args_json = call
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": args_json}}


def catalog_for(base_url):
    """Builds a catalog TOML naming one no-key vendor at the stub's URL.

    Input: the stub base URL ending in `/v1`. Output: catalog TOML text with
    vendor VENDOR_ID and model MODEL_ID (selector `stub/stub-model`).
    """
    return f"""
[[vendor]]
id = "{VENDOR_ID}"
name = "Stub"
protocol = "openai_compatible"
base_url = "{base_url}"
auth = "none"

[[vendor.model]]
id = "{MODEL_ID}"
context_window = 8192
input_usd_micros_per_mtok = 0
output_usd_micros_per_mtok = 0
price_verified = true
"""


def start(chunks, tool_calls=()):
    """Starts a stub on an ephemeral 127.0.0.1 port in a daemon thread.

    Input: the answer chunks and the tool calls served first. Output:
    (server, state, base_url), where base_url ends in `/v1`; call
    `server.shutdown()` to stop it.
    """
    state = StubState(chunks, tool_calls)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(state))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    if not 0 < port < 65536:
        raise RuntimeError("the stub got no port")
    return server, state, f"http://127.0.0.1:{port}/v1"


def main():
    """Runs the stub in the foreground and prints its base URL.

    Input: optional answer words on the command line. Output: the base URL on
    stdout; serves until interrupted.
    """
    server, _, base_url = start(sys.argv[1:] or ["Hello", " from", " the stub"])
    print(base_url, flush=True)
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
