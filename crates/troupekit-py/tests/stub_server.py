import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

BODY_BYTES_MAX = 1024 * 1024
REQUESTS_MAX = 1000
CHAT_PATH = "/v1/chat/completions"


class StubState:
    """Shared state of one stub server: the scripted answer and the requests seen.

    Input: the answer text and how it is split into stream chunks. Output: a
    thread-safe record of every request body the server received.
    """

    def __init__(self, chunks):
        if not chunks:
            raise ValueError("the stub needs at least one answer chunk")
        self.chunks = list(chunks)
        self.lock = threading.Lock()
        self.bodies = []

    def record(self, body):
        """Stores one request body; refuses more than REQUESTS_MAX requests.

        Input: the decoded JSON body. Output: the number of requests seen so far.
        """
        with self.lock:
            if len(self.bodies) >= REQUESTS_MAX:
                raise RuntimeError("stub saw more than REQUESTS_MAX requests")
            self.bodies.append(body)
            return len(self.bodies)

    def count(self):
        """Returns how many chat requests the stub has answered.

        Input: none. Output: an int, 0 or more.
        """
        with self.lock:
            return len(self.bodies)


def make_handler(state):
    """Builds the request handler class bound to one StubState.

    Input: the StubState to record into. Output: a BaseHTTPRequestHandler
    subclass that speaks the OpenAI-compatible chat API, plain or SSE.
    """

    class Handler(BaseHTTPRequestHandler):
        """Answers POST /v1/chat/completions with the scripted chunks.

        Input: an HTTP request. Output: JSON, or SSE when the body asks for
        `stream`; 404 for other paths, 413 for bodies over BODY_BYTES_MAX.
        """

        def log_message(self, format, *args):
            """Keeps the test output quiet. Input: log args. Output: none."""
            return

        def do_POST(self):
            """Handles one chat request. Input: the request. Output: the reply."""
            if self.path != CHAT_PATH:
                self.send_error(404)
                return
            length = int(self.headers.get("Content-Length", "0"))
            if length < 0 or length > BODY_BYTES_MAX:
                self.send_error(413)
                return
            body = json.loads(self.rfile.read(length) or b"{}")
            state.record(body)
            if body.get("stream"):
                self.reply_stream()
            else:
                self.reply_json()

        def reply_json(self):
            """Sends the whole answer as one completion. Input: none. Output: none."""
            text = "".join(state.chunks)
            payload = {
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 3, "completion_tokens": len(state.chunks)},
            }
            data = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def reply_stream(self):
            """Sends one SSE event per chunk, then stop, usage and [DONE].

            Input: none. Output: none; the connection closes after [DONE].
            """
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            for chunk in state.chunks:
                self.send_event({"choices": [{"index": 0, "delta": {"content": chunk}}]})
            self.send_event({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
            usage = {"prompt_tokens": 3, "completion_tokens": len(state.chunks)}
            self.send_event({"choices": [], "usage": usage})
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

        def send_event(self, payload):
            """Writes one SSE `data:` line. Input: a JSON-able dict. Output: none."""
            self.wfile.write(b"data: " + json.dumps(payload).encode() + b"\n\n")
            self.wfile.flush()

    return Handler


def start(chunks):
    """Starts a stub on an ephemeral 127.0.0.1 port in a daemon thread.

    Input: the answer chunks. Output: (server, state, base_url), where
    base_url ends in `/v1`; call `server.shutdown()` to stop it.
    """
    state = StubState(chunks)
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
