"""Stand-ins for tests: a router that follows a script, and an in-process OpenAI-compatible HTTP server."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from sessionroutex import Router, register_router
from sessionroutex.parts import CacheCost

PRICES = {"small": {"input": 1.0, "cached": 0.1, "output": 2.0},     # USD per 1M tokens
          "big": {"input": 10.0, "cached": 1.0, "output": 30.0}}


@register_router
class Scripted(Router):
    """Picks the models of `script` in turn (cycling); still goes through pick() with zero quality and CacheCost."""
    name = "scripted-test"
    script = ["small", "small", "big", "small"]
    cost = CacheCost()

    def __init__(self):
        self.i = 0

    def decide(self, turn, state):
        want = self.script[self.i % len(self.script)]
        self.i += 1
        c = self.cost(turn, state.kv_cache)
        q = np.array([1.0 if m == want else 0.0 for m in turn.models])
        self.lam = 0.0
        return self.pick(q, c)


class FakeOpenAI:
    """POST /v1/chat/completions. The reply depends on the requested model:
    "fake-user" -> the next scripted user message, then "[END]"; "fake-judge" -> "Score: 7";
    anything else -> "answer N" with usage (prompt = 10 per message, half of it cached after the first call)."""

    def __init__(self, user_turns: int = 2):
        self.requests, self.user_calls, self.user_turns = [], 0, user_turns
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append({"path": self.path, "body": body, "auth": self.headers.get("Authorization")})
                text, usage = outer.reply(body)
                data = json.dumps({"choices": [{"message": {"role": "assistant", "content": text}}],
                                   "usage": usage}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def reply(self, body):
        m = body["model"]
        if m == "fake-user":
            self.user_calls += 1
            return ("[END]" if self.user_calls % (self.user_turns + 1) == 0 else f"follow-up {self.user_calls}"), {}
        if m == "fake-judge":
            return "Score: 7", {}
        n = 10 * len(body["messages"])
        return f"answer {len(self.requests)}", {"prompt_tokens": n, "completion_tokens": 5,
                                               "prompt_tokens_details": {"cached_tokens": n // 2 if n > 10 else 0}}

    def close(self):
        self.server.shutdown()
