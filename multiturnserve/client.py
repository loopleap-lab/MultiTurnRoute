"""The app's side: multiturnserve.open(chat_id, gateway=":8080") -> a session that talks to a running gateway.

  s = multiturnserve.open("chat-1", gateway=":8080")
  reply = s.chat("Summarize this paper")      # route, run, observe on the gateway
  print(reply.turn, reply.model, reply.cost)

Without gateway=, open() builds a local session from a serve config (config="serve.yaml") instead.
"""
from __future__ import annotations

from .serving.http import post_json
from .types import Decision, Reply, Usage


def gateway_url(gateway: str) -> str:
    """":8080" -> http://127.0.0.1:8080; "host:8080" -> http://host:8080; URLs are kept."""
    g = gateway.rstrip("/")
    if g.startswith(("http://", "https://")):
        return g
    return "http://" + ("127.0.0.1" + g if g.startswith(":") else g)


class RemoteSession:
    """A chat on a gateway. gateway: an address, or an HTTP client object with .post / .get (e.g. FastAPI's
    TestClient or an httpx.Client) whose base URL is the gateway."""

    def __init__(self, chat_id: str, gateway=":8080"):
        self.chat_id = chat_id
        self.http = None if isinstance(gateway, str) else gateway
        self.url = gateway_url(gateway) if isinstance(gateway, str) else ""
        self.messages: list = []

    def _post(self, path: str, payload: dict) -> dict:
        if self.http is None:
            return post_json(self.url + path, payload)
        r = self.http.post(self.url + path, json=payload)
        if r.status_code >= 400:
            raise RuntimeError(f"{path}: HTTP {r.status_code}: {r.text[:500]}")
        return r.json()

    def chat(self, msg: str) -> Reply:
        self.messages.append({"role": "user", "content": msg})
        r = self._post("/v1/chat/completions", {"chat_id": self.chat_id, "messages": self.messages})
        text = r["choices"][0]["message"]["content"]
        self.messages.append({"role": "assistant", "content": text})
        x = r["multiturnserve"]
        return Reply(self.chat_id, x["turn"], x["model"], text, x["cost"], Usage.of(r["usage"]))

    def route(self, messages: list | None = None) -> Decision:
        r = self._post("/v1/route", {"chat_id": self.chat_id, "messages": messages or self.messages})
        return Decision(r["model"], r["quality"], r["cost"], r["turn"] - 1)

    def observe(self, outcome) -> dict:
        u = Usage.of(outcome.usage)
        return self._post("/v1/observe", {"chat_id": self.chat_id, "model": outcome.model, "text": outcome.text,
                                          "query": outcome.query, "cost": outcome.cost,
                                          "usage": {"prompt_tokens": u.prompt_tokens, "cached_tokens": u.cached_tokens,
                                                    "output_tokens": u.output_tokens}})

    def state(self) -> dict:
        if self.http is None:
            import json
            import urllib.request
            with urllib.request.urlopen(f"{self.url}/v1/sessions/{self.chat_id}") as r:
                return json.loads(r.read().decode())
        return self.http.get(f"{self.url}/v1/sessions/{self.chat_id}").json()
