"""Sessions: one per chat. A session holds the chat's state (history and KV cache), asks the router for each turn,
runs the chosen model on its serving backend, and updates the state from the usage the backend reports.

  route(messages) -> Decision(model, quality, cost)    the router's pick for the next turn (no state change)
  run(decision, messages) -> Outcome                   the chosen model answers; realized cost from its usage
  observe(outcome) -> SessionState                     history and KV cache updated, one log line written
  chat(msg) -> Reply                                   all three, for a local session

A Runtime is the shared part: the router, the backends, the encoder and the sessions by chat id (the gateway serves
one). Prompt lengths are estimated before a turn at 4 characters per token; the accounting uses the real usage.
"""
from __future__ import annotations

import dataclasses
import json
import os
import threading
import time
from pathlib import Path

import numpy as np

from .arena.data import Conversation, Turn as Record
from .parts.encoder import history_text, make_encoder
from .pricing import turn_cost
from .types import Decision, Outcome, Reply, SessionState, Turn, Usage

DEFAULT_OUT_TOKENS = 256   # predicted output tokens per model when the router has no length head


def estimate_tokens(messages: list) -> int:
    return max(1, round(sum(len(str(m["content"])) for m in messages) / 4))


def last_user(messages: list) -> str:
    return next((str(m["content"]) for m in reversed(messages) if m["role"] == "user"), "")


class TurnLog:
    """One JSONL line per answered turn: chat, turn, model, tokens, cost (no message texts)."""

    def __init__(self, folder):
        self.path = Path(folder) / "turns.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()

    def write(self, row: dict) -> None:
        with self.lock, open(self.path, "a") as f:
            f.write(json.dumps(row) + "\n")


class Session:
    def __init__(self, chat_id: str, router, serving: dict, encoder=None, log: TurnLog | None = None):
        self.chat_id, self.router, self.serving, self.encoder, self.log = chat_id, router, serving, encoder, log
        self.models = tuple(serving)              # model -> backend, in config order (ties go to the first)
        self.state = SessionState()
        self.messages: list = []                  # the chat as chat() sends it
        self.total = 0.0                          # USD so far
        self.lock = threading.Lock()
        self._emb: dict = {}                      # turn_id -> {"q", "a", "h"} of this chat (with an encoder)
        self._pending = None                      # (Turn, Decision) of the routed, not yet observed turn

    def _turn(self, messages: list) -> Turn:
        t, query = len(self.state.history), last_user(messages)
        tid = lambda i: f"{self.chat_id}#{i + 1}"  # noqa: E731
        recs = [Record(tid(i), i + 1, x.query, x.answer, "", i > 0, {}, {}, {}) for i, x in enumerate(self.state.history)]
        recs.append(Record(tid(t), t + 1, query, "", "", t > 0, {}, {}, {}))
        conv = Conversation(self.chat_id, "live", None, None, tuple(recs))
        emb = None
        if self.encoder is not None:
            q, h = self.encoder([query, history_text(self.state.history, query)])
            self._emb[tid(t)] = {"q": q, "a": np.zeros_like(q), "h": h}
            emb = self._emb
        n = estimate_tokens(messages)
        prices = {m: self.serving[m].prices(m) for m in self.models}
        turn = Turn(self.chat_id, t, query, self.models, {m: n for m in self.models}, None, prices, conv, emb,
                    tuple(messages))
        length = getattr(self.router, "length", None)
        out = length(turn) if length is not None and emb is not None else \
            np.full(len(self.models), float(DEFAULT_OUT_TOKENS))
        return dataclasses.replace(turn, out_tokens=out)

    def route(self, messages: list | None = None) -> Decision:
        turn = self._turn(list(self.messages if messages is None else messages))
        d = self.router.route(turn, self.state)
        if d.model not in self.serving:
            raise ValueError(f"router {self.router.name!r} picked {d.model!r}, which no backend serves "
                             f"(served: {', '.join(self.models)})")
        self._pending = (turn, d)
        return d

    def run(self, decision: Decision, messages: list | None = None) -> Outcome:
        backend = self.serving[decision.model]
        msgs = list(self.messages if messages is None else messages)
        text, usage = backend.run(decision.model, msgs)
        usage = Usage.of(usage)
        cost = turn_cost(decision.model, usage.prompt_tokens, usage.cached_tokens, usage.output_tokens,
                         {decision.model: backend.prices(decision.model)})
        return Outcome(decision.model, text, usage, cost, last_user(msgs))

    def observe(self, outcome: Outcome) -> SessionState:
        turn, d = self._pending if self._pending else (None, None)
        query = outcome.query if outcome.query is not None else (turn.query if turn else "")
        usage = Usage.of(outcome.usage)
        cost = outcome.cost
        if cost is None:
            backend = self.serving[outcome.model]
            cost = turn_cost(outcome.model, usage.prompt_tokens, usage.cached_tokens, usage.output_tokens,
                             {outcome.model: backend.prices(outcome.model)})
        t = len(self.state.history)
        self.state = self.state.after(query, outcome.text, outcome.model, usage.prompt_tokens, usage.output_tokens)
        self.total += cost
        if self.encoder is not None and f"{self.chat_id}#{t + 1}" in self._emb:
            self._emb[f"{self.chat_id}#{t + 1}"]["a"] = self.encoder([outcome.text])[0]
        if self.log is not None:
            self.log.write({"chat_id": self.chat_id, "turn": t + 1, "model": outcome.model,
                            "n_prompt_tokens": usage.prompt_tokens, "n_cached_tokens": usage.cached_tokens,
                            "n_output_tokens": usage.output_tokens, "cost": cost,
                            "quality_hat": d.quality if d else None, "cost_hat": d.cost if d else None,
                            "time": round(time.time(), 3)})
        self._pending = None
        return self.state

    def chat(self, msg: str) -> Reply:
        """Route, run and observe one user message."""
        with self.lock:
            self.messages.append({"role": "user", "content": msg})
            d = self.route()
            o = self.run(d)
            self.observe(o)
            self.messages.append({"role": "assistant", "content": o.text})
            return Reply(self.chat_id, d.t + 1, o.model, o.text, o.cost, o.usage)

    def summary(self) -> dict:
        return {"chat_id": self.chat_id, "turns": len(self.state.history), "total_cost": self.total,
                "kv_cache": dict(self.state.kv_cache.held), "models": [x.model for x in self.state.history]}


CONFIG_KEYS = {"router", "serving", "models", "log", "lam", "checkpoint", "encoder", "backends", "plugins"}


class Runtime:
    """The router, the backends (model -> backend), the encoder, the log and the sessions by chat id."""

    def __init__(self, router, backends: list, encoder=None, log=None):
        self.router, self.encoder = router, encoder
        self.serving = {}
        for b in backends:
            for m in b.models:
                if m in self.serving:
                    raise ValueError(f"model {m!r} is served by both {self.serving[m].name} and {b.name}")
                self.serving[m] = b
        self.log = TurnLog(log) if log else None
        self.sessions: dict = {}
        self.lock = threading.Lock()

    @property
    def models(self) -> list:
        return list(self.serving)

    def session(self, chat_id: str) -> Session:
        with self.lock:
            if chat_id not in self.sessions:
                self.sessions[chat_id] = Session(chat_id, self.router, self.serving, self.encoder, self.log)
            return self.sessions[chat_id]

    @classmethod
    def from_config(cls, config, **overrides) -> "Runtime":
        """From a serve config (a YAML path or a dict): router, serving, models, log; optionally lam, checkpoint,
        encoder, backends (per-backend options) and plugins (.py files). routers/*.py and backends/*.py next to the
        config file are loaded too, so `router: my-router` finds routers/my_router.py."""
        import yaml
        from .registry import get_router, get_serving, load_plugin
        from .router import load_router
        base = None
        if isinstance(config, (str, Path)):
            base = Path(config).resolve().parent
            config = yaml.safe_load(Path(config).read_text()) or {}
        cfg = {**config, **{k: v for k, v in overrides.items() if v is not None}}
        bad = sorted(set(cfg) - CONFIG_KEYS)
        if bad:
            raise ValueError(f"unknown config key(s) {bad}; allowed: {sorted(CONFIG_KEYS)}")
        folder = base or Path.cwd()
        for sub in ("routers", "backends"):
            for f in sorted((folder / sub).glob("*.py")) if (folder / sub).is_dir() else []:
                load_plugin(str(f))
        for f in cfg.get("plugins") or []:
            load_plugin(str(folder / f) if not os.path.isabs(f) else f)
        spec = str(cfg["router"])
        if spec.endswith(".py") and not os.path.isabs(spec):
            spec = str(folder / spec)
        rcls = get_router(spec)
        if rcls.trained:
            ckpt = cfg.get("checkpoint") or os.path.join(os.environ.get("SRX_RUNS", "runs"), rcls.name)
            if not (Path(ckpt) / "router.pt").exists():
                raise FileNotFoundError(f"router {rcls.name!r} needs training first: no {ckpt}/router.pt "
                                        f"(sessionroutex train --router {rcls.name} --out {ckpt})")
            router = load_router(ckpt, spec)
        else:
            router = rcls()
        if cfg.get("lam") is not None:
            router.lam = float(cfg["lam"])
        names = cfg["serving"] if isinstance(cfg["serving"], list) else [cfg["serving"]]
        backends = []
        for name in names:
            bcls = get_serving(name)
            models = (cfg.get("models") or {}).get(name)
            backends.append(bcls(models=models, **((cfg.get("backends") or {}).get(name) or {})))
        enc = cfg.get("encoder", "granite" if getattr(router, "uses_embeddings", False) else None)
        dim = getattr(router, "emb_dim", None)
        return cls(router, backends, make_encoder(enc, dim), cfg.get("log"))
