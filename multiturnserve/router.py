"""The router interface: decide(turn, state) -> the name of the model that answers this turn.

The same decide() runs in offline replay (the state follows the recorded token counts) and online (the state follows
the usage the serving backend reports). pick(q, c) is the decision rule of every built-in router:
argmax over models of (1 - lam) * q - lam * c, with c normalized by the most expensive model; ties go to the model
listed first. lam weighs cost against quality; replay sweeps it, serving uses Router.lam.
"""
from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

from .types import Costs, Decision

_PICKED = threading.local()


class Router:
    name: str | None = None
    label: str | None = None        # name in results.json and the leaderboard (default: name)
    lam: float = 0.5
    encoder_overhead = True         # replay charges the frozen encoder's input tokens per turn, as for learned routers
    trained = False                 # needs fit() or a saved checkpoint before it can route
    uses_embeddings = False         # reads turn.emb (online: served with an encoder)
    quality = None                  # history -> predicted quality of every model
    cost = None                     # KV cache -> estimated cost of every model
    length = None                   # predicted output tokens (parts.LengthHead), set by training
    models: list | None = None      # the models it was trained on

    def decide(self, turn, state) -> str:
        q = self.quality(turn, state.history)
        c = self.cost(turn, state.kv_cache)
        return self.pick(q, c)

    def pick(self, q, c: Costs) -> str:
        lam = self.lam
        util = [(1 - lam) * q[k] - lam * c.values[k] / c.scale for k in range(len(c.models))]
        k = int(np.argmax(util))
        _PICKED.last = (q, c, k)
        return c.models[k]

    def route(self, turn, state) -> Decision:
        """decide(), with the predicted quality and estimated cost of the chosen model when pick() made the choice."""
        _PICKED.last = None
        model = self.decide(turn, state)
        last, quality, cost = getattr(_PICKED, "last", None), None, None
        if last is not None and last[1].models[last[2]] == model:
            q, c, k = last
            quality = float(q[k])
            cost = float(c.usd[k]) if c.usd is not None else None
        return Decision(model, quality, cost, turn.t)

    @property
    def display(self) -> str:
        return self.label or self.name

    @property
    def emb_dim(self) -> int | None:
        """Dimension of the turn embeddings it was trained on (from the length head)."""
        return None if self.length is None else int(np.asarray(self.length.W).shape[0] - 1)

    # training and replay
    def fit(self, ctx) -> None:
        """Train on ctx.train (early stopping on ctx.val); see arena.train.Context. Default: nothing to train."""

    def prepare(self, convs: list, emb: dict) -> None:
        """Before a replay: predict whole recorded conversations once (quality parts that support it)."""
        if hasattr(self.quality, "prepare") and getattr(self.quality, "predict", None) is not None:
            self.quality.prepare(convs, emb)

    def predictors(self, emb: dict) -> dict:
        """name -> predict(conv) [T, K], for the quality-error report of a replay."""
        return {}

    # checkpoints
    def state(self) -> dict:
        """Everything a trained router needs to route again (tensors and plain values)."""
        return {}

    def load_state(self, state: dict) -> None:
        pass

    def save(self, path) -> Path:
        import torch
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        blob = {"router": self.name, "models": list(self.models or []), "lam": float(self.lam), "state": self.state(),
                "length": None if self.length is None else torch.from_numpy(np.asarray(self.length.W))}
        torch.save(blob, path / "router.pt")
        return path / "router.pt"


def load_router(path, spec: str | None = None) -> Router:
    """A trained router from a folder written by Router.save (multiturnserve train). spec: the router name or plugin
    file, when the checkpoint comes from a plugin."""
    import torch
    from .parts.length import LengthHead
    from .registry import get_router
    blob = torch.load(Path(path) / "router.pt", weights_only=True)
    r = get_router(spec or blob["router"])()
    r.models, r.lam = blob["models"], blob["lam"]
    if blob["length"] is not None:
        r.length = LengthHead(blob["length"].numpy(), r.models)
    r.load_state(blob["state"])
    return r
