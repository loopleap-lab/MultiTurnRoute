"""What routers, sessions and serving backends pass to each other."""
from __future__ import annotations

from dataclasses import dataclass, field

from .pricing import CacheState


@dataclass(frozen=True)
class Usage:
    """Tokens of one model call: the prompt it read (of which cached), and what it wrote."""
    prompt_tokens: int
    cached_tokens: int = 0
    output_tokens: int = 0

    @classmethod
    def of(cls, u) -> "Usage":
        """A Usage from a Usage, a tuple (prompt, cached, output) or a dict (ours or OpenAI's usage field)."""
        if isinstance(u, Usage):
            return u
        if isinstance(u, (tuple, list)):
            return cls(*(int(x) for x in u))
        if "prompt_tokens_details" in u or "completion_tokens" in u:
            cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
            return cls(int(u["prompt_tokens"]), int(cached), int(u.get("completion_tokens") or 0))
        return cls(int(u["prompt_tokens"]), int(u.get("cached_tokens") or 0), int(u.get("output_tokens") or 0))


@dataclass(frozen=True)
class Turn:
    """One turn to route: what a router may read before any model answers.

    models, n_prompt, out_tokens and prices describe the candidates: n_prompt[model] is the prompt the model would
    read (recorded offline, estimated online), out_tokens[k] the predicted output tokens of models[k]. conv is the
    conversation so far (arena.data.Conversation; this turn is conv.turns[t]) and emb its turn embeddings
    (turn_id -> {"q", "a", "h"}), for routers that learn from them; emb is None online without an encoder."""
    chat_id: str
    t: int
    query: str
    models: tuple
    n_prompt: dict
    out_tokens: object
    prices: dict
    conv: object = None
    emb: dict | None = None
    messages: tuple = ()


@dataclass(frozen=True)
class Exchange:
    """One answered turn of the history."""
    query: str
    answer: str
    model: str


@dataclass(frozen=True)
class SessionState:
    """The two states of a session: the history (what was said, and by which model) and the KV cache (tokens each
    model holds for this session)."""
    history: tuple = ()
    kv_cache: CacheState = field(default_factory=CacheState)

    def after(self, query: str, answer: str, model: str, n_prompt: int, n_output: int) -> "SessionState":
        """New state after `model` answered (immutable update)."""
        return SessionState(self.history + (Exchange(query, answer, model),),
                            self.kv_cache.after_serving(model, n_prompt, n_output))


@dataclass(frozen=True)
class Costs:
    """Estimated cost of every model for one turn, as a router weighs it: values[k] / scale for models[k].
    usd[k] is the estimate in USD, when the part knows it."""
    values: list
    scale: float
    models: tuple
    usd: list | None = None


@dataclass(frozen=True)
class Decision:
    """The model a router picked for turn t, with the predicted quality and estimated USD cost of that model
    (the estimate uses a predicted output length)."""
    model: str
    quality: float | None = None
    cost: float | None = None
    t: int = 0


@dataclass(frozen=True)
class Outcome:
    """What the chosen model returned: its answer, its usage and the realized cost (USD)."""
    model: str
    text: str
    usage: Usage
    cost: float | None = None
    query: str | None = None


@dataclass(frozen=True)
class Reply:
    """One turn of a session as the app sees it (turn counts from 1)."""
    chat_id: str
    turn: int
    model: str
    text: str
    cost: float
    usage: Usage
