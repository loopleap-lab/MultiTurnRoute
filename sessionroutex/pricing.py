"""Token pricing and the prompt-cache state of a conversation (used to account the realized cost of any routed path).

A model that served turn s holds the prefix up to the end of its own answer: n_prompt(s) + n_output(s) tokens.
At a later turn t its cached part of the prompt is min(that, n_prompt(t)); the rest is prefilled at full price.
Cost = input price * (L - cached) + cached price * cached + output price * L_gen   (USD; prices per 1M tokens).

PRICES is the static track's price table (arena/static.yaml). Online, each serving backend gives its own prices.
"""
from dataclasses import dataclass, field
from pathlib import Path

import yaml

STATIC = yaml.safe_load((Path(__file__).parent / "arena" / "static.yaml").read_text())
PRICES = STATIC["prices"]


def turn_cost(arm: str, n_prompt: int, n_cached: int, n_output: float, prices: dict | None = None) -> float:
    p = (PRICES if prices is None else prices)[arm]
    cached = min(n_cached, n_prompt)
    return ((n_prompt - cached) * p["input"] + cached * p["cached"] + n_output * p["output"]) / 1e6


def static_cost(arm: str, n_prompt: int, n_output: float, prices: dict | None = None) -> float:
    """Every prompt token at the full input price (no prompt caching)."""
    return turn_cost(arm, n_prompt, 0, n_output, prices)


@dataclass
class CacheState:
    """Tokens each model currently holds for this conversation."""
    held: dict = field(default_factory=dict)

    def cached(self, arm: str, n_prompt: int) -> int:
        return min(self.held.get(arm, 0), n_prompt)

    def after_serving(self, arm: str, n_prompt: int, n_output: int) -> "CacheState":
        """New state after `arm` answered (immutable update; other models keep what they held)."""
        return CacheState({**self.held, arm: n_prompt + n_output})
