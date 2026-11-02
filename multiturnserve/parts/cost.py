"""Cost parts: the estimated cost of every candidate model for a turn, as a router weighs it.

Each part returns Costs(values, scale, models, usd); Router.pick divides values by scale. Output lengths are the
predicted ones (turn.out_tokens); prompt lengths are turn.n_prompt; prices are turn.prices (USD per 1M tokens).
"""
from ..pricing import static_cost, turn_cost
from ..types import Costs


def _base(turn, oh) -> float:
    """The most expensive model's full-price estimate: the unit of normalized cost."""
    return max(static_cost(a, turn.n_prompt[a], oh[k], turn.prices) for k, a in enumerate(turn.models))


class CacheCost:
    """Prompt-cache-aware cost (CoR's Serving State Tracker): a model that already holds part of this conversation's
    prompt in its KV cache is charged the cached price for that part."""

    def __call__(self, turn, kv_cache, out=None) -> Costs:
        oh = turn.out_tokens if out is None else out
        base_hat = _base(turn, oh)
        est = [turn_cost(a, turn.n_prompt[a], kv_cache.cached(a, turn.n_prompt[a]), oh[k], turn.prices)
               for k, a in enumerate(turn.models)]
        return Costs(est, base_hat, tuple(turn.models), est)


class StaticCost:
    """Every prompt token at the full input price (the KV cache is ignored)."""

    def __call__(self, turn, kv_cache=None, out=None) -> Costs:
        oh = turn.out_tokens if out is None else out
        base_hat = _base(turn, oh)
        est = [static_cost(a, turn.n_prompt[a], oh[k], turn.prices) for k, a in enumerate(turn.models)]
        return Costs(est, base_hat, tuple(turn.models), est)


class SwitchCost:
    """Full input price, normalized by the most expensive model, plus a fixed penalty (in normalized-cost units) for
    any model other than prev, the one that served the previous turn (Bao et al., simplified)."""

    def __init__(self, penalty: float = 0.1):
        self.penalty = penalty

    def __call__(self, turn, kv_cache=None, prev: str | None = None, out=None) -> Costs:
        oh = turn.out_tokens if out is None else out
        base_hat = _base(turn, oh)
        vals, usd = [], []
        for k, a in enumerate(turn.models):
            usd.append(static_cost(a, turn.n_prompt[a], oh[k], turn.prices))
            c_hat = usd[-1] / base_hat
            if prev is not None and a != prev:
                c_hat += self.penalty
            vals.append(c_hat)
        return Costs(vals, 1.0, tuple(turn.models), usd)
