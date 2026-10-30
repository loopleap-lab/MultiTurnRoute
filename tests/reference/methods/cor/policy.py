"""CoR's cache-aware decision (Serving State Tracker): a model that already holds part of this conversation's prompt
in its KV cache is charged the cached price for that part when estimating the cost of the next turn."""
from dataclasses import dataclass

import numpy as np

from common.pricing import CacheState, turn_cost
from common.simulate import UtilityPolicy


@dataclass(frozen=True)
class CacheAwarePolicy(UtilityPolicy):
    def estimated_cost(self, turn, t: int, arms: list, state: CacheState, k: int, oh: np.ndarray) -> float:
        a = arms[k]
        return turn_cost(a, turn.n_prompt[a], state.cached(a, turn.n_prompt[a]), oh[t][k])
