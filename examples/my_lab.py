"""Stand-ins for "your code" in my_router.py and my_backend.py, so the examples run anywhere."""
from types import SimpleNamespace

import numpy as np

from sessionroutex.pricing import PRICES
from sessionroutex.serving.echo import Echo

STRENGTH = {"gemma-e4b": 0.55, "gemma-26b-a4b": 0.68, "gemma-31b": 0.70, "mistral-medium-128b": 0.72}


class MyQuality:
    """A toy quality estimate: each model's base strength, plus a bonus for larger models once the chat has a
    history to keep track of."""

    def __call__(self, turn, history):
        bonus = 0.02 * min(len(history), 5)
        return np.array([STRENGTH.get(m, 0.5) + (bonus if STRENGTH.get(m, 0.5) > 0.6 else 0.0) for m in turn.models])


class _Client:
    """A model API client stand-in: the echo backend, wrapped as reply.text / reply.usage."""

    def __init__(self):
        self.echo = Echo()

    def chat(self, model, messages):
        text, usage = self.echo.run(model, messages)
        return SimpleNamespace(text=text, usage=usage)


my_client = _Client()
__all__ = ["MyQuality", "my_client", "PRICES"]
