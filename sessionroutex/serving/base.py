"""The serving interface: run(model, messages) -> (text, usage) and prices(model)."""
from __future__ import annotations

from ..pricing import PRICES


class Serving:
    """A backend that serves some models. usage: Usage(prompt_tokens, cached_tokens, output_tokens), or a dict
    with those keys, or OpenAI's usage field. prices(model): USD per 1M tokens, {"input", "cached", "output"}."""
    name: str | None = None
    models: list = []

    def __init__(self, models: list | None = None, prices: dict | None = None, **options):
        if models is not None:
            self.models = list(models)
        self._prices = dict(prices or {})
        if options:
            raise TypeError(f"serving backend {self.name!r} takes no option(s) {sorted(options)}")

    def run(self, model: str, messages: list):
        raise NotImplementedError

    def prices(self, model: str) -> dict:
        """The backend's own prices if given, else the static track's price table."""
        p = self._prices.get(model) or PRICES.get(model)
        if p is None:
            raise KeyError(f"no prices for model {model!r}: give them in the config (backends: {{{self.name}: "
                           f"{{prices: {{{model}: {{input: ..., cached: ..., output: ...}}}}}}}})")
        return p
