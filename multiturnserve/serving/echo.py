"""An offline backend for tests and dry runs: it repeats the last user message and keeps a prefix cache.

Tokens are whitespace-separated words; a user message adds a "<user>" token before its words. A model's cache holds
every prompt it read followed by its reply, so the cached part of a new prompt is the longest prefix it shares with
one of them: n_prompt + n_output of the model's last turn in the same chat, as pricing.CacheState assumes."""
from ..registry import register_serving
from ..types import Usage
from .base import Serving


def tokens(messages: list) -> list:
    out = []
    for m in messages:
        out += (["<user>"] if m["role"] == "user" else []) + str(m["content"]).split()
    return out


def _shared(a: list, b: list) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


@register_serving
class Echo(Serving):
    name = "echo"

    def __init__(self, models: list | None = None, prices: dict | None = None):
        super().__init__(models, prices)
        self.cache: dict = {}   # model -> [token lists]

    def run(self, model: str, messages: list):
        prompt = tokens(messages)
        last = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        text = f"echo: {last}"
        held = self.cache.setdefault(model, [])
        cached = max((_shared(prompt, h) for h in held), default=0)
        held.append(prompt + text.split())
        return text, Usage(len(prompt), cached, len(text.split()))
