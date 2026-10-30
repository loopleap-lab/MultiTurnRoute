"""Serving backends: a Serving runs one model call and reports usage (new and cached tokens) and prices."""
from .base import Serving
from . import echo, openrouter, vllm  # noqa: F401,E402  (registers the built-ins)

__all__ = ["Serving"]
