"""MultiTurnServe: a session layer for multi-turn LLM routing.

Each session keeps two states, the history and the KV cache each model holds for it; a router reads both to pick the
model for every turn, and the serving backend's usage keeps them up to date.
"""
from .registry import register_router, register_serving
from .router import Router, load_router
from .serving.base import Serving
from .session import Runtime, Session
from .types import Decision, Exchange, Outcome, Reply, SessionState, Turn, Usage

__version__ = "0.1.0"


def open(chat_id: str, gateway=None, config=None, runtime=None):  # noqa: A001  (multiturnserve.open, like a file)
    """A session. gateway=":8080" (or an HTTP client object): talk to a running `multiturnserve serve`.
    Otherwise a local session from a serve config (YAML path or dict) or an existing Runtime."""
    if gateway is not None:
        from .client import RemoteSession
        return RemoteSession(chat_id, gateway)
    if runtime is None:
        if config is None:
            raise ValueError("multiturnserve.open needs gateway=':8080', config='serve.yaml' or runtime=")
        runtime = Runtime.from_config(config)
    return runtime.session(chat_id)


__all__ = ["open", "register_router", "register_serving", "Router", "Serving", "Runtime", "Session", "load_router",
           "Turn", "SessionState", "Exchange", "Decision", "Outcome", "Reply", "Usage"]
