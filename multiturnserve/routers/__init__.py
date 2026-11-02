"""Built-in routers, one package each; importing this module registers them."""
from . import bao, cor, full_history, knn, rag, single_turn  # noqa: F401

BUILTIN = ["cor", "single-turn", "full-history", "rag", "knn", "bao"]
