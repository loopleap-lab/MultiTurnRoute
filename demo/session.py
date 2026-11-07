"""One multi-turn session through MultiTurnRoute, turn by turn, with no GPU and no API keys.

The router, backends and cost weight are the ones in examples/config.yaml (examples/my_router.py on the echo
backend and a stand-in backend, lam 0.2), so the numbers are those of the echo backend: it answers with the last
message and caches the session's prefix like a real serving engine. Unlike the config, it writes no turn log.

    python -m demo.session
"""
import yaml

import multiturnserve

TURNS = [
    "Summarize the Chain-of-Route paper in three sentences.",
    "Shorter, please.",
    "How does it estimate the cost of each model?",
    "Compare it with RouteLLM on multi-turn sessions.",
    "Thanks!",
]

cfg = yaml.safe_load(open("examples/config.yaml"))
cfg["plugins"] = ["examples/" + p for p in cfg["plugins"]]
cfg["log"] = None                       # no turn log: the demo writes no files
s = multiturnserve.open("demo-session", config=cfg)
print(f"{'turn':<5}{'model':<16}{'prompt':>7}{'cached':>8}{'cost (USD)':>13}  query")
total = 0.0
for q in TURNS:
    r = s.chat(q)                       # route, run, observe
    total += r.cost
    print(f"{r.turn:<5}{r.model:<16}{r.usage.prompt_tokens:>7}{r.usage.cached_tokens:>8}{r.cost:>13.2e}  {q}")
print(f"{'':<5}{'session total':<31}{total:>13.2e}")
