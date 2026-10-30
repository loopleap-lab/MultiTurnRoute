"""Session state and pricing: after several turns the KV cache, the cost and the log equal hand-computed values."""
import json

import numpy as np
import pytest

import sessionroutex as srx
from fakes import PRICES, Scripted
from sessionroutex.pricing import CacheState, static_cost, turn_cost
from sessionroutex.serving.echo import Echo


def runtime(tmp_path=None):
    return srx.Runtime(Scripted(), [Echo(models=["small", "big"], prices=PRICES)], log=tmp_path)


def test_kv_cache_and_cost_by_hand(tmp_path):
    s = srx.open("chat-1", runtime=runtime(tmp_path))
    replies = [s.chat(m) for m in ["a b c", "d e", "f", "g h i j"]]
    # echo: prompt tokens = words of every message + one <user> per user message; reply = "echo: <last user msg>"
    # turn 1, small: prompt <user> a b c = 4, cached 0, output "echo: a b c" = 4
    # turn 2, small: prompt 4 + 4 + 3 = 11, cached 8 (its turn-1 prompt + reply), output 3
    # turn 3, big:   prompt 11 + 3 + 2 = 16, cached 0 (big holds nothing), output 2
    # turn 4, small: prompt 16 + 2 + 5 = 23, cached 14 (its turn-2 prompt + reply), output 5
    assert [r.model for r in replies] == ["small", "small", "big", "small"]
    assert [r.turn for r in replies] == [1, 2, 3, 4]
    assert [(r.usage.prompt_tokens, r.usage.cached_tokens, r.usage.output_tokens) for r in replies] == \
        [(4, 0, 4), (11, 8, 3), (16, 0, 2), (23, 14, 5)]
    by_hand = [(4 * 1.0 + 4 * 2.0) / 1e6, (3 * 1.0 + 8 * 0.1 + 3 * 2.0) / 1e6, (16 * 10.0 + 2 * 30.0) / 1e6,
               (9 * 1.0 + 14 * 0.1 + 5 * 2.0) / 1e6]
    assert [r.cost for r in replies] == pytest.approx(by_hand, rel=1e-12)
    assert s.state.kv_cache.held == {"small": 23 + 5, "big": 16 + 2}
    assert s.total == pytest.approx(sum(by_hand), rel=1e-12)
    assert [x.model for x in s.state.history] == ["small", "small", "big", "small"]
    assert s.state.history[1].query == "d e" and s.state.history[1].answer == "echo: d e"
    rows = [json.loads(x) for x in open(tmp_path / "turns.jsonl")]
    assert [(r["turn"], r["model"], r["n_cached_tokens"]) for r in rows] == [(1, "small", 0), (2, "small", 8),
                                                                            (3, "big", 0), (4, "small", 14)]
    assert "a b c" not in (tmp_path / "turns.jsonl").read_text()     # the log has no message texts


def test_cache_cost_estimate_reads_the_kv_cache():
    """Before turn 3 the router's estimate charges small's held prefix at the cached price, big's at full price."""
    s = srx.open("chat-2", runtime=runtime())
    s.chat("a b c"), s.chat("d e")
    s.messages.append({"role": "user", "content": "f"})
    turn = s._turn(s.messages)
    n = turn.n_prompt["small"]
    assert n == round(len("a b c" + "echo: a b c" + "d e" + "echo: d e" + "f") / 4)
    c = Scripted.cost(turn, s.state.kv_cache)
    held = s.state.kv_cache.held["small"]
    assert c.usd[0] == pytest.approx(((n - min(held, n)) * 1.0 + min(held, n) * 0.1 + 256 * 2.0) / 1e6)
    assert c.usd[1] == pytest.approx((n * 10.0 + 256 * 30.0) / 1e6) and c.scale == c.usd[1]
    d = s.route()
    assert d.model == "big" and d.t == 2 and d.quality == 1.0 and d.cost == pytest.approx(c.usd[1])


def test_cache_state_is_immutable_and_capped_by_the_prompt():
    s0 = CacheState()
    s1 = s0.after_serving("gemma-31b", 500, 50)
    assert s0.held == {} and s1.cached("gemma-31b", 900) == 550 and s1.cached("gemma-31b", 300) == 300
    assert turn_cost("mistral-medium-128b", 10000, 9000, 100) < static_cost("mistral-medium-128b", 10000, 100)
    assert turn_cost("small", 100, 500, 0, PRICES) == pytest.approx(100 * 0.1 / 1e6)   # cached <= prompt


def test_session_state_after_and_usage_formats():
    st = srx.SessionState().after("q", "a", "m", 10, 5).after("q2", "a2", "n", 20, 3)
    assert st.kv_cache.held == {"m": 15, "n": 23} and [x.model for x in st.history] == ["m", "n"]
    assert srx.Usage.of({"prompt_tokens": 9, "completion_tokens": 2, "prompt_tokens_details": {"cached_tokens": 4}}) \
        == srx.Usage(9, 4, 2)
    assert srx.Usage.of({"prompt_tokens": 9, "cached_tokens": 4, "output_tokens": 2}) == srx.Usage(9, 4, 2)
    assert srx.Usage.of((9, 4, 2)) == srx.Usage(9, 4, 2)


def test_pick_ties_go_to_the_first_model_and_lam_trades_cost():
    from sessionroutex.types import Costs
    r = srx.Router()
    c = Costs([1.0, 3.0], 3.0, ("a", "b"), [1.0, 3.0])
    r.lam = 0.0
    assert r.pick(np.array([0.5, 0.5]), c) == "a"
    assert r.pick(np.array([0.4, 0.6]), c) == "b"
    r.lam = 0.9
    assert r.pick(np.array([0.4, 0.6]), c) == "a"
