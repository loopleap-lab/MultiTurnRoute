"""The gateway end to end, through FastAPI's TestClient: echo backend, a scripted router and a trained CoR."""
import json

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

import sessionroutex as srx  # noqa: E402
from fakes import PRICES, FakeOpenAI, Scripted  # noqa: E402,F401
from sessionroutex.gateway import create_app  # noqa: E402
from synth import MODELS, write_dataset  # noqa: E402


def client(config, **kw):
    return TestClient(create_app(srx.Runtime.from_config(config, **kw)))


ECHO = {"router": "scripted-test", "serving": ["echo"], "models": {"echo": ["small", "big"]},
        "backends": {"echo": {"prices": PRICES}}}


def test_chat_completions_keep_one_session_per_chat(tmp_path):
    c = client({**ECHO, "log": str(tmp_path / "logs")})
    msgs = [{"role": "user", "content": "a b c"}]
    r = c.post("/v1/chat/completions", json={"chat_id": "x", "messages": msgs}).json()
    assert r["object"] == "chat.completion" and r["model"] == "small"
    assert r["choices"][0]["message"]["content"] == "echo: a b c"
    assert r["usage"] == {"prompt_tokens": 4, "completion_tokens": 4, "total_tokens": 8,
                          "prompt_tokens_details": {"cached_tokens": 0}}
    assert r["sessionroutex"]["turn"] == 1 and r["sessionroutex"]["cost"] == pytest.approx(12e-6)
    msgs += [{"role": "assistant", "content": "echo: a b c"}, {"role": "user", "content": "d e"}]
    r = c.post("/v1/chat/completions", json={"messages": msgs}, headers={"X-Chat-Id": "x"}).json()
    assert r["sessionroutex"]["turn"] == 2 and r["usage"]["prompt_tokens_details"]["cached_tokens"] == 8
    r = c.post("/v1/chat/completions", json={"user": "y", "messages": msgs[:1]}).json()    # another chat
    assert r["sessionroutex"]["turn"] == 1
    st = c.get("/v1/sessions/x").json()
    assert st["turns"] == 2 and st["kv_cache"] == {"small": 14} and st["models"] == ["small", "small"]
    assert c.get("/v1/sessions/nope").status_code == 404
    assert c.post("/v1/chat/completions", json={"messages": msgs}).status_code == 400
    assert c.get("/health").json() == {"router": "scripted-test", "models": ["small", "big"]}
    assert [m["id"] for m in c.get("/v1/models").json()["data"]] == ["small", "big"]
    assert len(open(tmp_path / "logs" / "turns.jsonl").readlines()) == 3


def test_route_and_observe_over_http():
    c = client(ECHO)
    s = srx.open("chat-1", gateway=c)
    d = s.route([{"role": "user", "content": "hello"}])
    assert (d.model, d.t) == ("small", 0)
    o = srx.Outcome(d.model, "hi there", srx.Usage(5, 0, 2), query="hello")
    st = s.observe(o)
    assert st["turns"] == 1 and st["kv_cache"] == {"small": 7}
    assert st["total_cost"] == pytest.approx((5 * 1.0 + 2 * 2.0) / 1e6)


def test_client_open_chat_against_the_gateway():
    c = client(ECHO)
    s = srx.open("chat-1", gateway=c)
    replies = [s.chat(m) for m in ["a b c", "d e", "f"]]
    assert [(r.turn, r.model) for r in replies] == [(1, "small"), (2, "small"), (3, "big")]
    assert replies[1].usage.cached_tokens == 8 and replies[2].text == "echo: f"
    assert s.state()["turns"] == 3


@pytest.fixture(scope="module")
def cor_ckpt(tmp_path_factory):
    from sessionroutex.cli import main
    root = write_dataset(tmp_path_factory.mktemp("data"))
    out = tmp_path_factory.mktemp("runs") / "cor"
    main(["train", "--router", "cor", "--data", str(root), "--models", *MODELS, "--epochs", "2", "--out", str(out)])
    return out


def test_trained_cor_routes_live_chats(cor_ckpt, tmp_path):
    """CoR from a checkpoint, live embeddings from a stand-in encoder, on a pool of two of its models."""
    cfg = {"router": "cor", "checkpoint": str(cor_ckpt), "serving": ["echo"], "models": {"echo": MODELS[1:3]},
           "encoder": "hash"}
    rt = srx.Runtime.from_config(cfg)
    assert rt.router.emb_dim == 16 and rt.encoder.dim == 16
    c = TestClient(create_app(rt))
    s = srx.open("chat-1", gateway=c)
    for m in ["first question here", "a follow-up", "and one more"]:
        r = s.chat(m)
        assert r.model in MODELS[1:3]
    st = c.get("/v1/sessions/chat-1").json()
    assert st["turns"] == 3 and sum(st["kv_cache"].values()) > 0
    d = rt.sessions["chat-1"].route([{"role": "user", "content": "x"}])
    assert 0.0 <= d.quality <= 1.0 and d.cost > 0


def test_untrained_router_says_how_to_train(tmp_path):
    with pytest.raises(FileNotFoundError, match="sessionroutex train --router cor"):
        srx.Runtime.from_config({"router": "cor", "checkpoint": str(tmp_path / "none"), "serving": ["echo"],
                                 "models": {"echo": ["gemma-31b"]}})
    with pytest.raises(ValueError, match="unknown config key"):
        srx.Runtime.from_config({**ECHO, "routr": "x"})


def test_vllm_backend_reads_cached_tokens_and_key_from_env(monkeypatch):
    fake = FakeOpenAI()
    monkeypatch.setenv("VLLM_API_KEY", "not-a-real-key")
    try:
        c = client({"router": "scripted-test", "serving": ["vllm"], "models": {"vllm": ["small", "big"]},
                    "backends": {"vllm": {"base_url": fake.url, "prices": PRICES,
                                          "model_ids": {"small": "org/small-model"}}}})
        msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"},
                {"role": "user", "content": "more"}]
        r = c.post("/v1/chat/completions", json={"chat_id": "x", "messages": msgs}).json()
        assert r["usage"]["prompt_tokens"] == 30 and r["usage"]["prompt_tokens_details"]["cached_tokens"] == 15
        assert r["sessionroutex"]["cost"] == pytest.approx((15 * 1.0 + 15 * 0.1 + 5 * 2.0) / 1e6)
        req = fake.requests[-1]
        assert req["path"] == "/v1/chat/completions" and req["body"]["model"] == "org/small-model"
        assert req["auth"] == "Bearer not-a-real-key"
    finally:
        fake.close()


def test_openrouter_asks_for_usage():
    from sessionroutex.serving.openrouter import OpenRouter
    b = OpenRouter(models=["gemma-31b"])
    assert b.base_url == "https://openrouter.ai/api/v1" and b.api_key_env == "OPENROUTER_API_KEY"
    assert b.extra == {"usage": {"include": True}} and b.prices("gemma-31b")["cached"] == 0.04
