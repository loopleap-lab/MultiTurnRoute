"""Every built-in router trains on synthetic data, saves, loads back with the same predictions and routes a turn."""
import types

import numpy as np
import pytest

from sessionroutex import load_router
from sessionroutex.arena import data, embeddings
from sessionroutex.arena.replay import replay_turns, run_conversation
from sessionroutex.arena.train import build_routers, make_context
from sessionroutex.routers import BUILTIN
from sessionroutex.types import SessionState
from synth import MODELS, write_dataset


@pytest.fixture(scope="module")
def ctx(tmp_path_factory):
    root = write_dataset(tmp_path_factory.mktemp("data"))
    convs, models = data.load("chat", MODELS, data=root)
    emb = embeddings.load("chat", data=root)
    args = types.SimpleNamespace(epochs=2, seed=0, n_models=2, rank_w=0.0, loss="mse", window=2, dsp=1, dsp_p=8,
                                 dsp_drop=0.0, switch_penalty=0.1)
    return make_context([c for c in convs if c.split == "train"], [c for c in convs if c.split == "test"], emb,
                        models, args)


@pytest.mark.parametrize("name", BUILTIN)
def test_save_load_round_trip(ctx, tmp_path, name):
    r = build_routers([name], ctx)[0]
    r.save(tmp_path / name)
    back = load_router(tmp_path / name)
    assert type(back) is type(r) and back.models == MODELS and back.emb_dim == 16
    for c in ctx.test:
        assert np.array_equal(back.quality.predict(c, ctx.emb), r.quality.predict(c, ctx.emb))
    c = ctx.test[0]
    assert run_conversation(c, MODELS, back, 0.5, ctx.out_hat, ctx.overhead, ctx.emb) == \
        run_conversation(c, MODELS, r, 0.5, ctx.out_hat, ctx.overhead, ctx.emb)


def test_quality_on_a_smaller_pool_follows_the_pool(ctx):
    r = build_routers(["cor"], ctx)[0]
    c = ctx.test[0]
    full = r.quality(replay_turns(c, MODELS, ctx.out_hat, ctx.emb)[1], ())
    pool = [MODELS[2], MODELS[0]]
    turn = replay_turns(c, pool, {c.conv_id: ctx.out_hat[c.conv_id][:, [2, 0]]}, ctx.emb)[1]
    assert np.array_equal(r.quality(turn, ()), full[[2, 0]])
    assert r.decide(turn, SessionState()) in pool


def test_bao_large_penalty_never_switches(ctx):
    r = build_routers(["bao"], ctx)[0]
    r.cost.penalty = 100.0
    for c in ctx.test:
        assert len({k for k, *_ in run_conversation(c, MODELS, r, 0.5, ctx.out_hat, None, ctx.emb)}) == 1
