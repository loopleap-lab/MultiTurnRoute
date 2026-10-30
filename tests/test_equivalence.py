"""The ported code gives the evaluation package's numbers: on synthetic data with fixed seeds, `arena replay` and
the original evaluate.py agree on every per-turn choice, AUDC, QNC, QNC_hull, peak, curve and bootstrap.
The original code is a test fixture in tests/reference/ (not shipped with the package)."""
import importlib
import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

from synth import MODELS, write_dataset

REF = Path(__file__).parent / "reference"
ROUTERS = ["bao", "cor", "full-history", "knn", "rag", "single-turn"]


def args(**kw):
    a = dict(folds=1, epochs=3, window=2, seed=0, rank_w=0.0, dsp=1, dsp_p=8, dsp_drop=0.0, bootstrap=20,
             n_models=1, loss="mse", switch_penalty=0.1, dev=False,
             methods=["bao", "cor", "full_history", "knn", "rag", "single_turn"])
    a.update(kw)
    return types.SimpleNamespace(**a)


@pytest.fixture(scope="module")
def ref(tmp_path_factory):
    """The evaluation package, reading a synthetic data folder (it reads MTR_DATA_DIR at import)."""
    root = write_dataset(tmp_path_factory.mktemp("data"), n_conv=20)
    os.environ["MTR_DATA_DIR"] = str(root)
    for m in [m for m in sys.modules if m.split(".")[0] in ("common", "methods", "evaluate")]:
        del sys.modules[m]
    sys.path.insert(0, str(REF))
    try:
        mods = types.SimpleNamespace(evaluate=importlib.import_module("evaluate"),
                                     data=importlib.import_module("common.data"),
                                     embeddings=importlib.import_module("common.embeddings"),
                                     simulate=importlib.import_module("common.simulate"),
                                     nn=importlib.import_module("common.nn"), root=root)
        yield mods
    finally:
        sys.path.remove(str(REF))
        os.environ.pop("MTR_DATA_DIR", None)


def same(a, b) -> bool:
    """Equal, NaN included (the bootstrap's QNC interval can be [x, nan] in both)."""
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


@pytest.mark.parametrize("folds", [1, 2])
def test_replay_cli_matches_evaluate(ref, tmp_path, folds):
    from sessionroutex.cli import main
    torch.set_num_threads(4)
    ref.nn.LOSS["kind"] = "mse"
    convs, _ = ref.data.load("chat", MODELS)
    emb = ref.embeddings.load("chat")
    want = ref.evaluate.evaluate(convs, emb, MODELS, args(folds=folds))
    main(["arena", "replay", "--routers", *ROUTERS, "--data", str(ref.root), "--models", *MODELS, "--epochs", "3",
          "--dsp-p", "8", "--dsp-drop", "0", "--bootstrap", "20", "--folds", str(folds), "--out", str(tmp_path),
          "--name", "x"])
    got = json.loads((tmp_path / "x" / "results.json").read_text())
    for k in ("n_test_turns", "fixed_models", "best_single_model", "policies", "quality_mse_per_split", "bootstrap"):
        assert same(got[k], json.loads(json.dumps(want[k]))), k
    assert len(want["policies"]) == 7 and got["models"] == MODELS
    for p, r in want["policies"].items():
        assert (got["policies"][p]["AUDC"], got["policies"][p]["QNC"], got["policies"][p]["QNC_hull"],
                got["policies"][p]["peak_quality"]) == (r["AUDC"], r["QNC"], r["QNC_hull"], r["peak_quality"])


def test_per_turn_choices_match(ref):
    from sessionroutex.arena import data, embeddings, replay
    torch.set_num_threads(4)
    ref.nn.LOSS["kind"] = "mse"
    a = args()
    convs, _ = ref.data.load("chat", MODELS)
    emb = ref.embeddings.load("chat")
    train_all = [c for c in convs if c.split == "train"]
    test = [c for c in convs if c.split == "test"]
    policies, _ = ref.evaluate.build_policies(train_all, test, emb, MODELS, a)

    convs2, _ = data.load("chat", MODELS, data=ref.root)
    emb2 = embeddings.load("chat", data=ref.root)
    keep = []
    replay.evaluate(convs2, emb2, MODELS, a, ROUTERS, keep=keep)
    ctx, routers = keep[0]
    routers = {r.display: r for r in routers + [replay.OracleRouter(ctx.test, MODELS)]}
    assert set(routers) == {p.name for p in policies}
    n = 0
    for pol in policies:
        r = routers[pol.name]
        for lam in replay.LAMBDAS:
            for c_ref, c in zip(test, ctx.test):
                want = ref.simulate.run_conversation(c_ref, MODELS, pol, lam)
                got = replay.run_conversation(c, MODELS, r, lam, ctx.out_hat, ctx.overhead, emb2)
                assert got == want, (pol.name, float(lam), c.conv_id)
                n += len(got)
    assert n == 7 * 41 * sum(len(c.turns) for c in test)
    assert len({k for pol in policies for c in test for k, *_ in ref.simulate.run_conversation(c, MODELS, pol, 0.5)}) > 1


def test_metrics_and_bootstrap_are_the_package_ones(ref):
    """Ported unit tests of the package: QNC_hull interpolation and pairwise bootstrap."""
    from sessionroutex.arena import replay
    pts = [(0.0, 0.1), (0.2, 0.5), (0.1, 0.2)]
    assert replay.qnc_hull(pts, 0.35) == ref.simulate.qnc_hull(pts, 0.35)
    assert abs(replay.qnc_hull(pts, 0.35) - 0.15) < 1e-9
    assert replay.audc(pts) == ref.simulate.audc(pts) and replay.qnc(pts, 0.2) == ref.simulate.qnc(pts, 0.2)
    good = np.zeros((3, 20, 3)); good[..., 0], good[..., 1], good[..., 2] = 0.9, 0.1, 1
    bad = good.copy(); bad[..., 0] = 0.5
    rep = replay.bootstrap({"a": good, "b": bad}, np.tile([0.7, 0.2, 1.0], (20, 1)), n=30)
    assert same(rep, ref.simulate.bootstrap({"a": good, "b": bad}, np.tile([0.7, 0.2, 1.0], (20, 1)), n=30))
    assert rep["AUDC_win"]["a"]["b"] == 1.0 and rep["AUDC_win"]["b"]["a"] == 0.0
