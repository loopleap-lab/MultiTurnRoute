"""Smoke tests of the command line and the shipped configs."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from synth import MODELS, write_dataset

ROOT = Path(__file__).resolve().parents[1]


def cli(*args, env=None, monkeypatch=None, capsys=None):
    """Run `multiturnserve ARGS` in this process; returns (exit code, stdout, stderr)."""
    from multiturnserve.cli import main
    for k, v in (env or {}).items():
        monkeypatch.setenv(k, v)
    code = 0
    try:
        main(list(args))
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
        if not isinstance(e.code, int):
            print(e.code, file=sys.stderr)
    out, err = capsys.readouterr()
    return code, out, err


def test_entry_point_and_help():
    r = subprocess.run([sys.executable, "-m", "multiturnserve.cli", "arena", "--help"], capture_output=True,
                       text=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""}, cwd=ROOT)
    assert r.returncode == 0
    for c in ("replay", "leaderboard", "release", "run", "judge", "score", "submit"):
        assert c in r.stdout


def test_help_lists_every_command(capsys):
    from multiturnserve.cli import parser
    text = parser().format_help()
    for c in ("serve", "train", "list", "arena"):
        assert c in text


def test_list(capsys):
    code, out, _ = cli("list", capsys=capsys)
    assert code == 0
    routers, serving = out.splitlines()
    assert all(f" {n}" in routers for n in ("bao", "cor", "full-history", "knn", "rag", "single-turn"))
    assert all(f" {n}" in serving for n in ("echo", "openrouter", "vllm"))


def test_serve_config_is_the_websites():
    cfg = yaml.safe_load((ROOT / "configs" / "serve.yaml").read_text())
    assert cfg == {"router": "cor", "serving": ["vllm", "openrouter"],
                   "models": {"vllm": ["gemma-e4b", "gemma-26b-a4b"],
                              "openrouter": ["gemma-31b", "mistral-medium-128b"]}, "log": "logs/"}
    assert "# router: cor, single-turn, full-history, rag, knn, bao, or your own" in \
        (ROOT / "configs" / "serve.yaml").read_text()


def test_static_config_is_the_packaged_one():
    from multiturnserve.pricing import STATIC
    assert yaml.safe_load((ROOT / "configs" / "arena" / "static.yaml").read_text()) == STATIC
    from multiturnserve.arena.replay import LAMBDAS
    lam = STATIC["lambdas"]
    assert len(LAMBDAS) == lam["n_kappa"] + 1 == 41 and len(STATIC["models"]) == 9


def test_serve_check(tmp_path, monkeypatch, capsys):
    code, out, err = cli("serve", str(ROOT / "examples" / "config.yaml"), "--check", "--log", str(tmp_path),
                         capsys=capsys)
    assert code == 0, err
    assert "my-router" in out and "gemma-e4b, gemma-26b-a4b, gemma-31b" in out
    code, out, err = cli("serve", str(ROOT / "configs" / "serve.yaml"), "--check", env={"MTS_RUNS": str(tmp_path)},
                         monkeypatch=monkeypatch, capsys=capsys)
    assert code != 0 and "multiturnserve train --router cor" in err


def test_replay_needs_hidden_labels_or_dev(tmp_path, monkeypatch, capsys):
    from multiturnserve.arena.release import release
    root = write_dataset(tmp_path / "data")
    release(root, tmp_path / "public", tmp_path / "hidden")
    common = ["arena", "replay", "--router", "knn", "--data", str(tmp_path / "public"), "--models", *MODELS,
              "--out", str(tmp_path / "runs")]
    code, _, err = cli(*common, "--name", "a", capsys=capsys)
    assert code != 0 and "MTS_HIDDEN_DIR" in err
    assert cli(*common, "--name", "b", "--dev", capsys=capsys)[0] == 0
    assert cli(*common, "--name", "c", env={"MTS_HIDDEN_DIR": str(tmp_path / "hidden")}, monkeypatch=monkeypatch,
               capsys=capsys)[0] == 0
    assert cli("arena", "replay", "--router", "knn", "--data", str(root), "--models", *MODELS,
               "--out", str(tmp_path / "runs"), "--name", "d", capsys=capsys)[0] == 0
    c, d = (json.loads((tmp_path / "runs" / n / "results.json").read_text()) for n in "cd")
    assert c["policies"] == d["policies"] and c["fixed_models"] == d["fixed_models"]


def test_leaderboard_from_results(tmp_path, capsys):
    res = {"models": MODELS, "n_test_turns": 100, "best_single_model": "gemma-31b",
           "fixed_models": {"gemma-31b": [0.05, 0.70], "granite-3b": [0.02, 0.5]},
           "policies": {"CoR (DSP + SST)": {"AUDC": 0.74, "QNC": 0.04, "QNC_hull": 0.035, "peak_quality": 0.745},
                        "kNN query + static": {"AUDC": 0.69, "QNC": None, "QNC_hull": None, "peak_quality": 0.69},
                        "oracle (true quality and cost)": {"AUDC": 0.86, "QNC": 0.005, "QNC_hull": 0.005,
                                                           "peak_quality": 0.86}}}
    (tmp_path / "results.json").write_text(json.dumps(res))
    code, _, err = cli("arena", "leaderboard", str(tmp_path / "results.json"), "--out", str(tmp_path / "LB.md"),
                       "--json", str(tmp_path / "lb.json"), capsys=capsys)
    assert code == 0, err
    md = (tmp_path / "LB.md").read_text()
    assert "| 1 | **CoR (ours)** (`cor`) | 0.7400 | 0.0350 | 70% | 0.7450 |" in md
    assert "| 2 | kNN (`knn`) | 0.6900 | - | - | 0.6900 |" in md and "oracle" not in md
    assert json.loads((tmp_path / "lb.json").read_text())["routers"][0]["router"] == "cor"


def test_static_submit_packs_the_router(tmp_path, capsys):
    import tarfile
    out = tmp_path / "sub.tar.gz"
    code, _, err = cli("arena", "submit", str(ROOT / "examples" / "my_router.py"), "--files",
                       str(ROOT / "examples" / "my_lab.py"), "--out", str(out), capsys=capsys)
    assert code == 0, err
    with tarfile.open(out) as t:
        assert sorted(t.getnames()) == ["meta.json", "router/my_lab.py", "router/my_router.py"]
        meta = json.loads(t.extractfile("meta.json").read())
    assert meta["router"] == "my-router" and meta["track"] == "static" and len(meta["files"]["my_router.py"]) == 64
    code, _, err = cli("arena", "submit", capsys=capsys)
    assert code != 0 and "router file" in err
