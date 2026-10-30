"""Train the selected routing methods and evaluate all of their policies offline.

  python evaluate.py --data lic --methods bao cor full_history knn rag single_turn --folds 5 --bootstrap 1000 \
      --loss bce --name lic_all_s0 --out results/

Single split (chat default): the dataset's train/test split. K folds (--folds K): every conversation is tested once.
Writes <out>/<name>/results.json with, per policy, AUDC, QNC (min cost to reach the best single model's quality),
QNC_hull (same, interpolated on the upper envelope) and peak quality, plus fixed single models, quality MSE of each
predictor, and (with --bootstrap) paired bootstrap CIs and pairwise win shares.
"""
import argparse
import importlib
import json
from pathlib import Path

import numpy as np
import torch

from common import data, embeddings, simulate
from common.config import ENCODER_MODEL, FULL_HISTORY_CHARS
from common.context import Context
from common.length import output_length_predictions
from common.nn import LOSS, targets
from common.pricing import PRICES

METHODS = ["bao", "cor", "full_history", "knn", "rag", "single_turn"]
# lambda trades quality against normalized cost; kappa = lambda / (1 - lambda) log-spaced over 1e-2..1e3
_KAPPA = np.logspace(-2, 3, 40)
LAMBDAS = np.concatenate([[0.0], _KAPPA / (1 + _KAPPA)])
LENGTH_GROUPS = (("2-3 turns", 2, 3), ("4-6 turns", 4, 6), ("7+ turns", 7, 10 ** 6))


def encoder_overhead(test: list) -> dict:
    """USD per turn of running the frozen encoder: (query + answer + history characters) / 4 tokens at its input price."""
    price = PRICES[ENCODER_MODEL]["input"] / 1e6
    return {c.conv_id: np.array([(min(len(t.query), 2000) + min(len(t.orig_answer), 2000) + FULL_HISTORY_CHARS) / 4
                                 * price for t in c.turns]) for c in test}


def build_policies(train_all: list, test: list, emb: dict, models: list, args) -> tuple:
    val = [c for c in train_all if data.holdout(c.conv_id)]
    if not val:  # tiny folds: still keep a held-out slice for early stopping
        val = train_all[-max(1, len(train_all) // 10):]
    train = [c for c in train_all if c not in val]
    print(f"models={models}\nconvs train/val/test = {len(train)}/{len(val)}/{len(test)}, "
          f"test turns {sum(len(c.turns) for c in test)}", flush=True)
    ctx = Context(train_all, train, val, test, emb, models, output_length_predictions(train_all, test, emb, models),
                  encoder_overhead(test), args)
    policies, preds = [], {}
    for name in args.methods:
        p, q = importlib.import_module(f"methods.{name}").build(ctx)
        policies += p
        preds.update(q)
    true_q = {c.conv_id: targets(c, models) for c in test}
    true_out = {c.conv_id: np.array([[t.n_output[m] for m in models] for t in c.turns]) for c in test}
    policies.append(simulate.OraclePolicy("oracle (true quality and cost)", true_q, true_out))
    return policies, preds


def mse_report(test, models, preds) -> dict:
    rep = {}
    for name, f in preds.items():
        err = {"all": [], "context_dependent": [], "context_independent": []}
        for c in test:
            e = ((f(c) - targets(c, models)) ** 2).mean(axis=1)
            for t, v in zip(c.turns, e):
                err["all"].append(v)
                err["context_dependent" if t.context_dependent else "context_independent"].append(v)
        rep[name] = {k: round(float(np.mean(v)), 5) for k, v in err.items() if v}
    return rep


def fixed_models(convs: list, models: list) -> dict:
    """model -> [n_conv, 3] of (sum quality, sum normalized cost, n turns) when that model serves every turn."""
    return {m: np.array([[sum(r[1] for r in rows), sum(r[2] for r in rows), len(rows)]
                         for rows in (simulate.fixed_arm(c, models, k) for c in convs)]) for k, m in enumerate(models)}


def summarize(stats: dict, fixed: dict, n_bootstrap: int) -> dict:
    pts = {m: (float(v[:, 1].sum() / v[:, 2].sum()), float(v[:, 0].sum() / v[:, 2].sum())) for m, v in fixed.items()}
    best = max(pts, key=lambda m: pts[m][1])
    out = {"n_test_turns": int(fixed[best][:, 2].sum()), "fixed_models": pts, "best_single_model": best, "policies": {}}
    for p, s in stats.items():
        agg = s.sum(1)
        cq = list(zip(agg[:, 1] / agg[:, 2], agg[:, 0] / agg[:, 2]))
        out["policies"][p] = {"AUDC": round(simulate.audc(cq), 4), "QNC": simulate.qnc(cq, pts[best][1]),
                              "QNC_hull": simulate.qnc_hull(cq, pts[best][1]),
                              "peak_quality": round(max(q for _, q in cq), 4),
                              "curve": [(float(l), float(c), float(q)) for l, (c, q) in zip(LAMBDAS, cq)]}
    if n_bootstrap:
        out["bootstrap"] = simulate.bootstrap({p: s for p, s in stats.items() if not p.startswith("oracle")},
                                              fixed[best], n=n_bootstrap)
    return out


def evaluate(convs: list, emb: dict, models: list, args) -> dict:
    if args.folds > 1:
        splits = [([c for c in convs if data.fold_of(c.conv_id, args.folds) != f],
                   [c for c in convs if data.fold_of(c.conv_id, args.folds) == f]) for f in range(args.folds)]
    else:
        split = lambda c: c.split or data.split_of(c.conv_id)  # noqa: E731
        splits = [([c for c in convs if split(c) == "train"], [c for c in convs if split(c) == "test"])]
    stats, all_test, mse = {}, [], []
    for f, (train_all, test) in enumerate(splits):
        print(f"split {f}: train {len(train_all)} test {len(test)}", flush=True)
        policies, preds = build_policies(train_all, test, emb, models, args)
        for pol in policies:
            stats.setdefault(pol.name, []).append(simulate.per_conv_curve(test, models, pol, LAMBDAS))
        mse.append(mse_report(test, models, preds))
        all_test += test
    stats = {p: np.concatenate(v, axis=1) for p, v in stats.items()}
    out = summarize(stats, fixed_models(all_test, models), args.bootstrap)
    out["quality_mse_per_split"] = mse
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", choices=["chat", "lic"], required=True)
    ap.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    ap.add_argument("--models", nargs="*", help="model pool (default: all 9)")
    ap.add_argument("--batch", help="chat only: use one sampling batch, e.g. pilot")
    ap.add_argument("--folds", type=int, default=1, help="K > 1: K-fold over conversations (lic: use 5)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-models", type=int, default=1, help="average every trained predictor over this many seeds")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--loss", choices=["mse", "bce"], default="mse", help="pointwise loss of every trained predictor")
    ap.add_argument("--rank-w", type=float, default=0.0, help="weight of a listwise ranking loss over models")
    ap.add_argument("--bootstrap", type=int, default=0, help="paired bootstrap resamples over test conversations")
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", default="results")
    g = ap.add_argument_group("cor")
    g.add_argument("--dsp", type=int, choices=[1, 2], default=1,
                   help="1 = original DSP; 2 = projected variant that also reads the history embedding")
    g.add_argument("--window", type=int, default=2, help="DSP local attention window (turns)")
    g.add_argument("--dsp-p", type=int, default=128, help="--dsp 2 only: projection width")
    g.add_argument("--dsp-drop", type=float, default=0.5, help="--dsp 2 only: dropout")
    g = ap.add_argument_group("bao")
    g.add_argument("--switch-penalty", type=float, default=0.1, help="normalized-cost penalty for changing models")
    args = ap.parse_args()
    torch.set_num_threads(4)
    LOSS["kind"] = args.loss
    convs, models = data.load(args.data, args.models, args.batch)
    emb = embeddings.load(args.data, {t.turn_id for c in convs for t in c.turns})
    print(f"{args.data}: {len(convs)} conversations, {sum(len(c.turns) for c in convs)} turns", flush=True)
    results = {"config": vars(args), "models": models, **evaluate(convs, emb, models, args)}
    out = Path(args.out) / args.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, indent=1))
    print(f"\nbest single model: {results['best_single_model']} {results['fixed_models'][results['best_single_model']]}")
    print(f"{'policy':48s} {'AUDC':>7s} {'QNC_hull':>9s} {'peak':>7s}")
    for name, r in sorted(results["policies"].items(), key=lambda kv: -kv[1]["AUDC"]):
        qn = f"{r['QNC_hull']:.4f}" if r["QNC_hull"] is not None else "-"
        print(f"{name:48s} {r['AUDC']:7.4f} {qn:>9s} {r['peak_quality']:7.4f}")
    print(f"-> {out / 'results.json'}")


if __name__ == "__main__":
    main()
