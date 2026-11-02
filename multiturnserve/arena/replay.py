"""Static track: offline turn-by-turn replay of routers over conversations with labels for every model at every turn.

A router is any Router: decide(turn, state) -> model name, where state holds the history and the prompt-cache state
of the conversation so far (pricing.CacheState) and router.lam in [0, 1] weighs cost against quality. The replay owns
the accounting: the quality of the chosen model is its label for that turn; the cost is the realized tokens priced
with the cache state along the routed path, normalized per turn by the most expensive model's full-price cost. All
models answer the same original history.

Metrics: AUDC (area under the upper envelope of the cost-quality curve), QNC / QNC_hull (minimum cost to reach the
best single model's quality), peak quality, and a paired bootstrap over conversations.
"""
import numpy as np

from ..parts import CacheCost
from ..parts.nn import targets
from ..pricing import PRICES, CacheState, static_cost, turn_cost
from ..router import Router
from ..types import SessionState, Turn
from . import data
from .train import build_routers, make_context

# lambda trades quality against normalized cost; kappa = lambda / (1 - lambda) log-spaced over 1e-2..1e3
_KAPPA = np.logspace(-2, 3, 40)
LAMBDAS = np.concatenate([[0.0], _KAPPA / (1 + _KAPPA)])


def norm_base(turn, arms: list) -> float:
    return max(static_cost(a, turn.n_prompt[a], turn.n_output[a]) for a in arms)


class OracleRouter(Router):
    """Reference point: true quality and the true (cache-accounted) cost of every model."""
    name = "oracle"
    label = "oracle (true quality and cost)"
    encoder_overhead = False

    def __init__(self, convs: list, models: list):
        self.true_q = {c.conv_id: targets(c, models) for c in convs}
        self.true_out = {c.conv_id: np.array([[t.n_output[m] for m in models] for t in c.turns]) for c in convs}
        self.cost = CacheCost()

    def decide(self, turn, state):
        q = self.true_q[turn.chat_id][turn.t]
        c = self.cost(turn, state.kv_cache, out=self.true_out[turn.chat_id][turn.t])
        return self.pick(q, c)


def replay_turns(conv, arms: list, out_hat: dict | None, emb: dict | None) -> list:
    """The Turn a router sees at every turn of a recorded conversation."""
    oh = out_hat.get(conv.conv_id) if out_hat else None
    return [Turn(conv.conv_id, t, turn.query, tuple(arms), turn.n_prompt, None if oh is None else oh[t], PRICES,
                 conv, emb) for t, turn in enumerate(conv.turns)]


def run_conversation(conv, arms: list, router, lam, out_hat=None, overhead=None, emb=None, turns=None) -> list:
    """Return per-turn (chosen model index, realized quality, realized normalized cost)."""
    router.lam = lam   # a numpy float64 from LAMBDAS, as in the evaluation package (it fixes the dtype of the utility)
    turns = turns or replay_turns(conv, arms, out_hat, emb)
    state, out = SessionState(), []
    for t, turn in enumerate(conv.turns):
        a = router.decide(turns[t], state)
        k = arms.index(a)
        cost = turn_cost(a, turn.n_prompt[a], state.kv_cache.cached(a, turn.n_prompt[a]), turn.n_output[a])
        if router.encoder_overhead and overhead is not None:
            cost += overhead[conv.conv_id][t]
        out.append((k, turn.quality[a], cost / norm_base(turn, arms)))
        state = state.after(turn.query, turn.orig_answer, a, turn.n_prompt[a], turn.n_output[a])
    return out


def fixed_arm(conv, arms: list, k: int) -> list:
    """Always one arm (its cache grows every turn)."""
    state, out, a = CacheState(), [], arms[k]
    for turn in conv.turns:
        cost = turn_cost(a, turn.n_prompt[a], state.cached(a, turn.n_prompt[a]), turn.n_output[a])
        out.append((k, turn.quality[a], cost / norm_base(turn, arms)))
        state = state.after_serving(a, turn.n_prompt[a], turn.n_output[a])
    return out


def curve(convs: list, arms: list, router, lambdas, out_hat=None, overhead=None, emb=None) -> list:
    """Deferral curve: (lambda, mean normalized cost, mean quality, arm shares) over all turns of all convs."""
    pts = []
    for lam in lambdas:
        rows = [r for c in convs for r in run_conversation(c, arms, router, lam, out_hat, overhead, emb)]
        shares = np.bincount([r[0] for r in rows], minlength=len(arms)) / len(rows)
        pts.append((float(lam), float(np.mean([r[2] for r in rows])), float(np.mean([r[1] for r in rows])), shares.tolist()))
    return pts


def audc(points: list) -> float:
    """Area under the upper envelope of (cost, quality) over cost in [0, 1]; flat extension to the right."""
    pts = sorted((c, q) for c, q in points)
    env, best = [], -1.0
    for c, q in pts:
        if q > best:
            env.append((c, q))
            best = q
    if not env:
        return 0.0
    xs = [0.0] + [c for c, _ in env] + [1.0]
    ys = [0.0] + [q for _, q in env] + [env[-1][1]]   # starts at (0, 0); linear between envelope points
    integrate = getattr(np, "trapezoid", None) or np.trapz
    return float(integrate(ys, xs))


def qnc(points: list, target: float) -> float | None:
    """Minimum normalized cost at which the curve reaches the target quality (e.g. the best single arm)."""
    ok = [c for c, q in points if q >= target - 1e-9]
    return min(ok) if ok else None


def qnc_hull(points: list, target: float) -> float | None:
    """Minimum normalized cost to reach the target on the upper envelope of the curve, interpolating linearly between
    envelope points (randomizing between two operating points reaches any point on the segment, as AUDC assumes)."""
    pts = sorted((c, q) for c, q in points)
    env, best = [], -1.0
    for c, q in pts:
        if q > best:
            env.append((c, q))
            best = q
    if not env or env[-1][1] < target - 1e-9:
        return None
    if env[0][1] >= target:
        return env[0][0]
    for (c0, q0), (c1, q1) in zip(env, env[1:]):
        if q1 >= target:
            return c0 + (target - q0) / (q1 - q0) * (c1 - c0)
    return None


def per_conv_curve(convs: list, arms: list, router, lambdas, out_hat=None, overhead=None, emb=None) -> np.ndarray:
    """[n_lambda, n_conv, 3] of (sum quality, sum normalized cost, n turns) per conversation, for bootstrapping."""
    out = np.zeros((len(lambdas), len(convs), 3))
    turns = {c.conv_id: replay_turns(c, arms, out_hat, emb) for c in convs}
    for i, lam in enumerate(lambdas):
        for j, c in enumerate(convs):
            rows = run_conversation(c, arms, router, lam, out_hat, overhead, emb, turns[c.conv_id])
            out[i, j] = (sum(r[1] for r in rows), sum(r[2] for r in rows), len(rows))
    return out


def bootstrap(stats: dict, target_stats: np.ndarray, n: int = 1000, seed: int = 0) -> dict:
    """Paired bootstrap over conversations. stats: policy -> per_conv_curve array; target_stats: [n_conv, 3] of the
    best single model (its quality on each resample is the QNC target). Returns, per policy, 95% CIs of AUDC and QNC,
    and for every pair of policies (a, b) the share of resamples where a has the higher AUDC / the lower QNC:
    {"ci": {policy: {...}}, "AUDC_win": {a: {b: share}}, "QNC_win": {a: {b: share}}}."""
    rng = np.random.default_rng(seed)
    n_conv = target_stats.shape[0]
    draws = {p: {"AUDC": [], "QNC": []} for p in stats}
    for _ in range(n):
        idx = rng.integers(n_conv, size=n_conv)
        t = target_stats[idx].sum(0)
        target = t[0] / t[2]
        for p, s in stats.items():
            agg = s[:, idx].sum(1)
            pts = list(zip(agg[:, 1] / agg[:, 2], agg[:, 0] / agg[:, 2]))
            draws[p]["AUDC"].append(audc(pts))
            draws[p]["QNC"].append(qnc_hull(pts, target))
    A = {p: np.array(d["AUDC"]) for p, d in draws.items()}
    Q = {p: np.array([np.inf if v is None else v for v in d["QNC"]]) for p, d in draws.items()}
    ci = {p: {"AUDC_ci": [round(float(np.percentile(A[p], 2.5)), 4), round(float(np.percentile(A[p], 97.5)), 4)],
              "QNC_ci": [round(float(np.percentile(Q[p], 2.5)), 4), round(float(np.percentile(Q[p], 97.5)), 4)]}
          for p in draws}
    return {"ci": ci,
            "AUDC_win": {a: {b: round(float(np.mean(A[a] > A[b])), 3) for b in draws if b != a} for a in draws},
            "QNC_win": {a: {b: round(float(np.mean(Q[a] < Q[b])), 3) for b in draws if b != a} for a in draws}}


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
                         for rows in (fixed_arm(c, models, k) for c in convs)]) for k, m in enumerate(models)}


def summarize(stats: dict, fixed: dict, n_bootstrap: int) -> dict:
    pts = {m: (float(v[:, 1].sum() / v[:, 2].sum()), float(v[:, 0].sum() / v[:, 2].sum())) for m, v in fixed.items()}
    best = max(pts, key=lambda m: pts[m][1])
    out = {"n_test_turns": int(fixed[best][:, 2].sum()), "fixed_models": pts, "best_single_model": best, "policies": {}}
    for p, s in stats.items():
        agg = s.sum(1)
        cq = list(zip(agg[:, 1] / agg[:, 2], agg[:, 0] / agg[:, 2]))
        out["policies"][p] = {"AUDC": round(audc(cq), 4), "QNC": qnc(cq, pts[best][1]),
                              "QNC_hull": qnc_hull(cq, pts[best][1]),
                              "peak_quality": round(max(q for _, q in cq), 4),
                              "curve": [(float(l), float(c), float(q)) for l, (c, q) in zip(LAMBDAS, cq)]}
    if n_bootstrap:
        out["bootstrap"] = bootstrap({p: s for p, s in stats.items() if not p.startswith("oracle")},
                                     fixed[best], n=n_bootstrap)
    return out


def splits_of(convs: list, folds: int = 1, dev: bool = False) -> list:
    """[(train, test)]: the dataset's train/test split, K folds over conversations, or (dev) a fifth of the training
    conversations held out as a local test set (the test labels are hidden)."""
    if folds > 1:
        return [([c for c in convs if data.fold_of(c.conv_id, folds) != f],
                 [c for c in convs if data.fold_of(c.conv_id, folds) == f]) for f in range(folds)]
    split = lambda c: c.split or data.split_of(c.conv_id)  # noqa: E731
    if dev:
        tr = [c for c in convs if split(c) == "train"]
        return [([c for c in tr if data.fold_of(c.conv_id, 5) != 0], [c for c in tr if data.fold_of(c.conv_id, 5) == 0])]
    return [([c for c in convs if split(c) == "train"], [c for c in convs if split(c) == "test"])]


def evaluate(convs: list, emb: dict, models: list, args, routers: list, options: dict | None = None,
             keep: list | None = None) -> dict:
    """Train each router on every split, replay it on the split's test conversations over LAMBDAS, and summarize.
    keep: if given, gets (ctx, trained routers) of every split."""
    splits = splits_of(convs, getattr(args, "folds", 1), getattr(args, "dev", False))
    stats, all_test, mse = {}, [], []
    for f, (train_all, test) in enumerate(splits):
        if not test:
            raise SystemExit("no test conversations with labels: the test labels are hidden (set SRX_HIDDEN_DIR on "
                             "the organizer's machine), or replay on held-out training conversations with --dev")
        print(f"split {f}: train {len(train_all)} test {len(test)}", flush=True)
        ctx = make_context(train_all, test, emb, models, args)
        trained = build_routers(routers, ctx, options)
        for r in trained:
            r.prepare(test, emb)
        if keep is not None:
            keep.append((ctx, trained))
        for r in trained + [OracleRouter(test, models)]:
            stats.setdefault(r.display, []).append(per_conv_curve(test, models, r, LAMBDAS, ctx.out_hat, ctx.overhead,
                                                                  emb))
        preds = {}
        for r in trained:
            preds.update(r.predictors(emb))
        mse.append(mse_report(test, models, preds))
        all_test += test
    stats = {p: np.concatenate(v, axis=1) for p, v in stats.items()}
    out = summarize(stats, fixed_models(all_test, models), getattr(args, "bootstrap", 0))
    out["quality_mse_per_split"] = mse
    return out
