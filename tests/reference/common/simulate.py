"""Offline turn-by-turn simulation of a router over conversations with labels for every model at every turn.

A router is any object with `name`, `overhead` (conv_id -> [T] USD of its own per turn, or None) and
`choose(conv, t, arms, state, prev, lam) -> model index`, where `state` is the prompt-cache state of the conversation
so far (pricing.CacheState), `prev` the index of the model that served the previous turn (None at turn 0), and `lam`
in [0, 1] the weight of cost against quality. The simulator owns the accounting: the quality of the chosen model is
its label for that turn; the cost is the realized tokens priced with the cache state along the routed path,
normalized per turn by the most expensive model's full-price cost. All models answer the same original history.
"""
from dataclasses import dataclass

import numpy as np

from common.pricing import CacheState, static_cost, turn_cost


def norm_base(turn, arms: list) -> float:
    return max(static_cost(a, turn.n_prompt[a], turn.n_output[a]) for a in arms)


@dataclass(frozen=True)
class UtilityPolicy:
    """Choose argmax over models of (1 - lam) * q_hat - lam * estimated cost, the cost estimated at full input price
    from predicted output lengths and normalized by the most expensive model. q_hat, out_hat: conv_id -> [T, K]."""
    name: str
    q_hat: dict
    out_hat: dict
    overhead: dict | None = None

    def estimated_cost(self, turn, t: int, arms: list, state: CacheState, k: int, oh: np.ndarray) -> float:
        a = arms[k]
        return static_cost(a, turn.n_prompt[a], oh[t][k])

    def choose(self, conv, t: int, arms: list, state: CacheState, prev, lam: float) -> int:
        turn, qh, oh = conv.turns[t], self.q_hat[conv.conv_id], self.out_hat[conv.conv_id]
        base_hat = max(static_cost(a, turn.n_prompt[a], oh[t][k]) for k, a in enumerate(arms))
        util = [(1 - lam) * qh[t][k] - lam * self.estimated_cost(turn, t, arms, state, k, oh) / base_hat
                for k in range(len(arms))]
        return int(np.argmax(util))


@dataclass(frozen=True)
class OraclePolicy(UtilityPolicy):
    """Reference point: true quality and the true (cache-accounted) cost of every model."""

    def estimated_cost(self, turn, t: int, arms: list, state: CacheState, k: int, oh: np.ndarray) -> float:
        a = arms[k]
        return turn_cost(a, turn.n_prompt[a], state.cached(a, turn.n_prompt[a]), oh[t][k])


def run_conversation(conv, arms: list, policy, lam: float) -> list:
    """Return per-turn (chosen model index, realized quality, realized normalized cost)."""
    state, prev, out = CacheState(), None, []
    for t, turn in enumerate(conv.turns):
        k = policy.choose(conv, t, arms, state, prev, lam)
        a = arms[k]
        cost = turn_cost(a, turn.n_prompt[a], state.cached(a, turn.n_prompt[a]), turn.n_output[a])
        if policy.overhead is not None:
            cost += policy.overhead[conv.conv_id][t]
        out.append((k, turn.quality[a], cost / norm_base(turn, arms)))
        state, prev = state.after_serving(a, turn.n_prompt[a], turn.n_output[a]), k
    return out


def fixed_arm(conv, arms: list, k: int) -> list:
    """Always one arm (its cache grows every turn)."""
    state, out, a = CacheState(), [], arms[k]
    for turn in conv.turns:
        cost = turn_cost(a, turn.n_prompt[a], state.cached(a, turn.n_prompt[a]), turn.n_output[a])
        out.append((k, turn.quality[a], cost / norm_base(turn, arms)))
        state = state.after_serving(a, turn.n_prompt[a], turn.n_output[a])
    return out


def curve(convs: list, arms: list, policy, lambdas) -> list:
    """Deferral curve: (lambda, mean normalized cost, mean quality, arm shares) over all turns of all convs."""
    pts = []
    for lam in lambdas:
        rows = [r for c in convs for r in run_conversation(c, arms, policy, lam)]
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


def per_conv_curve(convs: list, arms: list, policy, lambdas) -> np.ndarray:
    """[n_lambda, n_conv, 3] of (sum quality, sum normalized cost, n turns) per conversation, for bootstrapping."""
    out = np.zeros((len(lambdas), len(convs), 3))
    for i, lam in enumerate(lambdas):
        for j, c in enumerate(convs):
            rows = run_conversation(c, arms, policy, lam)
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
