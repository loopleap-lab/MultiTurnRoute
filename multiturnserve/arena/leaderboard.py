"""results.json (arena replay) -> LEADERBOARD.md, and the JSON the website's leaderboard reads."""
import json

NAMES = {  # label in results.json -> (router, name on the leaderboard)
    "CoR (DSP + SST)": ("cor", "CoR (ours)"),
    "RAG history + static": ("rag", "RAG history"),
    "single-turn (query-only + static)": ("single-turn", "Single-turn"),
    "full history + switch penalty (Bao-style)": ("bao", "Bao et al. (simplified)"),
    "full history + static": ("full-history", "Full history"),
    "kNN query + static": ("knn", "kNN"),
}


WEIGHTS = (0.0, 0.05, 0.1, 0.2, 0.5)   # cost weights of the score table (the website's slider)


def best_score(curve: list, ref_cost: float, w: float) -> float:
    """max over the curve of quality - w * (cost / the best single model's cost)."""
    return max(q - w * c / ref_cost for _, c, q in curve)


def rows(results: dict) -> tuple:
    """(router rows sorted by AUDC, reference row of the best single model). Oracles are left out."""
    best = results["best_single_model"]
    ref_cost, ref_q = results["fixed_models"][best]
    out = []
    for label, r in results["policies"].items():
        if label.startswith("oracle"):
            continue
        router, name = NAMES.get(label, (label, label))
        share = None if r["QNC_hull"] is None else r["QNC_hull"] / ref_cost
        row = {"router": router, "name": name, "label": label, "AUDC": r["AUDC"], "QNC_hull": r["QNC_hull"],
               "cost_vs_best": share, "peak_quality": r["peak_quality"]}
        if r.get("curve"):
            row["curve"] = [[round(c / ref_cost, 4), round(q, 4)] for _, c, q in r["curve"]]
            row["scores"] = {str(w): round(best_score(r["curve"], ref_cost, w), 4) for w in WEIGHTS}
        out.append(row)
    out.sort(key=lambda x: -x["AUDC"])
    return out, {"model": best, "cost": ref_cost, "quality": ref_q}


def markdown(results: dict, source: str = "results.json", note: str = "") -> str:
    rs, ref = rows(results)
    m = ref["model"]
    lines = [
        "# Leaderboard",
        "",
        f"Static track of MultiTurnServe Arena: {results['n_test_turns']:,} test turns, {len(results['models'])} models "
        f"(`configs/arena/static.yaml`). Every router is trained on the train split and replayed on the test split "
        f"over 41 values of the cost weight. Source: {source}.",
        "",
        "- **AUDC**: area under the cost-quality curve (cost normalized by the most expensive model). Higher is better.",
        f"- **QNC**: the lowest normalized cost at which a router reaches the quality of the best single model, "
        f"{m} ({ref['quality']:.4f}); interpolated on the curve's upper envelope. Lower is better; - means never.",
        f"- **Cost vs {m}**: QNC as a share of what {m} alone costs ({ref['cost']:.4f}).",
        "- **Peak**: the highest mean quality on the curve.",
        "",
        f"| Rank | Router | AUDC | QNC | Cost vs {m} | Peak |",
        "|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(rs, 1):
        q = "-" if r["QNC_hull"] is None else f"{r['QNC_hull']:.4f}"
        s = "-" if r["cost_vs_best"] is None else f"{100 * r['cost_vs_best']:.0f}%"
        name = f"**{r['name']}**" if i == 1 else r["name"]
        lines.append(f"| {i} | {name} (`{r['router']}`) | {r['AUDC']:.4f} | {q} | {s} | {r['peak_quality']:.4f} |")
    lines += [f"| | {m} alone (reference) | | {ref['cost']:.4f} | 100% | {ref['quality']:.4f} |", ""]
    if all("scores" in r for r in rs):
        lines += ["### Score at a cost weight", "",
                  f"How much cost matters is a choice. For a weight w, each router is taken at its best point of the "
                  f"curve for quality - w x cost, with cost as a share of what {m} alone costs.", "",
                  "| Router | " + " | ".join(f"w = {w:g}" for w in WEIGHTS) + " |",
                  "|---|" + "---|" * len(WEIGHTS)]
        for r in sorted(rs, key=lambda r: -r["scores"][str(WEIGHTS[1])]):
            lines.append(f"| {r['name']} | " + " | ".join(f"{r['scores'][str(w)]:.4f}" for w in WEIGHTS) + " |")
        lines += [f"| {m} alone | " + " | ".join(f"{ref['quality'] - w:.4f}" for w in WEIGHTS) + " |", ""]
    if note:
        lines += [note, ""]
    return "\n".join(lines)


def site_json(results: dict) -> dict:
    rs, ref = rows(results)
    return {"track": "static", "n_test_turns": results["n_test_turns"], "models": results["models"],
            "reference": ref, "routers": rs}


def write(results_path: str, out: str = "LEADERBOARD.md", json_out: str | None = None, note: str = "",
          source: str | None = None) -> None:
    results = json.loads(open(results_path).read())
    with open(out, "w") as f:
        f.write(markdown(results, source or results_path, note))
    if json_out:
        with open(json_out, "w") as f:
            json.dump(site_json(results), f, indent=1)
