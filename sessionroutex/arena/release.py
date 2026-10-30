"""The public release of the data: every field of the training turns; the test turns without their labels.

Test rows lose `labels` (judge_quality, judge_expected_score, judge_score, judge_repaired), the response texts and any
other field named like a score or a quality; they keep the inputs (history, query, original answer) and every model's
token counts, which the cost accounting needs. The labels go to a separate folder for the organizer
(SRX_HIDDEN_DIR/labels.jsonl), never into the public one."""
import json
import shutil
from collections import Counter
from pathlib import Path

SCORE_WORDS = ("judge", "score", "quality", "label")


def _clean(x):
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items() if not any(w in k.lower() for w in SCORE_WORDS)}
    if isinstance(x, list):
        return [_clean(v) for v in x]
    return x


def strip_test_row(r: dict) -> dict:
    out = _clean({k: v for k, v in r.items() if k != "labels"})
    for resp in out.get("responses", {}).values():
        resp.pop("text", None)
    return out


def release(data: str, out: str, hidden_out: str | None = None, name: str = "chat") -> dict:
    src, dst = Path(data).resolve(), Path(out).resolve()
    if dst == src or src in dst.parents:
        raise SystemExit(f"--out {dst} must be outside the data folder {src}")
    if hidden_out and (dst == Path(hidden_out).resolve() or dst in Path(hidden_out).resolve().parents):
        raise SystemExit("--hidden-out must not be inside --out (the labels would be published)")
    dst.mkdir(parents=True, exist_ok=True)
    counts, hidden = Counter(), []
    with open(src / f"{name}.jsonl") as f, open(dst / f"{name}.jsonl", "w") as g:
        for line in f:
            r = json.loads(line)
            split = r.get("split") or "train"
            counts[split] += 1
            if split == "test":
                hidden.append({"turn_id": r["turn_id"], "labels": r.get("labels") or {}})
                r = strip_test_row(r)
            g.write(json.dumps(r) + "\n")
    if (src / "embeddings").is_dir():
        shutil.copytree(src / "embeddings", dst / "embeddings", dirs_exist_ok=True)
    if (src / "prices.json").exists():
        shutil.copy(src / "prices.json", dst / "prices.json")
    stats = {"turns": dict(counts), "test_labels": "hidden"}
    (dst / "stats.json").write_text(json.dumps(stats, indent=1))
    if hidden_out:
        h = Path(hidden_out)
        h.mkdir(parents=True, exist_ok=True)
        with open(h / "labels.jsonl", "w") as f:
            for r in hidden:
                f.write(json.dumps(r) + "\n")
    return stats
