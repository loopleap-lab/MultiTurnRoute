"""Load the dataset (chat.jsonl) into immutable Turn / Conversation objects.

Per turn: quality of every model in [0, 1] (judge_quality), the tokens it read and wrote.
Only the longest prefix of consecutive turns where every model has a label is kept (a missing middle turn would break
the cost accounting of a routed path). The split comes from the dataset's `split` column (train/test).

The data folder is MTS_DATA_DIR (default ./data): chat.jsonl, embeddings/emb_chat_granite3b.npz, prices.json. In the
public release the test turns carry no labels; the organizer's labels are in MTS_HIDDEN_DIR/labels.jsonl
({"turn_id", "labels"} per line) and are joined back by turn_id.
"""
import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from ..pricing import STATIC

MODELS = STATIC["models"]
LABEL = {"chat": "judge_quality", "lic": "evaluator_score"}


def data_dir(path=None) -> Path:
    return Path(path or os.environ.get("MTS_DATA_DIR") or "data")


def hidden_dir(path=None) -> Path | None:
    p = path or os.environ.get("MTS_HIDDEN_DIR")
    return Path(p) if p else None


@dataclass(frozen=True)
class Turn:
    turn_id: str
    t: int
    query: str
    orig_answer: str
    category: str
    context_dependent: bool
    quality: dict    # model -> quality in [0, 1]
    n_prompt: dict   # model -> prompt tokens it read
    n_output: dict   # model -> output tokens it wrote


@dataclass(frozen=True)
class Conversation:
    conv_id: str
    source: str
    split: str | None = None   # "train" / "test"
    fold: int | None = None    # a fold column, for datasets that have one
    turns: tuple = field(default_factory=tuple)


def split_of(conv_id: str, test_frac: float = 0.2) -> str:
    """80/20 by conversation (md5 of the id); identical to the dataset's `split` column."""
    return "test" if int(hashlib.md5(conv_id.encode()).hexdigest(), 16) % 1000 < test_frac * 1000 else "train"


def fold_of(conv_id: str, k: int) -> int:
    """K folds by conversation (md5 of the id + "#fold"); for k = 5 identical to a dataset's `fold` column."""
    return int(hashlib.md5((conv_id + "#fold").encode()).hexdigest(), 16) % k


def holdout(conv_id: str, frac: float = 0.1) -> bool:
    """Deterministic 10% of training conversations held out for early stopping."""
    return int(hashlib.md5((conv_id + "#val").encode()).hexdigest(), 16) % 1000 < frac * 1000


def hidden_labels(hidden=None) -> dict:
    """turn_id -> labels, from MTS_HIDDEN_DIR/labels.jsonl (empty if no hidden folder)."""
    h = hidden_dir(hidden)
    if h is None or not (h / "labels.jsonl").exists():
        return {}
    with open(h / "labels.jsonl") as f:
        return {r["turn_id"]: r["labels"] for r in map(json.loads, f)}


def load(name: str = "chat", models: list | None = None, batch: str | None = None, data=None, hidden=None) -> tuple:
    """Return (conversations, models). name: the dataset file (default chat.jsonl); batch: keep one sampling batch, e.g. "pilot".
    Streams the JSONL file and keeps only the fields routing needs (no response texts)."""
    models = models or MODELS
    extra = hidden_labels(hidden)
    by_conv: dict = {}
    path, seen = data_dir(data) / f"{name}.jsonl", set()
    if not path.exists():
        raise FileNotFoundError(f"no {path}: set MTS_DATA_DIR (or --data) to the MultiTurnServe data folder. "
                                f"The data are not public yet (see ROADMAP.md)")
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            seen.update(r.get("responses") or {})
            if batch and r.get("batch") != batch:
                continue
            labels = r.get("labels") or extra.get(r["turn_id"]) or {}
            if not all(labels.get(m) and labels[m].get(LABEL[name]) is not None for m in models):
                continue
            turn = Turn(r["turn_id"], r["turn"], r["query"], r["original_answer"], r.get("category") or r.get("task", ""),
                        bool(r.get("context_dependent", r["turn"] > 1)),
                        {m: float(labels[m][LABEL[name]]) for m in models},
                        {m: r["responses"][m]["n_prompt_tokens"] for m in models},
                        {m: r["responses"][m]["n_output_tokens"] for m in models})
            meta = (r["source"], r.get("split"), r.get("fold"))
            by_conv.setdefault(r["conversation_id"], (meta, []))[1].append(turn)
    convs = []
    for cid, ((source, split, fold), turns) in by_conv.items():
        turns = sorted(turns, key=lambda x: x.t)
        keep = [x for i, x in enumerate(turns) if x.t == i + 1]
        if keep:
            convs.append(Conversation(cid, source, split, fold, tuple(keep)))
    if not convs:
        raise ValueError(f"no turn of {path} has labels for every model of the pool ({' '.join(models)}); "
                         f"choose the models of the data with --models (the data have: {' '.join(sorted(seen))})")
    return convs, models
