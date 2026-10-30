"""Load the dataset (chat or lic) into immutable Turn / Conversation objects.

Per turn: quality of every model in [0, 1] (chat: judge_quality; lic: evaluator_score), the tokens it read and wrote.
Only the longest prefix of consecutive turns where every model has a label is kept (a missing middle turn would break
the cost accounting of a routed path). Splits come from the dataset: chat `split` (train/test), lic `fold` (0-4).
"""
import hashlib
import json
from dataclasses import dataclass, field

from common.config import DATA_DIR, MODELS

LABEL = {"chat": "judge_quality", "lic": "evaluator_score"}


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
    split: str | None = None   # chat: "train" / "test"
    fold: int | None = None    # lic: 0-4
    turns: tuple = field(default_factory=tuple)


def prices() -> dict:
    return json.loads((DATA_DIR / "prices.json").read_text())


def split_of(conv_id: str, test_frac: float = 0.2) -> str:
    """80/20 by conversation (md5 of the id); identical to the dataset's chat `split` column."""
    return "test" if int(hashlib.md5(conv_id.encode()).hexdigest(), 16) % 1000 < test_frac * 1000 else "train"


def fold_of(conv_id: str, k: int) -> int:
    """K folds by conversation (md5 of the id + "#fold"); for k = 5 identical to the dataset's lic `fold` column."""
    return int(hashlib.md5((conv_id + "#fold").encode()).hexdigest(), 16) % k


def holdout(conv_id: str, frac: float = 0.1) -> bool:
    """Deterministic 10% of training conversations held out for early stopping."""
    return int(hashlib.md5((conv_id + "#val").encode()).hexdigest(), 16) % 1000 < frac * 1000


def load(name: str, models: list | None = None, batch: str | None = None) -> tuple:
    """Return (conversations, models). name: "chat" or "lic"; batch (chat only): keep one sampling batch, e.g. "pilot".
    Streams the JSONL file and keeps only the fields routing needs (no response texts)."""
    models = models or MODELS
    by_conv: dict = {}
    with open(DATA_DIR / f"{name}.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if batch and r.get("batch") != batch:
                continue
            labels = r["labels"]
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
    return convs, models
