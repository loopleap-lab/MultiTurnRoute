"""Train routers on the training split: what every router's fit() receives (Context), the per-turn encoder overhead,
and the split into train / early-stopping hold-out / test (the build part of the evaluation package)."""
from dataclasses import dataclass

import numpy as np

from ..parts.length import length_weights, predict_length
from ..pricing import PRICES, STATIC
from . import data
from .embeddings import conv_arrays

ENCODER_MODEL = STATIC["encoder"]["price_as"]          # the frozen encoder's input tokens are charged at this price
FULL_HISTORY_CHARS = STATIC["encoder"]["history_chars"]  # characters of "history + query" behind the "h" embedding


@dataclass(frozen=True)
class Context:
    train_all: list    # all training conversations, original order
    train: list        # training conversations (early-stopping hold-out removed)
    val: list          # early-stopping hold-out
    test: list         # evaluation conversations
    emb: dict          # turn_id -> {"q", "a", "h"}
    models: list       # model pool (names, fixed order)
    out_hat: dict      # conv_id -> [T, K] predicted output tokens (shared length heads)
    overhead: dict     # conv_id -> [T] USD per turn of running the frozen encoder
    args: object       # run arguments (epochs, seed, n_models, loss, rank_w, router-specific options)
    length: np.ndarray | None = None   # the shared length heads' weights [d + 1, K]

    @property
    def seeds(self) -> list:
        """Every trained predictor is averaged over n_models seeds."""
        return [self.args.seed + 100 * m for m in range(getattr(self.args, "n_models", 1))]


def encoder_overhead(test: list) -> dict:
    """USD per turn of running the frozen encoder: (query + answer + history characters) / 4 tokens at its input price."""
    price = PRICES[ENCODER_MODEL]["input"] / 1e6
    return {c.conv_id: np.array([(min(len(t.query), 2000) + min(len(t.orig_answer), 2000) + FULL_HISTORY_CHARS) / 4
                                 * price for t in c.turns]) for c in test}


def make_context(train_all: list, test: list, emb: dict, models: list, args) -> Context:
    val = [c for c in train_all if data.holdout(c.conv_id)]
    if not val:  # tiny folds: still keep a held-out slice for early stopping
        val = train_all[-max(1, len(train_all) // 10):]
    train = [c for c in train_all if c not in val]
    print(f"models={models}\nconvs train/val/test = {len(train)}/{len(val)}/{len(test)}, "
          f"test turns {sum(len(c.turns) for c in test)}", flush=True)
    W = length_weights(train_all, emb, models)
    out_hat = {c.conv_id: predict_length(W, conv_arrays(c, emb)[0]) for c in test}
    return Context(train_all, train, val, test, emb, models, out_hat, encoder_overhead(test), args, W)


def build_routers(specs: list, ctx: Context, options: dict | None = None) -> list:
    """Instantiate and train each router (name or plugin file), in order; each one keeps the shared length heads."""
    from ..parts.length import LengthHead
    from ..registry import get_router
    routers = []
    for spec in specs:
        cls = get_router(spec)
        r = cls(**(options or {}).get(cls.name, {}))
        r.fit(ctx)
        if r.length is None and ctx.length is not None:
            r.length = LengthHead(ctx.length, ctx.models)
        if r.models is None:
            r.models = list(ctx.models)
        routers.append(r)
    return routers


def train(spec: str, args, data_path=None, out=None):
    """Fit one router on every training conversation of the dataset and save it (multiturnserve train)."""
    from . import embeddings
    convs, models = data.load(getattr(args, "dataset", "chat"), args.models, data=data_path)
    split = lambda c: c.split or data.split_of(c.conv_id)  # noqa: E731
    train_all = [c for c in convs if split(c) == "train"]
    emb = embeddings.load(getattr(args, "dataset", "chat"), {t.turn_id for c in train_all for t in c.turns},
                          data=data_path)
    ctx = make_context(train_all, [], emb, models, args)
    r = build_routers([spec], ctx)[0]
    return r.save(out) if out else r
