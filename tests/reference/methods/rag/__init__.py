"""Retrieval-augmented routing: an MLP predicts every model's quality from [query; mean of the 2 past turns most
similar to the query]; decision by quality-cost utility with the cost at full input price."""
import numpy as np

from common.embeddings import conv_arrays
from common.nn import averaged, train_mlp
from common.simulate import UtilityPolicy

NAME = "rag"


def rag_context(q: np.ndarray, past_x: np.ndarray, k: int = 2) -> np.ndarray:
    """Mean of the k past turns most similar to the query (cosine on the query half); zeros if no past."""
    d = len(q)
    if len(past_x) == 0:
        return np.zeros(2 * d, dtype=np.float32)
    sims = past_x[:, :d] @ q
    return past_x[np.argsort(-sims)[:k]].mean(axis=0)


def build(ctx) -> tuple:
    def inputs(c):
        q, x, _ = conv_arrays(c, ctx.emb)
        return np.stack([np.concatenate([q[t], rag_context(q[t], x[:t])]) for t in range(len(q))])
    pred = averaged([train_mlp(ctx.train, ctx.val, inputs, ctx.models, ctx.args.epochs, sd, ctx.args.rank_w)
                     for sd in ctx.seeds])
    q_hat = {c.conv_id: pred(c) for c in ctx.test}
    return [UtilityPolicy("RAG history + static", q_hat, ctx.out_hat, ctx.overhead)], {"rag": pred}
