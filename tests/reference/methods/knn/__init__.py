"""kNN routing: every model's quality is the similarity-weighted mean of its labels on the 20 training turns whose
query embeddings are most similar; decision by quality-cost utility with the cost at full input price."""
import numpy as np

from common.embeddings import conv_arrays
from common.nn import targets
from common.simulate import UtilityPolicy

NAME = "knn"


def knn_query(train, emb, models, k: int = 20):
    K = np.concatenate([conv_arrays(c, emb)[0] for c in train])
    K = K / np.linalg.norm(K, axis=1, keepdims=True).clip(1e-6)
    Y = np.concatenate([targets(c, models) for c in train])
    k = min(k, len(K) - 1)

    def predict(c):
        q = conv_arrays(c, emb)[0]
        sims = (q / np.linalg.norm(q, axis=1, keepdims=True).clip(1e-6)) @ K.T
        idx = np.argpartition(-sims, k, axis=1)[:, :k]
        w = np.exp(np.take_along_axis(sims, idx, 1) / 0.05)
        return (w[:, :, None] * Y[idx]).sum(1) / w.sum(1, keepdims=True)
    return predict


def build(ctx) -> tuple:
    pred = knn_query(ctx.train_all, ctx.emb, ctx.models)
    q_hat = {c.conv_id: pred(c) for c in ctx.test}
    return [UtilityPolicy("kNN query + static", q_hat, ctx.out_hat, ctx.overhead)], {"knn_query": pred}
