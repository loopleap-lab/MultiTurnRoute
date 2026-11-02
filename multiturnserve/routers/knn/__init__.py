"""kNN routing: every model's quality is the similarity-weighted mean of its labels on the 20 training turns whose
query embeddings are most similar; decision by quality-cost utility with the cost at full input price."""
import numpy as np
import torch

from ...arena.embeddings import conv_arrays
from ...parts import QualityHead, StaticCost
from ...parts.nn import targets
from ...registry import register_router
from ...router import Router


class KNNPredictor:
    """predict(conv, emb) -> [T, K]: labels of the k nearest training queries, weighted by exp(similarity / 0.05)."""

    def __init__(self, K: np.ndarray, Y: np.ndarray, k: int):
        self.K, self.Y, self.k = K, Y, k

    def __call__(self, c, emb) -> np.ndarray:
        q = conv_arrays(c, emb)[0]
        sims = (q / np.linalg.norm(q, axis=1, keepdims=True).clip(1e-6)) @ self.K.T
        idx = np.argpartition(-sims, self.k, axis=1)[:, :self.k]
        w = np.exp(np.take_along_axis(sims, idx, 1) / 0.05)
        return (w[:, :, None] * self.Y[idx]).sum(1) / w.sum(1, keepdims=True)


def knn_query(train, emb, models, k: int = 20) -> KNNPredictor:
    K = np.concatenate([conv_arrays(c, emb)[0] for c in train])
    K = K / np.linalg.norm(K, axis=1, keepdims=True).clip(1e-6)
    Y = np.concatenate([targets(c, models) for c in train])
    k = min(k, len(K) - 1)
    return KNNPredictor(K, Y, k)


@register_router
class KNN(Router):
    name = "knn"
    label = "kNN query + static"
    trained = True
    uses_embeddings = True

    def __init__(self):
        self.quality, self.cost = QualityHead(), StaticCost()

    def decide(self, turn, state):
        q = self.quality(turn, state.history)
        c = self.cost(turn, state.kv_cache)
        return self.pick(q, c)

    def fit(self, ctx) -> None:
        self.models = list(ctx.models)
        self.quality = QualityHead(knn_query(ctx.train_all, ctx.emb, ctx.models), ctx.models)

    def predictors(self, emb) -> dict:
        return {"knn_query": lambda c: self.quality.predict(c, emb)}

    def state(self) -> dict:
        p = self.quality.predict
        return {"K": torch.from_numpy(p.K), "Y": torch.from_numpy(p.Y), "k": p.k}

    def load_state(self, state: dict) -> None:
        self.quality = QualityHead(KNNPredictor(state["K"].numpy(), state["Y"].numpy(), state["k"]), self.models)
