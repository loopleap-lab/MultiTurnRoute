"""Retrieval-augmented routing: an MLP predicts every model's quality from [query; mean of the 2 past turns most
similar to the query]; decision by quality-cost utility with the cost at full input price."""
import numpy as np

from ...arena.embeddings import conv_arrays
from ...registry import register_router
from ..mlp import MLPRouter


def rag_context(q: np.ndarray, past_x: np.ndarray, k: int = 2) -> np.ndarray:
    """Mean of the k past turns most similar to the query (cosine on the query half); zeros if no past."""
    d = len(q)
    if len(past_x) == 0:
        return np.zeros(2 * d, dtype=np.float32)
    sims = past_x[:, :d] @ q
    return past_x[np.argsort(-sims)[:k]].mean(axis=0)


@register_router
class RAG(MLPRouter):
    name = "rag"
    label = "RAG history + static"
    predictor_name = "rag"

    @staticmethod
    def features(c, emb):
        q, x, _ = conv_arrays(c, emb)
        return np.stack([np.concatenate([q[t], rag_context(q[t], x[:t])]) for t in range(len(q))])
