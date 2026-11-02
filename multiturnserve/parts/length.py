"""Per-model output-length prediction (shared by every router's cost estimate): ridge regression of
log(1 + output tokens) on [query embedding; 1], fitted on the training conversations."""
import numpy as np

from ..arena.embeddings import conv_arrays


def fit_length_heads(Q: np.ndarray, out_tokens: np.ndarray, ridge: float = 1.0) -> np.ndarray:
    X = np.hstack([Q, np.ones((len(Q), 1), dtype=np.float32)])
    Y = np.log1p(out_tokens)
    return np.linalg.solve(X.T @ X + ridge * np.eye(X.shape[1]), X.T @ Y)


def predict_length(W: np.ndarray, Q: np.ndarray) -> np.ndarray:
    return np.expm1(np.hstack([Q, np.ones((len(Q), 1), dtype=np.float32)]) @ W).clip(min=1)


def length_weights(train_all: list, emb: dict, models: list) -> np.ndarray:
    """The ridge weights [d + 1, K], fitted on every training turn."""
    Q = np.concatenate([conv_arrays(c, emb)[0] for c in train_all])
    O = np.array([[t.n_output[m] for m in models] for c in train_all for t in c.turns], dtype=np.float32)
    return fit_length_heads(Q, O)


def output_length_predictions(train_all: list, test: list, emb: dict, models: list) -> dict:
    """conv_id -> [T, K] predicted output tokens for the test conversations."""
    W = length_weights(train_all, emb, models)
    return {c.conv_id: predict_length(W, conv_arrays(c, emb)[0]) for c in test}


class LengthHead:
    """Predicted output tokens of every model for a live turn, from its query embedding (online serving)."""

    def __init__(self, W: np.ndarray, models: list):
        self.W, self.models = W, list(models)

    def __call__(self, turn) -> np.ndarray:
        q = turn.emb[turn.conv.turns[turn.t].turn_id]["q"]
        out = predict_length(self.W, q[None, :])[0]
        return np.array([out[self.models.index(m)] for m in turn.models], dtype=out.dtype)
