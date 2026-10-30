"""Per-model output-length prediction (shared by every method's cost estimate): ridge regression of
log(1 + output tokens) on [query embedding; 1], fitted on the training conversations."""
import numpy as np

from common.embeddings import conv_arrays


def fit_length_heads(Q: np.ndarray, out_tokens: np.ndarray, ridge: float = 1.0) -> np.ndarray:
    X = np.hstack([Q, np.ones((len(Q), 1), dtype=np.float32)])
    Y = np.log1p(out_tokens)
    return np.linalg.solve(X.T @ X + ridge * np.eye(X.shape[1]), X.T @ Y)


def predict_length(W: np.ndarray, Q: np.ndarray) -> np.ndarray:
    return np.expm1(np.hstack([Q, np.ones((len(Q), 1), dtype=np.float32)]) @ W).clip(min=1)


def output_length_predictions(train_all: list, test: list, emb: dict, models: list) -> dict:
    """conv_id -> [T, K] predicted output tokens for the test conversations."""
    Q = np.concatenate([conv_arrays(c, emb)[0] for c in train_all])
    O = np.array([[t.n_output[m] for m in models] for c in train_all for t in c.turns], dtype=np.float32)
    W = fit_length_heads(Q, O)
    return {c.conv_id: predict_length(W, conv_arrays(c, emb)[0]) for c in test}
