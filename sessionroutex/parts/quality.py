"""Quality parts: the predicted quality of every candidate model for a turn, from the dialogue so far."""
import numpy as np


class QualityHead:
    """Wraps a trained predictor(conv, emb) -> [T, K] (K = the models it was trained on, in that order).

    quality(turn, history) -> [len(turn.models)] predicted quality for this turn. The predictor reads the conversation
    so far (turn.conv) and its embeddings (turn.emb); the history is already in them. For replay, prepare() predicts
    whole recorded conversations once, as the evaluation package does."""

    def __init__(self, predict=None, models=None):
        self.predict, self.models = predict, list(models or [])
        self.q_hat, self._emb = {}, None

    def prepare(self, convs: list, emb: dict) -> None:
        self.q_hat, self._emb = {c.conv_id: self.predict(c, emb) for c in convs}, emb

    def __call__(self, turn, history=None) -> np.ndarray:
        if self.predict is None:
            raise RuntimeError("this router is not trained: fit it (sessionroutex train) or load a checkpoint")
        if turn.emb is None:
            raise RuntimeError("this router reads turn embeddings; serve it with an encoder (encoder: granite)")
        q = self.q_hat.get(turn.chat_id) if turn.emb is self._emb else None
        row = (q if q is not None else self.predict(turn.conv, turn.emb))[turn.t]
        if list(turn.models) == self.models:
            return row
        missing = [m for m in turn.models if m not in self.models]
        if missing:
            raise KeyError(f"no quality prediction for {missing}: this router was trained on {self.models}")
        return np.array([row[self.models.index(m)] for m in turn.models], dtype=row.dtype)
