"""Simplified Bao et al. multi-turn router (not a reproduction of the original algorithm or code).

Idea kept: the quality of every model is predicted from the whole conversation so far, and the decision penalizes
switching to a different model than the one that served the previous turn. Simplifications: the conversation is
represented by one frozen embedding of "history + query" (features "h"), the predictor is a two-layer MLP, the
switch penalty is a fixed constant in normalized-cost units, and cost is estimated at full input price.

Self-contained: its own predictor and training loop; it uses only the shared turn embeddings. The decision is in
__init__.py (parts.SwitchCost).
"""
import numpy as np
import torch
from torch import nn

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class _MLP(nn.Module):
    def __init__(self, d_in: int, k: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, 256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, k), nn.Sigmoid())

    def forward(self, x):
        return self.net(x)


class BaoModel:
    """fit(train conversations) -> predict(conversation) [T, K]."""

    def __init__(self, arms: list, loss: str = "mse", epochs: int = 60, seed: int = 0, patience: int = 8):
        self.arms, self.loss = arms, loss
        self.epochs, self.seed, self.patience = epochs, seed, patience
        self.net = None

    @staticmethod
    def _x(conv, emb) -> np.ndarray:
        return np.stack([emb[t.turn_id]["h"] for t in conv.turns])

    def _y(self, conv) -> np.ndarray:
        return np.array([[t.quality[a] for a in self.arms] for t in conv.turns], dtype=np.float32)

    def _loss(self, pred, y):
        if self.loss == "bce":
            return nn.functional.binary_cross_entropy(pred.clamp(1e-4, 1 - 1e-4), y)
        return ((pred - y) ** 2).mean()

    def fit(self, train: list, val: list, emb: dict) -> "BaoModel":
        torch.manual_seed(self.seed)
        xtr = torch.tensor(np.concatenate([self._x(c, emb) for c in train]), device=DEV)
        ytr = torch.tensor(np.concatenate([self._y(c) for c in train]), device=DEV)
        xva = torch.tensor(np.concatenate([self._x(c, emb) for c in val]), device=DEV)
        yva = torch.tensor(np.concatenate([self._y(c) for c in val]), device=DEV)
        net = _MLP(xtr.shape[1], len(self.arms)).to(DEV)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
        best, state, bad = 1e9, None, 0
        for _ in range(self.epochs):
            net.train()
            for idx in torch.randperm(len(xtr), device=DEV).split(64):
                opt.zero_grad()
                self._loss(net(xtr[idx]), ytr[idx]).backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                v = self._loss(net(xva), yva).item()
            if v < best - 1e-5:
                best, state, bad = v, {k: t.clone() for k, t in net.state_dict().items()}, 0
            elif (bad := bad + 1) >= self.patience:
                break
        net.load_state_dict(state)
        self.net = net.eval()
        return self

    def predict(self, conv, emb) -> np.ndarray:
        with torch.no_grad():
            return self.net(torch.tensor(self._x(conv, emb), device=DEV)).cpu().numpy()

    def state(self) -> dict:
        return {k: v.cpu() for k, v in self.net.state_dict().items()}

    def load_state(self, state: dict) -> "BaoModel":
        net = _MLP(state["net.0.weight"].shape[1], state["net.3.weight"].shape[0])
        net.load_state_dict(state)
        self.net = net.to(DEV).eval()
        return self
