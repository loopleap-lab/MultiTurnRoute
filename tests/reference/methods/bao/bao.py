"""Simplified Bao et al. multi-turn router (not a reproduction of the original algorithm or code).

Idea kept: the quality of every model is predicted from the whole conversation so far, and the decision penalizes
switching to a different model than the one that served the previous turn. Simplifications: the conversation is
represented by one frozen embedding of "history + query" (features "h"), the predictor is a two-layer MLP, the
switch penalty is a fixed constant in normalized-cost units, and cost is estimated at full input price.

Self-contained: its own predictor, training loop and decision rule; it uses only the shared turn embeddings and
the shared price table, and plugs into the shared simulator through choose().
"""
import numpy as np
import torch
from torch import nn

from common.pricing import static_cost

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class _MLP(nn.Module):
    def __init__(self, d_in: int, k: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, 256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, k), nn.Sigmoid())

    def forward(self, x):
        return self.net(x)


class BaoRouter:
    """fit(train conversations) -> predict(conversation) [T, K] -> policy(...) for the shared simulator."""

    name = "full history + switch penalty (Bao-style)"

    def __init__(self, arms: list, emb: dict, switch_penalty: float = 0.1, loss: str = "mse", epochs: int = 60,
                 seed: int = 0, patience: int = 8):
        self.arms, self.emb, self.penalty, self.loss = arms, emb, switch_penalty, loss
        self.epochs, self.seed, self.patience = epochs, seed, patience
        self.net, self.q_hat, self.out_hat, self.overhead = None, {}, {}, None

    def _x(self, conv) -> np.ndarray:
        return np.stack([self.emb[t.turn_id]["h"] for t in conv.turns])

    def _y(self, conv) -> np.ndarray:
        return np.array([[t.quality[a] for a in self.arms] for t in conv.turns], dtype=np.float32)

    def _loss(self, pred, y):
        if self.loss == "bce":
            return nn.functional.binary_cross_entropy(pred.clamp(1e-4, 1 - 1e-4), y)
        return ((pred - y) ** 2).mean()

    def fit(self, train: list, val: list) -> "BaoRouter":
        torch.manual_seed(self.seed)
        xtr = torch.tensor(np.concatenate([self._x(c) for c in train]), device=DEV)
        ytr = torch.tensor(np.concatenate([self._y(c) for c in train]), device=DEV)
        xva = torch.tensor(np.concatenate([self._x(c) for c in val]), device=DEV)
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

    def predict(self, conv) -> np.ndarray:
        with torch.no_grad():
            return self.net(torch.tensor(self._x(conv), device=DEV)).cpu().numpy()

    def policy(self, test: list, out_hat: dict, overhead: dict | None = None) -> "BaoRouter":
        """Prepare predictions for these conversations; the router itself is the policy (see choose)."""
        self.q_hat = {c.conv_id: self.predict(c) for c in test}
        self.out_hat, self.overhead = out_hat, overhead
        return self

    def choose(self, conv, t: int, arms: list, state, prev, lam: float) -> int:
        """argmax of (1 - lam) * q_hat - lam * (static cost estimate / most expensive + switch penalty)."""
        turn, qh, oh = conv.turns[t], self.q_hat[conv.conv_id], self.out_hat[conv.conv_id]
        base_hat = max(static_cost(a, turn.n_prompt[a], oh[t][k]) for k, a in enumerate(arms))
        util = []
        for k, a in enumerate(arms):
            c_hat = static_cost(a, turn.n_prompt[a], oh[t][k]) / base_hat
            if prev is not None and k != prev:
                c_hat += self.penalty
            util.append((1 - lam) * qh[t][k] - lam * c_hat)
        return int(np.argmax(util))
