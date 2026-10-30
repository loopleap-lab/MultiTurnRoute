"""Small neural-network utilities shared by the methods: losses, an MLP head and its per-turn training loop."""
import numpy as np
import torch
from torch import nn

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOSS = {"kind": "mse"}  # "bce" for labels that are mostly 0/1; set once per run for every trained predictor


def targets(conv, models) -> np.ndarray:
    return np.array([[t.quality[m] for m in models] for t in conv.turns], dtype=np.float32)


def point_loss(pred, y):
    if LOSS["kind"] == "bce":
        return nn.functional.binary_cross_entropy(pred.clamp(1e-4, 1 - 1e-4), y)
    return ((pred - y) ** 2).mean()


def loss_fn(pred, y, rank_w: float, tau: float = 0.05):
    """Pointwise loss (MSE or BCE) plus an optional listwise term: cross-entropy between softmax(y/tau) and
    softmax(pred/tau) over models. The listwise term only rewards getting the models' order right within a turn."""
    loss = point_loss(pred, y)
    if rank_w > 0:
        loss = loss - rank_w * tau * (torch.softmax(y / tau, -1) * torch.log_softmax(pred / tau, -1)).sum(-1).mean()
    return loss


class Head(nn.Module):
    def __init__(self, d_in: int, k: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, 256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, k), nn.Sigmoid())

    def forward(self, x):
        return self.net(x)


def train_mlp(train, val, inputs, models, epochs, seed, rank_w=0.0):
    """Fit Head on per-turn inputs(conv) -> [T, d] against every model's quality; early stopping on val (patience 8).
    Returns predict(conv) -> [T, K]."""
    torch.manual_seed(seed)
    X = {c.conv_id: torch.tensor(inputs(c), device=DEV) for c in train + val}
    Y = {c.conv_id: torch.tensor(targets(c, models), device=DEV) for c in train + val}
    net = Head(X[train[0].conv_id].shape[1], len(models)).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    xtr, ytr = torch.cat([X[c.conv_id] for c in train]), torch.cat([Y[c.conv_id] for c in train])
    xva, yva = torch.cat([X[c.conv_id] for c in val]), torch.cat([Y[c.conv_id] for c in val])
    best, best_state, bad = 1e9, None, 0
    for _ in range(epochs):
        net.train()
        for idx in torch.randperm(len(xtr), device=DEV).split(64):
            opt.zero_grad()
            loss = loss_fn(net(xtr[idx]), ytr[idx], rank_w)
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            v = loss_fn(net(xva), yva, rank_w).item()
        if v < best - 1e-5:
            best, best_state, bad = v, {k: t.clone() for k, t in net.state_dict().items()}, 0
        elif (bad := bad + 1) >= 8:
            break
    net.load_state_dict(best_state)
    net.eval()
    return lambda c: net(torch.tensor(inputs(c), device=DEV)).detach().cpu().numpy()


def averaged(fs: list):
    """Average several predictors (e.g. trained with different seeds)."""
    return lambda c: np.mean([f(c) for f in fs], axis=0)
