"""Dialogue State Propagator (DSP) of Chain-of-Route.

DSP (default): local state = cross-attention of the query over the last W turn embeddings x = [q; a]; global state =
GRU over all past x; gate alpha = sigmoid(W[q; H]); quality = head([q; alpha*H]).
DSP2 (--dsp 2): every input is first projected to P dims with dropout and the history-text embedding h joins the
query, followed by the same local attention + GRU + gate.
"""
import random

import numpy as np
import torch
from torch import nn

from ...arena.embeddings import conv_arrays
from ...parts.nn import DEV, Head, loss_fn, targets


class DSP(nn.Module):
    def __init__(self, k: int, d: int, window: int = 2):
        super().__init__()
        self.window, self.d = window, d
        self.wq, self.wk, self.wv = nn.Linear(d, d, bias=False), nn.Linear(2 * d, d, bias=False), nn.Linear(2 * d, d, bias=False)
        self.pos = nn.Parameter(torch.zeros(window, 2 * d))
        self.gru = nn.GRUCell(2 * d, d)
        self.gate = nn.Linear(d + 2 * d, 1)
        self.head = Head(d + 2 * d, k)

    def forward(self, q: torch.Tensor, x: torch.Tensor) -> tuple:
        """q: [T, d] query embeddings, x: [T, 2d] turn embeddings of one conversation -> ([T, K] quality, [T] alpha)."""
        d = self.d
        g = torch.zeros(d, device=q.device)
        outs, alphas = [], []
        for t in range(q.shape[0]):
            past = x[max(0, t - self.window):t]
            if len(past):
                kv = past + self.pos[-len(past):]
                att = torch.softmax(self.wq(q[t]) @ self.wk(kv).T / d ** 0.5, dim=-1)
                local = att @ self.wv(kv)
            else:
                local = torch.zeros(d, device=q.device)
            h = torch.cat([local, g])
            alpha = torch.sigmoid(self.gate(torch.cat([q[t], h])))
            outs.append(self.head(torch.cat([q[t], alpha * h])))
            alphas.append(alpha)
            g = self.gru(x[t].unsqueeze(0), g.unsqueeze(0)).squeeze(0)  # advance after the turn is answered
        return torch.stack(outs), torch.cat(alphas)


class DSP2(nn.Module):
    """Inputs per conversation: q [T, d], h [T, d], x [T, 2d] -> ([T, K] quality, [T] alpha)."""

    def __init__(self, k: int, d: int, window: int = 2, p: int = 256, dropout: float = 0.3):
        super().__init__()
        self.window, self.p = window, p
        self.pq, self.ph, self.px = nn.Linear(d, p), nn.Linear(d, p), nn.Linear(2 * d, p)
        self.drop = nn.Dropout(dropout)
        self.wq, self.wk, self.wv = nn.Linear(p, p, bias=False), nn.Linear(p, p, bias=False), nn.Linear(p, p, bias=False)
        self.pos = nn.Parameter(torch.zeros(window, p))
        self.gru = nn.GRUCell(p, p)
        self.gate = nn.Linear(4 * p, 1)
        self.head = nn.Sequential(nn.Linear(4 * p, 256), nn.ReLU(), nn.Dropout(dropout), nn.Linear(256, k), nn.Sigmoid())

    def forward(self, q: torch.Tensor, x: torch.Tensor, h: torch.Tensor) -> tuple:
        pq, ph, px = self.drop(self.pq(q)), self.drop(self.ph(h)), self.drop(self.px(x))
        g = torch.zeros(self.p, device=q.device)
        outs, alphas = [], []
        for t in range(q.shape[0]):
            past = px[max(0, t - self.window):t]
            if len(past):
                kv = past + self.pos[-len(past):]
                att = torch.softmax(self.wq(pq[t]) @ self.wk(kv).T / self.p ** 0.5, dim=-1)
                local = att @ self.wv(kv)
            else:
                local = torch.zeros(self.p, device=q.device)
            state = torch.cat([local, g])
            alpha = torch.sigmoid(self.gate(torch.cat([pq[t], ph[t], state])))
            outs.append(self.head(torch.cat([pq[t], ph[t], alpha * state])))
            alphas.append(alpha)
            g = self.gru(px[t].unsqueeze(0), g.unsqueeze(0)).squeeze(0)
        return torch.stack(outs), torch.cat(alphas)


def build_net(version: int, k: int, d: int, window: int, p: int, dropout: float) -> nn.Module:
    return DSP(k, d, window) if version == 1 else DSP2(k, d, window, p, dropout)


class DSPPredictor:
    """predict(conv, emb) -> [T, K] quality with a trained DSP or DSP2."""

    def __init__(self, net: nn.Module, version: int, config: dict):
        self.net, self.version, self.config = net, version, config

    def fwd(self, a):
        return self.net(a[0], a[1]) if self.version == 1 else self.net(a[0], a[1], a[3])

    def __call__(self, c, emb) -> np.ndarray:
        q, x, h = conv_arrays(c, emb)
        with torch.no_grad():
            out, _ = self.fwd((torch.tensor(q, device=DEV), torch.tensor(x, device=DEV), None, torch.tensor(h, device=DEV)))
        return out.cpu().numpy()

    def state(self) -> dict:
        return {"config": dict(self.config), "net": {k: v.cpu() for k, v in self.net.state_dict().items()}}

    @classmethod
    def from_state(cls, state: dict) -> "DSPPredictor":
        cfg = state["config"]
        net = build_net(cfg["version"], cfg["k"], cfg["d"], cfg["window"], cfg["p"], cfg["dropout"])
        net.load_state_dict(state["net"])
        return cls(net.to(DEV).eval(), cfg["version"], cfg)


def train_dsp(train, val, emb, models, epochs, window, seed, rank_w=0.0, version=1, p=128, dropout=0.5) -> DSPPredictor:
    """Train DSP (version 1) or DSP2 on whole conversations; returns predict(conv, emb) -> [T, K] quality."""
    torch.manual_seed(seed)
    d = conv_arrays(train[0], emb)[0].shape[1]
    net = build_net(version, len(models), d, window, p, dropout).to(DEV)
    fwd = (lambda a: net(a[0], a[1])) if version == 1 else (lambda a: net(a[0], a[1], a[3]))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    T = {c.conv_id: tuple(torch.tensor(a, device=DEV) for a in conv_arrays(c, emb)[:2])
         + (torch.tensor(targets(c, models), device=DEV), torch.tensor(conv_arrays(c, emb)[2], device=DEV))
         for c in train + val}

    def conv_loss(cs):
        pred = torch.cat([fwd(T[c.conv_id])[0] for c in cs])
        return loss_fn(pred, torch.cat([T[c.conv_id][2] for c in cs]), rank_w)

    def val_loss():
        net.eval()
        with torch.no_grad():
            return conv_loss(val).item()

    best, best_state, bad = 1e9, None, 0
    for _ in range(epochs):
        net.train()
        order = list(train)
        random.Random(seed + _).shuffle(order)
        for i in range(0, len(order), 16):
            opt.zero_grad()
            loss = conv_loss(order[i:i + 16])
            loss.backward()
            opt.step()
        v = val_loss()
        if v < best - 1e-5:
            best, best_state, bad = v, {k: t.clone() for k, t in net.state_dict().items()}, 0
        elif (bad := bad + 1) >= 8:
            break
    net.load_state_dict(best_state)
    net.eval()
    return DSPPredictor(net, version, {"version": version, "k": len(models), "d": d, "window": window, "p": p,
                                       "dropout": dropout})
