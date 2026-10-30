"""Precomputed turn embeddings shipped with the dataset (granite-4.2-3b layer-30 mean-pooled, L2-normalized):
q = query, a = original answer, h = last 2,000 characters of history + query."""
import numpy as np

from common.config import DATA_DIR

FILES = {"chat": "emb_chat_granite3b.npz", "lic": "emb_lic_granite3b.npz"}


def load(name: str, turn_ids: set | None = None) -> dict:
    """turn_id -> {"q", "a", "h"}. Each array is read once (z[key] re-reads the whole array on every access)."""
    with np.load(DATA_DIR / "embeddings" / FILES[name], allow_pickle=True) as z:
        ids, q, a, h = z["ids"].tolist(), z["q"], z["a"], z["h"]
    return {k: {"q": q[i], "a": a[i], "h": h[i]} for i, k in enumerate(ids) if turn_ids is None or k in turn_ids}


def conv_arrays(conv, emb) -> tuple:
    """q [T, d], x [T, 2d] = [q; a] per turn, h [T, d]."""
    q = np.stack([emb[t.turn_id]["q"] for t in conv.turns])
    x = np.stack([np.concatenate([emb[t.turn_id]["q"], emb[t.turn_id]["a"]]) for t in conv.turns])
    h = np.stack([emb[t.turn_id]["h"] for t in conv.turns])
    return q, x, h
