"""Live turn embeddings for online routing, in the format of the precomputed ones (q = query, a = answer,
h = last 2,000 characters of history + query). Offline replay reads the precomputed npz instead."""
import hashlib

import numpy as np

GRANITE = "ibm-granite/granite-4.2-3b"   # the encoder of the precomputed embeddings
HISTORY_CHARS = 2000


def history_text(history, query: str) -> str:
    """The text behind the "h" embedding: the last 2,000 characters of the history and the query."""
    s = "".join(f"User: {x.query}\nAssistant: {x.answer}\n" for x in history) + f"User: {query}"
    return s[-HISTORY_CHARS:]


class Encoder:
    """granite-4.2-3b: hidden states of layer 30, mean-pooled over the tokens, L2-normalized (2560-d), as the
    precomputed embeddings. Needs the [encoder] extra (transformers); the model downloads on first use."""

    def __init__(self, model: str = GRANITE, layer: int = 30, device: str | None = None, max_chars: int = 2000):
        import torch
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError:
            raise ImportError('live embeddings need the [encoder] extra: pip install -e ".[encoder]"') from None
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model)
        self.model = AutoModel.from_pretrained(model, torch_dtype=torch.float32).to(self.device).eval()
        self.layer, self.max_chars = layer, max_chars

    def __call__(self, texts: list) -> np.ndarray:
        import torch
        out = []
        for s in texts:
            enc = self.tok(s[:self.max_chars], return_tensors="pt").to(self.device)
            with torch.no_grad():
                hs = self.model(**enc, output_hidden_states=True).hidden_states[self.layer][0]
            v = hs.mean(0).float().cpu().numpy()
            out.append(v / max(np.linalg.norm(v), 1e-6))
        return np.stack(out).astype(np.float32)


class HashEncoder:
    """Deterministic stand-in embeddings from a hash of the text (tests and dry runs; they carry no meaning)."""

    def __init__(self, dim: int = 16):
        self.dim = dim

    def __call__(self, texts: list) -> np.ndarray:
        out = []
        for s in texts:
            seed = int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)
            v = np.random.default_rng(seed).normal(size=self.dim)
            out.append(v / np.linalg.norm(v))
        return np.stack(out).astype(np.float32)


def make_encoder(spec, dim: int | None = None):
    """An encoder from a config value: "granite" (default model), a Hugging Face id, "hash[:dim]" or None/"none"."""
    if spec is None or spec == "none":
        return None
    if not isinstance(spec, str):
        return spec
    if spec.startswith("hash"):
        return HashEncoder(int(spec.split(":")[1]) if ":" in spec else (dim or 16))
    return Encoder(GRANITE if spec == "granite" else spec)
