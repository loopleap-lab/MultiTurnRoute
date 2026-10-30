"""Small synthetic datasets in the chat.jsonl format (with embeddings and prices), written to a temporary folder."""
import json
from pathlib import Path

import numpy as np

from sessionroutex.pricing import PRICES

MODELS = ["granite-3b", "gemma-e4b", "gemma-31b", "mistral-medium-128b"]


def write_dataset(root, n_conv: int = 16, d: int = 16, seed: int = 0, models: list = MODELS) -> Path:
    """chat.jsonl (every field of the real file), embeddings/emb_chat_granite3b.npz and prices.json under root.
    Conversations have 2-4 turns; every 4th conversation is in the test split."""
    root = Path(root)
    (root / "embeddings").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    w = rng.normal(size=(len(models), d))
    rows, ids, Q, A, H = [], [], [], [], []
    for i in range(n_conv):
        cid, n_turns, split = f"syn-{i:03d}", 2 + i % 3, "test" if i % 4 == 0 else "train"
        history, ctx_tokens = [], 0
        for t in range(1, n_turns + 1):
            tid = f"{cid}-t{t}"
            query = f"question {i}.{t} " + "word " * int(rng.integers(1, 30))
            answer = f"answer {i}.{t} " + "text " * int(rng.integers(5, 60))
            q, a, h = (rng.normal(size=d).astype(np.float32) for _ in range(3))
            qual = 1 / (1 + np.exp(-(w @ q))) * 0.8 + 0.2 * rng.random(len(models))
            n_q = len(query.split()) + 3
            responses, labels = {}, {}
            for k, m in enumerate(models):
                n_out = int(rng.integers(20, 400))
                responses[m] = {"text": f"reply of {m} to turn {t}", "n_prompt_tokens": ctx_tokens + n_q + k,
                                "n_output_tokens": n_out, "finish": "stop", "history_pairs_dropped": 0,
                                "query_truncated": False}
                s = 1 + 8 * float(qual[k])
                labels[m] = {"judge_expected_score": s, "judge_score": int(round(s)),
                             "judge_quality": (s - 1) / 8, "judge_repaired": False}
            rows.append({"dataset": "chat", "turn_id": tid, "conversation_id": cid, "turn": t, "n_turns": n_turns,
                         "source": "synthetic", "batch": "main", "split": split, "language": "English",
                         "category": "other", "context_dependent": t > 1, "history": list(history), "query": query,
                         "original_answer": answer, "responses": responses, "labels": labels,
                         "static_cost_usd": {m: 0.0 for m in models}})
            ids.append(tid), Q.append(q), A.append(a), H.append(h)
            history.append({"user": query, "assistant": answer})
            ctx_tokens += n_q + len(answer.split())
    with open(root / "chat.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    np.savez(root / "embeddings" / "emb_chat_granite3b.npz", ids=np.array(ids, dtype=object), q=np.stack(Q),
             a=np.stack(A), h=np.stack(H))
    (root / "prices.json").write_text(json.dumps(PRICES, indent=1))
    return root
