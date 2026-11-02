"""Dynamic track: whole conversations played live. A simulated user writes each user turn, the router picks a model,
the model answers, and the answer joins the history. Run, judge and score on your own machine with your own endpoints
(any OpenAI-compatible API; keys come from environment variables); submit packs the logs.

  run     task cards (chat.jsonl rows of the test conversations) -> <log>/log.jsonl, one line per answered turn
  judge   every answer, rated 1-9 in its own context            -> <log>/judged.jsonl (judge_quality = (score - 1) / 8)
  score   quality and bill per lambda, AUDC, peak (QNC against a reference run) -> <log>/score.json
  submit  meta, logs and score                                   -> <log>/submission.tar.gz

Conversations are compared per task, not per turn: after a few turns two routers no longer face the same question.
"""
from __future__ import annotations

import json
import re
import tarfile
from pathlib import Path

import numpy as np

from ..pricing import static_cost
from ..serving.http import chat_completion
from .replay import audc, qnc, qnc_hull

USER_MODEL = "DeepSeek-V4.1-Flash-1017"
JUDGE_MODEL = "NVIDIA-Nemotron-3.5-Super"
PROMPT_VERSION = "v1"
_KAPPA = (0.01, 0.1, 1.0, 10.0, 100.0)
LAMS = [k / (1 + k) for k in _KAPPA]     # 5 cost weights (the static track sweeps 41)

USER_PROMPT = ("You play the user in a chat with an AI assistant. Below are the user's goal and the user's original "
               "messages, in order. Write the user's next message and nothing else. Use the next original message if "
               "it still fits the conversation; if it does not, rewrite it toward the goal. If the goal is reached, "
               "reply with [END] only.")
JUDGE_PROMPT = ("You rate one assistant reply in a conversation. Score it from 1 (useless or wrong) to 9 (excellent), "
                "taking the conversation so far into account. Reply with the number only.")


def task_cards(path: str, limit: int | None = None) -> list:
    """One card per conversation: its id, the goal (the first user message) and the original user messages."""
    by_conv: dict = {}
    with open(path) as f:
        for r in map(json.loads, f):
            by_conv.setdefault(r["conversation_id"], []).append((r["turn"], r["query"]))
    cards = [{"conversation_id": cid, "goal": sorted(ts)[0][1], "user_turns": [q for _, q in sorted(ts)]}
             for cid, ts in by_conv.items()]
    return cards[:limit] if limit else cards


def transcript(messages: list) -> str:
    return "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in messages)


def next_user_message(card: dict, messages: list, url: str, model: str, key_env: str | None) -> str | None:
    """The simulated user's next message, or None when it ends the conversation."""
    script = "\n".join(f"{i + 1}. {q}" for i, q in enumerate(card["user_turns"]))
    prompt = f"Goal: {card['goal']}\n\nOriginal user messages:\n{script}\n\nConversation so far:\n{transcript(messages)}"
    r = chat_completion(url, model, [{"role": "system", "content": USER_PROMPT}, {"role": "user", "content": prompt}],
                        key_env, temperature=0.0)
    text = (r["choices"][0]["message"]["content"] or "").strip()
    return None if not text or text.startswith("[END]") else text


def _read(path: Path) -> list:
    return [json.loads(x) for x in open(path)] if path.exists() else []


def run(config, tasks: str, out: str, router: str | None = None, lams: list | None = None, repeats: int = 1,
        limit: int | None = None, max_turns: int | None = None, user_url: str = "", user_model: str = USER_MODEL,
        user_key_env: str | None = "MTS_USER_API_KEY") -> Path:
    """Play every task card once per lambda and repeat; a stopped run continues where it stopped."""
    from ..session import Runtime
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if router and router.endswith(".py"):
        router = str(Path(router).resolve())
    rt = Runtime.from_config(config, router=router)
    lams = list(lams or LAMS)
    meta = {"track": "dynamic", "router": rt.router.name, "router_file": router if router and router.endswith(".py")
            else None, "models": rt.models, "prices": {m: rt.serving[m].prices(m) for m in rt.models},
            "serving": {m: rt.serving[m].name for m in rt.models}, "lams": lams, "repeats": repeats,
            "tasks": Path(tasks).name, "user_model": user_model, "prompt_version": PROMPT_VERSION}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    log = out / "log.jsonl"
    done = {(r["conversation_id"], r["lam_index"], r["repeat"]) for r in _read(log)}
    for card in task_cards(tasks, limit):
        for i, lam in enumerate(lams):
            for rep in range(repeats):
                if (card["conversation_id"], i, rep) in done:
                    continue
                rt.router.lam = lam
                chat_id = f"{card['conversation_id']}|lam{i}|r{rep}"
                rt.sessions.pop(chat_id, None)
                s, rows = rt.session(chat_id), []
                for t in range(max_turns or len(card["user_turns"])):
                    msg = card["user_turns"][0] if t == 0 else \
                        next_user_message(card, s.messages, user_url, user_model, user_key_env)
                    if msg is None:
                        break
                    reply = s.chat(msg)
                    u = reply.usage
                    rows.append({"conversation_id": card["conversation_id"], "chat_id": chat_id, "lam_index": i,
                                 "lam": lam, "repeat": rep, "turn": reply.turn, "query": msg, "model": reply.model,
                                 "text": reply.text, "n_prompt_tokens": u.prompt_tokens,
                                 "n_cached_tokens": u.cached_tokens, "n_output_tokens": u.output_tokens,
                                 "cost": reply.cost})
                with open(log, "a") as f:          # one conversation at a time, so a stop never leaves half of one
                    for r in rows:
                        f.write(json.dumps(r) + "\n")
                rt.sessions.pop(chat_id, None)
    return log


def parse_score(text: str) -> int | None:
    m = re.search(r"\b([1-9])\b", text or "")
    return int(m.group(1)) if m else None


def judge(log_dir: str, judge_url: str = "", judge_model: str = JUDGE_MODEL,
          judge_key_env: str | None = "MTS_JUDGE_API_KEY") -> Path:
    """Rate every logged answer in its own context; a stopped run continues where it stopped."""
    d = Path(log_dir)
    rows, out = _read(d / "log.jsonl"), d / "judged.jsonl"
    done = {(r["chat_id"], r["turn"]) for r in _read(out)}
    by_chat: dict = {}
    for r in rows:
        by_chat.setdefault(r["chat_id"], []).append(r)
    with open(out, "a") as f:
        for chat_id, rs in by_chat.items():
            rs.sort(key=lambda r: r["turn"])
            for i, r in enumerate(rs):
                if (chat_id, r["turn"]) in done:
                    continue
                ctx = [m for p in rs[:i] for m in ({"role": "user", "content": p["query"]},
                                                     {"role": "assistant", "content": p["text"]})]
                prompt = (f"Conversation so far:\n{transcript(ctx) or '(none)'}\n\nUser: {r['query']}\n\n"
                          f"Assistant reply to rate:\n{r['text']}")
                resp = chat_completion(judge_url, judge_model, [{"role": "system", "content": JUDGE_PROMPT},
                                                                {"role": "user", "content": prompt}],
                                       judge_key_env, temperature=0.0)
                s = parse_score(resp["choices"][0]["message"]["content"])
                f.write(json.dumps({"chat_id": chat_id, "turn": r["turn"], "judge_score": s,
                                    "judge_quality": None if s is None else (s - 1) / 8, "judge_model": judge_model,
                                    "prompt_version": PROMPT_VERSION}) + "\n")
    return out


def score(log_dir: str, reference: str | None = None) -> dict:
    """Per lambda: mean judge quality, mean normalized cost per turn (cost / the most expensive pool model's full price
    for the same tokens) and the bill in USD; then AUDC and peak over the lambdas. With reference (another scored run,
    e.g. one model alone), QNC is the lowest cost that reaches the reference's peak quality."""
    d = Path(log_dir)
    meta = json.loads((d / "meta.json").read_text())
    q = {(r["chat_id"], r["turn"]): r["judge_quality"] for r in _read(d / "judged.jsonl")}
    prices, models = meta["prices"], meta["models"]
    pts = []
    for i, lam in enumerate(meta["lams"]):
        rows = [r for r in _read(d / "log.jsonl") if r["lam_index"] == i]
        rated = [r for r in rows if q.get((r["chat_id"], r["turn"])) is not None]
        if not rated:
            continue
        norm = [r["cost"] / max(static_cost(m, r["n_prompt_tokens"], r["n_output_tokens"], prices) for m in models)
                for r in rated]
        pts.append({"lam": lam, "cost": float(np.mean(norm)),
                    "quality": float(np.mean([q[(r["chat_id"], r["turn"])] for r in rated])),
                    "bill_usd": float(sum(r["cost"] for r in rows)), "turns": len(rows), "rated": len(rated),
                    "conversations": len({r["chat_id"] for r in rows})})
    cq = [(p["cost"], p["quality"]) for p in pts]
    res = {"router": meta["router"], "models": models, "points": pts, "AUDC": round(audc(cq), 4) if cq else None,
           "peak_quality": round(max(x for _, x in cq), 4) if cq else None, "bill_usd": sum(p["bill_usd"] for p in pts)}
    if reference:
        ref = json.loads((Path(reference) / "score.json").read_text())
        target = ref["peak_quality"]
        res.update({"reference": {"router": ref["router"], "quality": target}, "QNC": qnc(cq, target),
                    "QNC_hull": qnc_hull(cq, target)})
    (d / "score.json").write_text(json.dumps(res, indent=1))
    return res


def submit_router(router_file: str, out: str | None = None, extra: list | None = None) -> Path:
    """Static track: check that the file registers one Router, and pack it (with extra files it imports) and a
    meta.json (router name, sha256 of every file) into submission.tar.gz for the organizers to replay."""
    import hashlib
    from .. import __version__
    from ..registry import get_router
    from ..router import Router
    cls = get_router(str(Path(router_file).resolve()))
    if not issubclass(cls, Router) or cls.decide is Router.decide and cls.quality is None:
        raise SystemExit(f"{router_file}: the router must subclass Router and define decide()")
    files = [Path(router_file)] + [Path(f) for f in extra or []]
    meta = {"track": "static", "router": cls.name, "files": {f.name: hashlib.sha256(f.read_bytes()).hexdigest()
                                                             for f in files}, "multiturnserve": __version__}
    out = Path(out or "submission.tar.gz")
    with tarfile.open(out, "w:gz") as tar:
        for f in files:
            tar.add(f, arcname=f"router/{f.name}")
        data = json.dumps(meta, indent=1).encode()
        info = tarfile.TarInfo("meta.json")
        info.size = len(data)
        import io
        tar.addfile(info, io.BytesIO(data))
    return out


def submit(log_dir: str, out: str | None = None) -> Path:
    """Pack meta.json, log.jsonl, judged.jsonl, score.json (and the router file, when the run names one)."""
    d = Path(log_dir)
    meta = json.loads((d / "meta.json").read_text())
    out = Path(out or d / "submission.tar.gz")
    with tarfile.open(out, "w:gz") as tar:
        for name in ("meta.json", "log.jsonl", "judged.jsonl", "score.json"):
            if not (d / name).exists():
                raise SystemExit(f"{d / name} is missing: run arena run, judge and score first")
            tar.add(d / name, arcname=name)
        if meta.get("router_file") and Path(meta["router_file"]).exists():
            tar.add(meta["router_file"], arcname=f"router/{Path(meta['router_file']).name}")
    return out
