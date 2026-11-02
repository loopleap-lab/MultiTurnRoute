"""The `multiturnserve` command line.

  Serve       multiturnserve serve config.yaml [--port 8080]          the gateway your app talks to
              multiturnserve train --router cor --out runs/cor         train a router on $SRX_DATA_DIR
              multiturnserve list                                      routers and serving backends
  Static      multiturnserve arena replay --router cor bao             train, replay over 41 cost weights -> results.json
              multiturnserve arena leaderboard results.json            -> LEADERBOARD.md
              multiturnserve arena release --data DIR --out DIR        public data: test turns without labels
              multiturnserve arena submit my_router.py                 pack a router for the organizers' replay
  Dynamic     multiturnserve arena run --config serve.yaml --tasks test.jsonl --out runs/dyn/my-router
              multiturnserve arena judge --log runs/dyn/my-router
              multiturnserve arena score --log runs/dyn/my-router
              multiturnserve arena submit --log runs/dyn/my-router
Data come from $SRX_DATA_DIR (default ./data); the hidden test labels from $SRX_HIDDEN_DIR.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEFAULT_ROUTERS = ["bao", "cor", "full-history", "knn", "rag", "single-turn"]


def _training_args(p) -> None:
    p.add_argument("--data", help="data folder (default: $SRX_DATA_DIR, else ./data)")
    p.add_argument("--dataset", default="chat", choices=["chat", "lic"], help=argparse.SUPPRESS)
    p.add_argument("--models", nargs="*", help="model pool, in this order (default: the static track's 9)")
    p.add_argument("--batch", help="use one sampling batch of the data, e.g. pilot")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-models", type=int, default=1, help="average every trained predictor over this many seeds")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--loss", choices=["mse", "bce"], default="mse", help="pointwise loss of every trained predictor")
    p.add_argument("--rank-w", type=float, default=0.0, help="weight of a listwise ranking loss over models")
    p.add_argument("--plugin", nargs="*", default=[], help="extra .py files that register routers")
    g = p.add_argument_group("cor")
    g.add_argument("--dsp", type=int, choices=[1, 2], default=1,
                   help="1 = original DSP; 2 = projected variant that also reads the history embedding")
    g.add_argument("--window", type=int, default=2, help="DSP local attention window (turns)")
    g.add_argument("--dsp-p", type=int, default=128, help="--dsp 2 only: projection width")
    g.add_argument("--dsp-drop", type=float, default=0.5, help="--dsp 2 only: dropout")
    g = p.add_argument_group("bao")
    g.add_argument("--switch-penalty", type=float, default=0.1, help="normalized-cost penalty for changing models")


def _setup(a) -> None:
    import torch
    from .parts.nn import LOSS
    from .registry import load_plugin
    torch.set_num_threads(4)
    LOSS["kind"] = a.loss
    for p in a.plugin:
        load_plugin(p)


def _config(a) -> dict:
    return {k: v for k, v in vars(a).items() if k not in ("func", "cmd", "arena_cmd")}


def cmd_serve(a) -> None:
    from .session import Runtime
    rt = Runtime.from_config(a.config, router=a.router, log=a.log)
    print(f"  router   {rt.router.name}\n  serving  {', '.join(sorted({b.name for b in rt.serving.values()}))}\n"
          f"  models   {', '.join(rt.models)}", flush=True)
    if a.check:
        return
    from .gateway import serve
    print(f"  listening on {a.host}:{a.port}", flush=True)
    serve(rt, a.host, a.port)


def cmd_train(a) -> None:
    from .arena.train import train
    _setup(a)
    path = train(a.router, a, a.data, a.out)
    print(f"-> {path}")


def cmd_list(a) -> None:
    from .registry import ROUTERS, SERVING, list_routers, list_serving
    print("routers:  " + ", ".join(f"{n}{'' if ROUTERS[n].__module__.startswith('multiturnserve.') else ' (plugin)'}"
                                   for n in list_routers()))
    print("serving:  " + ", ".join(f"{n}{'' if SERVING[n].__module__.startswith('multiturnserve.') else ' (plugin)'}"
                                   for n in list_serving()))


def cmd_replay(a) -> None:
    from .arena import data, embeddings
    from .arena.replay import evaluate
    _setup(a)
    convs, models = data.load(a.dataset, a.models, a.batch, data=a.data, hidden=a.hidden)
    emb = embeddings.load(a.dataset, {t.turn_id for c in convs for t in c.turns}, data=a.data)
    print(f"{a.dataset}: {len(convs)} conversations, {sum(len(c.turns) for c in convs)} turns", flush=True)
    results = {"config": _config(a), "models": models, **evaluate(convs, emb, models, a, a.routers)}
    out = Path(a.out) / a.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, indent=1))
    print(f"\nbest single model: {results['best_single_model']} {results['fixed_models'][results['best_single_model']]}")
    print(f"{'policy':48s} {'AUDC':>7s} {'QNC_hull':>9s} {'peak':>7s}")
    for name, r in sorted(results["policies"].items(), key=lambda kv: -kv[1]["AUDC"]):
        qn = f"{r['QNC_hull']:.4f}" if r["QNC_hull"] is not None else "-"
        print(f"{name:48s} {r['AUDC']:7.4f} {qn:>9s} {r['peak_quality']:7.4f}")
    print(f"-> {out / 'results.json'}")


def cmd_leaderboard(a) -> None:
    from .arena.leaderboard import write
    write(a.results, a.out, a.json, a.note or "", a.source)
    print(f"-> {a.out}" + (f", {a.json}" if a.json else ""))


def cmd_release(a) -> None:
    from .arena.release import release
    stats = release(a.data, a.out, a.hidden_out)
    print(json.dumps(stats))
    print(f"-> {a.out}" + (f" (labels: {a.hidden_out}/labels.jsonl)" if a.hidden_out else ""))


def cmd_run(a) -> None:
    from .arena.dynamic import run
    path = run(a.config, a.tasks, a.out, a.router, a.lams, a.repeats, a.limit, a.max_turns, a.user_url,
               a.user_model, a.user_key_env)
    print(f"-> {path}")


def cmd_judge(a) -> None:
    from .arena.dynamic import judge
    print(f"-> {judge(a.log, a.judge_url, a.judge_model, a.judge_key_env)}")


def cmd_score(a) -> None:
    from .arena.dynamic import score
    res = score(a.log, a.reference)
    print(json.dumps({k: res.get(k) for k in ("router", "AUDC", "peak_quality", "QNC_hull", "bill_usd")}))
    print(f"-> {Path(a.log) / 'score.json'}")


def cmd_submit(a) -> None:
    from .arena.dynamic import submit, submit_router
    if bool(a.router) == bool(a.log):
        raise SystemExit("arena submit: give a router file (static track) or --log (dynamic track)")
    print(f"-> {submit_router(a.router, a.out, a.files) if a.router else submit(a.log, a.out)}")


def parser() -> argparse.ArgumentParser:
    from .arena.dynamic import JUDGE_MODEL, USER_MODEL
    ap = argparse.ArgumentParser(prog="multiturnserve", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("serve", help="run the gateway (OpenAI-compatible /v1/chat/completions)")
    p.add_argument("config", help="serve config: router, serving, models, log")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--router", help="override the config's router")
    p.add_argument("--log", help="override the config's log folder")
    p.add_argument("--check", action="store_true", help="load everything, print it and exit without serving")
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("train", help="train a router on the training split and save it")
    p.add_argument("--router", required=True, help="router name or my_router.py")
    p.add_argument("--out", required=True, help="checkpoint folder (serve finds runs/<router> by default)")
    _training_args(p)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("list", help="routers and serving backends")
    p.set_defaults(func=cmd_list)

    arena = sub.add_parser("arena", help="MultiTurnServe Arena: static and dynamic tracks")
    asub = arena.add_subparsers(dest="arena_cmd", required=True)

    p = asub.add_parser("replay", help="static track: train routers, replay them on the test split, write results.json")
    p.add_argument("--routers", "--router", nargs="+", default=DEFAULT_ROUTERS, dest="routers",
                   help="router names or .py files (default: the six built-ins)")
    p.add_argument("--hidden", help="folder with the hidden test labels (default: $SRX_HIDDEN_DIR)")
    p.add_argument("--folds", type=int, default=1, help="K > 1: K-fold over conversations instead of the test split")
    p.add_argument("--dev", action="store_true", help="replay on a fifth of the training conversations (no hidden labels)")
    p.add_argument("--bootstrap", type=int, default=0, help="paired bootstrap resamples over test conversations")
    p.add_argument("--name", default="static")
    p.add_argument("--out", default="runs/arena")
    _training_args(p)
    p.set_defaults(func=cmd_replay)

    p = asub.add_parser("leaderboard", help="results.json -> LEADERBOARD.md")
    p.add_argument("results")
    p.add_argument("--out", default="LEADERBOARD.md")
    p.add_argument("--json", help="also write the leaderboard as JSON (for the website)")
    p.add_argument("--note", help="a paragraph added under the table")
    p.add_argument("--source", help="how to name the source in the text (default: the results path)")
    p.set_defaults(func=cmd_leaderboard)

    p = asub.add_parser("release", help="public data: every training field; test inputs without labels")
    p.add_argument("--data", required=True, help="the full data folder")
    p.add_argument("--out", required=True, help="the public folder")
    p.add_argument("--hidden-out", help="where the test labels go (labels.jsonl; keep it private)")
    p.set_defaults(func=cmd_release)

    p = asub.add_parser("run", help="dynamic track: play task cards with a simulated user through your router")
    p.add_argument("--config", required=True, help="serve config: router, serving backends and models (your pool)")
    p.add_argument("--tasks", required=True, help="chat.jsonl rows of the conversations to play (test inputs)")
    p.add_argument("--out", required=True, help="log folder")
    p.add_argument("--router", help="override the config's router (name or .py file)")
    p.add_argument("--lams", nargs="*", type=float, help="cost weights (default: 5 values)")
    p.add_argument("--repeats", type=int, default=1, help="plays per task and cost weight")
    p.add_argument("--limit", type=int, help="only the first N tasks")
    p.add_argument("--max-turns", type=int, help="default: the task's number of original user turns")
    p.add_argument("--user-url", required=True, help="OpenAI-compatible base URL of the simulated user")
    p.add_argument("--user-model", default=USER_MODEL)
    p.add_argument("--user-key-env", default="SRX_USER_API_KEY", help="environment variable with its API key")
    p.set_defaults(func=cmd_run)

    p = asub.add_parser("judge", help="dynamic track: rate every logged answer")
    p.add_argument("--log", required=True)
    p.add_argument("--judge-url", required=True, help="OpenAI-compatible base URL of the judge")
    p.add_argument("--judge-model", default=JUDGE_MODEL)
    p.add_argument("--judge-key-env", default="SRX_JUDGE_API_KEY", help="environment variable with its API key")
    p.set_defaults(func=cmd_judge)

    p = asub.add_parser("score", help="dynamic track: quality and bill per cost weight, AUDC, peak (and QNC)")
    p.add_argument("--log", required=True)
    p.add_argument("--reference", help="another scored log folder (e.g. one model alone) for QNC")
    p.set_defaults(func=cmd_score)

    p = asub.add_parser("submit", help="pack a submission: a router file (static) or a run's logs (dynamic)")
    p.add_argument("router", nargs="?", help="static track: my_router.py")
    p.add_argument("--files", nargs="*", default=[], help="static track: other files the router imports")
    p.add_argument("--log", help="dynamic track: the log folder of arena run / judge / score")
    p.add_argument("--out", help="default: submission.tar.gz (static), <log>/submission.tar.gz (dynamic)")
    p.set_defaults(func=cmd_submit)
    return ap


def main(argv: list | None = None) -> None:
    a = parser().parse_args(argv)
    try:
        a.func(a)
    except (FileNotFoundError, KeyError, ValueError) as e:
        if "--debug" in (argv or sys.argv):
            raise
        sys.exit(f"multiturnserve: {e}")


if __name__ == "__main__":
    main()
