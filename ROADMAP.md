# MultiTurnServe Roadmap

MultiTurnServe routes every turn of a multi-turn session to one model of a pool, keeping the session's dialogue
history and KV cache state for the router. Its arena compares routers by quality and cost in a static and a
dynamic track.

Done items are checked, with the release that delivered them. Planned items are unchecked. In each section the
done items come first, then the planned items under Next and Later.

- [1. Sessions and serving](#1-sessions-and-serving)
- [2. Online serving](#2-online-serving)
- [3. Routers](#3-routers)
- [4. Static track](#4-static-track)
- [5. Dynamic track](#5-dynamic-track)
- [6. Data and checkpoints](#6-data-and-checkpoints)

## 1. Sessions and serving

- [x] Sessions with two states, the dialogue history and the KV cache each model holds (v0.1.0)
- [x] The router interface `decide(turn, state)`, shared by serving and the arena's replay (v0.1.0)
- [x] The serving interface `run(model, messages)` and `prices(model)`; the `vllm`, `openrouter` and `echo` backends (v0.1.0)
- [x] A gateway with an OpenAI-compatible `/v1/chat/completions`, plus `/v1/route` and `/v1/observe` (v0.1.0)
- [x] The Python session API: `multiturnserve.open`, with `chat`, `route`, `run` and `observe` (v0.1.0)
- [x] Routers and backends from plugin files (v0.1.0)

## 2. Online serving

- [x] The cost of every turn from the cached tokens the backend reports (v0.1.0)
- [x] Live turn embeddings with granite-4.2-3b for the built-in routers (v0.1.0)

Next:
- [ ] Check the live encoder against the script that made the precomputed embeddings. Its Hugging Face id,
  `ibm-granite/granite-4.2-3b` (layer 30, mean-pooled, L2-normalized), comes from the evaluation package's
  description. The text behind the live `h` embedding (`User: ...` / `Assistant: ...` lines, last 2,000
  characters) may differ from the precomputed one, so a router served online may see slightly different inputs
  than in training.
- [ ] Test `vllm` and `openrouter` against live servers (the tests use an in-process fake). vLLM reports cached
  tokens only with `--enable-prompt-tokens-details`; OpenRouter's reporting depends on the provider.

Later:
- [ ] Per-model tokenizers for the prompt lengths before a turn (today: 4 characters per token for every model)
- [ ] Sessions that persist across gateway restarts, with expiry
- [ ] Streaming responses
- [ ] Eviction and TTL in the KV cache state (today a model keeps a session's prefix for the whole session;
  eviction shows up in the bill but not in the router's estimate)
- [ ] A built-in table of provider model ids for the 9 models (today only through `model_ids`)

## 3. Routers

- [x] Six built-in routers: `cor`, `single-turn`, `full-history`, `rag`, `knn`, `bao` (v0.1.0)
- [x] Router parts: `CacheCost`, `StaticCost`, `SwitchCost`, `QualityHead`, `LengthHead`, `Encoder` (v0.1.0)
- [x] `multiturnserve train` and checkpoints that `serve` loads (v0.1.0)

Next:
- [ ] Adapters for multi-turn routers that read the whole session: MTRouter, DialRouter, GMTRouter
- [ ] Adapters for single-turn routers that read one message: RouteLLM, CARROT

Later:
- [ ] Ready-made quality parts from single-turn toolkits, such as a RouteLLM score next to `CacheCost`
- [ ] The vLLM Semantic Router session rule

`bao` stays a simplified reimplementation of Bao et al., not the authors' algorithm or code.

## 4. Static track

- [x] Replay of routers over 41 cost weights with hidden test labels; AUDC, QNC, QNC_hull, peak quality, paired bootstrap (v0.1.0)
- [x] The public release of the data with the test labels held apart (`arena release`) (v0.1.0)
- [x] Router submissions packed and checked (`arena submit my_router.py`) (v0.1.0)
- [x] The leaderboard of the six built-in routers (`arena leaderboard`) (v0.1.0)

Next:
- [ ] Organizer side: a container or sandbox for submitted code, a CI job that replays submissions from the main
  branch, and an upload (today `arena submit` only packs and checks the file)
- [ ] Leaderboard runs over several seeds, with bootstrap intervals (today one run, seed 0)

Later:
- [ ] Switch counts per router in `results.json`
- [ ] Version stamps of the dataset, pool, price table and judge on each result
- [ ] A rule on unsupervised tuning on the test inputs

## 5. Dynamic track

- [x] Live play with a user simulator and a judge: `arena run`, `judge`, `score`, `submit --log` (v0.1.0)
- [x] Resumable runs and judging (v0.1.0)

Next:
- [ ] Organizer side: re-judging of a sample, a check of logged tokens against the price table, a "verified" or
  "self-reported" label on each result, and an upload (today `arena submit` only writes a tarball, and results
  are self-reported)
- [ ] 3 plays per task with confidence intervals (`--repeats` exists; `score` reports means only)

Later:
- [ ] A per-task "goal reached" score
- [ ] A calibration run that replays the original user turns
- [ ] Single-model reference lines run by the organizers (today `--reference` takes any scored run)
- [ ] User personas in the task cards (today cards hold the goal and the original user turns)

## 6. Data and checkpoints

Next:
- [ ] A public download location for the released data (`arena release` makes the folder)
- [ ] Published checkpoints of the built-in routers (today `serve` with a built-in router needs
  `multiturnserve train` on the data first)
- [ ] The package on PyPI (today: install from the repository)
