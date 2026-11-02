# Not implemented yet

What the website (multiturnserve_site.html, arena.html) or the design documents show that this repository does not
do yet. Internal: keep this list in step with the code; the README lists only what runs.

## Routers

- **MTRouter, DialRouter** (website Toolkit, "look at the whole chat") and **RouteLLM, CARROT** ("look at one
  message"): no adapters. The README lists only the six that run (cor, single-turn, full-history, rag, knn, bao).
  ARENA_design.md also names GMTRouter for the first leaderboard.
- Ready-made quality parts from single-turn toolkits (e.g. a `RouteLLMScore()` part next to `CacheCost()`), and
  the vLLM Semantic Router session rule: not written.
- Bao et al. is the evaluation package's simplified reimplementation, not the authors' algorithm or code.

## Arena

- **Dynamic track, organizer side**: no re-judging of a sample, no check of logged tokens against the price table,
  no "verified / self-reported" label, no upload (`arena submit` only writes a tarball). Results are self-reported.
- Dynamic track, other parts of ARENA_design.md: 3 plays per task with confidence intervals (`--repeats` exists,
  `score` reports means only); per-task "goal reached" score; the calibration run that replays the original user
  turns; organizer-run single-model reference lines (`--reference` takes any scored run instead); user personas in
  the task cards (cards hold only the goal and the original user turns).
- **Static track, organizer side**: no container or sandbox for submitted code, no CI job that replays submissions
  from the main branch, no upload. `arena submit my_router.py` only packs and checks the file.
- Static track outputs not in `results.json`: switch counts per router; version stamps of dataset, pool, price table
  and judge on each result.
- The leaderboard is one run (seed 0) with no variance across seeds and no bootstrap intervals.
- Open rule (ARENA_design.md, 待定): whether unsupervised tuning on the test inputs is allowed.

## Data and checkpoints

- No public download location for the released data (`arena release` makes the folder; where it is published is
  open). No trained checkpoints are published (see the Hugging Face plan); `serve` with a built-in router needs
  `multiturnserve train` on the data first.
- The website says `pip install multiturnserve`; the package is not on PyPI (install from the repository).

## Online serving

- The encoder's Hugging Face id is set to `ibm-granite/granite-4.2-3b` (layer 30, mean-pooled, L2-normalized) from
  the evaluation package's description; it has not been checked against the script that made the npz, and the text
  behind the live "h" embedding ("User: ... / Assistant: ..." lines, last 2,000 characters) may differ from the
  precomputed one. Routers served online may therefore see slightly different inputs than in training.
- Prompt lengths before a turn are estimated at 4 characters per token for every model (no per-model tokenizer).
- Sessions live in the gateway's memory: no persistence across restarts, no expiry, no streaming responses.
- The KV cache state assumes a model keeps a chat's prefix for the whole chat (no eviction or TTL); the bill uses the
  cached tokens the backend reports, so eviction shows up in the cost but not in the router's estimate.
- vLLM reports cached tokens only when started with `--enable-prompt-tokens-details`; OpenRouter's cached-token
  reporting depends on the provider. Neither was tested against a live server (tests use an in-process fake).
- Model names map to provider ids only through `backends: {name: {model_ids: ...}}`; there is no built-in table of
  provider ids for the 9 models.
