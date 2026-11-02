# User guide

How to install MultiTurnServe, write its config, serve multi-turn sessions, train the built-in routers and run
the arena. [How MultiTurnServe works](architecture.md) explains the session state and the cost accounting;
[Routers](routers.md) and [Serving backends](backends.md) cover the two interfaces you can extend.

- [Installation](#installation)
- [The config](#the-config)
- [Serving](#serving)
- [Training the built-in routers](#training-the-built-in-routers)
- [The data](#the-data)
  - [Toy data](#toy-data)
- [Running the arena](#running-the-arena)
- [Track rules](#track-rules)
- [Reproducing the leaderboard](#reproducing-the-leaderboard)

## Installation

From source:

```bash
git clone https://github.com/UCF-ML-Research/MultiTurnServe.git && cd MultiTurnServe
pip install -e ".[serve,test]"      # the package, the `multiturnserve` command, the gateway (and pytest)
pip install -e ".[encoder]"         # transformers, for live embeddings with granite-4.2-3b
pytest -q                           # synthetic data, CPU, about 10 seconds
```

MultiTurnServe needs Python 3.10 or later. The package is not on PyPI yet.

| Extra | Adds | Needed for |
|---|---|---|
| `serve` | fastapi, uvicorn | `multiturnserve serve` |
| `encoder` | transformers | serving the built-in routers (they read turn embeddings) |
| `test` | pytest, fastapi, httpx | the tests |

`multiturnserve list` prints the registered routers and serving backends. `multiturnserve --help` and
`multiturnserve <command> --help` print every option.

### Environment variables

| Variable | What | Default |
|---|---|---|
| `MTS_DATA_DIR` | the data folder (see [The data](#the-data)) | `./data` |
| `MTS_HIDDEN_DIR` | the hidden test labels, `labels.jsonl` (organizers only) | none |
| `MTS_RUNS` | where `serve` looks for a trained router, `<MTS_RUNS>/<router>/router.pt` | `./runs` |
| `VLLM_API_KEY`, `OPENROUTER_API_KEY` | API keys of the serving backends | none |
| `MTS_USER_API_KEY`, `MTS_JUDGE_API_KEY` | API keys of the dynamic track's user simulator and judge | none |

Keys are read from the environment when a request is sent. They are never written to a config, a log or a
submission.

## The config

The config is a YAML file:

```yaml
router: cor                      # a router name, or a .py file
serving: [vllm, openrouter]      # backends, in this order
models:                          # the models of each backend (the pool, in this order)
  vllm: [gemma-e4b, gemma-26b-a4b]
  openrouter: [gemma-31b, mistral-medium-128b]
log: logs/                       # one JSONL line per answered turn (no message texts)
# optional
lam: 0.5                         # cost weight of the router
checkpoint: runs/cor             # a trained router (default: $MTS_RUNS/<router>, else runs/<router>)
encoder: granite                 # granite (default for the built-in routers), a Hugging Face id, none, or hash:D (tests)
plugins: [my_router.py]          # extra files that register routers or backends
backends:                        # options per backend
  vllm: {base_url: "http://localhost:8000/v1", model_ids: {gemma-e4b: my-org/gemma-e4b}, timeout: 120}
  openrouter: {prices: {gemma-31b: {input: 0.2, cached: 0.04, output: 0.4}}}
```

- Any other key is an error.
- The pool is the models of every backend, in the order of `serving:` and then `models:`. Ties in a decision go
  to the model listed first. A model may belong to one backend only.
- `.py` files in `routers/` and `backends/` folders next to the config are loaded, so `router: my-router` finds
  `routers/my_router.py`. A file anywhere else is loaded only when it is listed under `plugins:`.
- Relative paths in `router` and `plugins` are relative to the config. `log` and `checkpoint` are relative to
  the working directory.
- `lam` overrides the value saved with a trained router (0.5 by default).
- `encoder` defaults to `granite` for routers that read turn embeddings (all six built-ins) and to none for the
  others. `hash` or `hash:D` gives stand-in embeddings for tests.
- `backends:` holds the options of each backend, listed in [Serving backends](backends.md#options). A backend's
  prices default to the static track's price table.

`configs/serve.yaml` is the config above without the optional keys. `examples/config.yaml` serves an example
router with the `echo` backend and needs no GPU and no keys. It sets `lam: 0.2`, which weighs quality over cost,
so the example router moves to a larger model after the first turn.

The log is `<log>/turns.jsonl`, one line per answered turn: `chat_id, turn, model, n_prompt_tokens,
n_cached_tokens, n_output_tokens, cost, quality_hat, cost_hat, time`. `quality_hat` and `cost_hat` are the
router's prediction for the chosen model, when the router made its choice with `pick`.

## Serving

### Starting the gateway

```bash
multiturnserve serve CONFIG [--host 127.0.0.1] [--port 8080] [--router NAME|FILE.py] [--log DIR] [--check]
```

`serve` loads the router, the backends and the encoder from the config and serves the gateway (needs `[serve]`).
`--check` loads everything, prints it and exits. `--router` and `--log` override the config. A relative
`--router FILE.py` is resolved against the config's folder, like `router:`.

A built-in router must be trained first (see [Training the built-in routers](#training-the-built-in-routers)):

```bash
multiturnserve train --router cor --out runs/cor
multiturnserve serve configs/serve.yaml
```

To try the gateway without a GPU or keys:

```bash
multiturnserve serve examples/config.yaml
python examples/app.py
```

### The OpenAI-compatible API

| Endpoint | Body | Returns |
|---|---|---|
| `POST /v1/chat/completions` | OpenAI chat completions request; the session is named by a `chat_id` field, an `X-Chat-Id` header or the `user` field | an OpenAI `chat.completion` (`model` = the chosen model, `usage` with `prompt_tokens_details.cached_tokens`) plus `multiturnserve: {chat_id, turn, model, cost, decision: {quality, cost}}` |
| `POST /v1/route` | `{chat_id, messages}` | `{model, quality, cost, turn}`; the state does not change |
| `POST /v1/observe` | `{chat_id, model, text, usage: {prompt_tokens, cached_tokens, output_tokens}[, cost][, query]}` | the session's summary |
| `GET /v1/sessions/{chat_id}` | | `{chat_id, turns, total_cost, kv_cache, models}` |
| `GET /v1/models` | | the served models |
| `GET /health` | | the router and the models |

- A request carries the whole session so far (`messages`), as OpenAI clients send it. The query of the turn is
  the last user message.
- An existing OpenAI client works once it names the session, for example with the `user` field or an
  `X-Chat-Id` header.
- The gateway keeps one session per `chat_id` in memory. A restart starts every session afresh. A new Python
  client (`multiturnserve.open(..., gateway=...)`) refuses a `chat_id` the gateway already has turns for.
- If the chosen model's backend fails, the gateway answers HTTP 502 and the session's state is unchanged.
- `/v1/route` and `/v1/observe` are for apps that call the model themselves: ask for the model, call it, then
  report the reply and its usage. Without `cost`, `observe` prices the usage with the backend's prices.

### The Python session API

`multiturnserve.open(chat_id, ...)` opens one session. It talks to a running gateway, or runs the session in
your process from a config:

```python
import multiturnserve

s = multiturnserve.open("session-1", gateway=":8080")      # a running gateway
reply = s.chat("Summarize this paper")                       # route, run, observe
print(reply.turn, reply.model, f"{reply.cost:.2e}")
```

`gateway` takes `":8080"`, `"host:8080"`, a URL, or an HTTP client object with `.post` and `.get` (for example
an `httpx.Client` or FastAPI's `TestClient`). `chat(msg)` returns `Reply(chat_id, turn, model, text, cost, usage)`
and keeps the messages of the session for the next call.

Each turn is three calls, which you can also make one by one. `messages` is the session so far as an OpenAI-style
list of `{"role", "content"}` dicts, ending with the user's new message:

```python
s = multiturnserve.open("session-1", config="examples/config.yaml")   # a session in this process, no gateway
messages = [{"role": "user", "content": "Summarize this paper"}]
d = s.route(messages)     # -> Decision(model, quality, cost, t)
o = s.run(d, messages)    # the chosen model answers -> Outcome(model, text, usage, cost, query)
s.observe(o)              # history and KV cache updated, one log line written
```

| Call | Gateway session | Session in this process |
|---|---|---|
| `chat(msg)` | yes | yes |
| `route(messages)` | yes (`POST /v1/route`) | yes |
| `run(decision, messages)` | no: call the model yourself, then `observe` | yes |
| `observe(outcome)` | returns the summary (dict) | returns the new `SessionState` |
| state | `state()`: the summary from `GET /v1/sessions/{chat_id}` | `s.state` (`SessionState`) and `s.summary()` |

Use either `chat()` or the three calls in one session, not both. `chat()` keeps its own message list and sends
it to the model; turns made with `route`, `run` and `observe` update the session's history and KV cache state but
are not added to that list, so a later `chat()` call would send the model a session without them.

`multiturnserve.open(chat_id, runtime=rt)` opens a session on an existing `multiturnserve.Runtime`, so several
sessions share one router, one set of backends and one log. `Runtime.from_config(config)` builds one from a YAML
path or a dict.

## Training the built-in routers

```bash
multiturnserve train --router NAME|FILE.py --out DIR [training options]
```

`train` trains the router on every training conversation of `$MTS_DATA_DIR` (or `--data`) and writes
`DIR/router.pt`. It holds the weights, the models the router was trained on, `lam` and the shared output-length
heads. `serve` finds `runs/<router>/router.pt` by default (`checkpoint:` in the config, or `MTS_RUNS`). The
models of the serving pool must be among the models the router was trained on.

Training options (also taken by `arena replay`):

| Option | Default | Meaning |
|---|---|---|
| `--data` | `$MTS_DATA_DIR`, else `./data` | data folder |
| `--models` | the static track's 9 | model pool, in this order (ties go to the first; the predictors' outputs follow it) |
| `--batch` | all | `arena replay` only: use one sampling batch of the data, e.g. `pilot` |
| `--seed` | 0 | random seed |
| `--n-models` | 1 | cor, single-turn, full-history, rag: average the trained predictor over this many seeds (`seed + 100 * i`) |
| `--epochs` | 60 | at most; early stopping on a 10% hold-out of the training conversations (patience 8) |
| `--loss` | `mse` | `mse` or `bce`, for every trained predictor |
| `--rank-w` | 0 | cor, single-turn, full-history, rag: weight of a listwise ranking term over models |
| `--plugin FILE.py ...` | | extra files that register routers |
| `--dsp` | 1 | cor: 1 = DSP; 2 = a variant that projects every input to `--dsp-p` dims and also reads the history embedding |
| `--window` | 2 | cor: local attention window (turns) |
| `--dsp-p`, `--dsp-drop` | 128, 0.5 | cor, `--dsp 2` only: projection width, dropout |
| `--switch-penalty` | 0.1 | bao: penalty for changing models, in normalized-cost units |

## The data

`$MTS_DATA_DIR` holds:

```
chat.jsonl                           one JSON object per conversation turn
embeddings/emb_chat_granite3b.npz    one precomputed embedding per turn: ids, q (query), a (original answer),
                                     h (last 2,000 characters of history + query); granite-4.2-3b, layer 30,
                                     mean-pooled, L2-normalized, 2560-d
prices.json                          USD per 1M tokens per model: input, output, cached
```

Fields of `chat.jsonl`:

| Field | What |
|---|---|
| `turn_id`, `conversation_id`, `turn`, `n_turns` | ids; turn counts from 1 |
| `source`, `batch`, `split`, `language`, `category`, `context_dependent` | metadata; `split` is `train` or `test` |
| `history` | the earlier turns, `[{"user", "assistant"}]` |
| `query`, `original_answer` | this turn's user message and the answer of the original conversation |
| `responses[model]` | `text`, `n_prompt_tokens`, `n_output_tokens`, `finish`, `history_pairs_dropped`, `query_truncated` |
| `labels[model]` | `judge_quality` in [0, 1] = `(judge_expected_score - 1) / 8`, `judge_expected_score`, `judge_score`, `judge_repaired` |
| `static_cost_usd[model]` | the response's cost with every prompt token at the full price |

- Every model answered every turn on the original history.
- Only the longest prefix of consecutive turns where every model in the pool has a label is used.
- In the public release the test turns have no `labels` and no response `text`.
- The dynamic track's logs add `n_cached_tokens`, the cached prompt tokens the backend reported.

The data have no public download location yet (see the [roadmap](../ROADMAP.md#6-data-and-checkpoints)).

### Toy data

`tests/synth.py` writes a small synthetic data folder in the same format, for trying `train` and `arena replay`
without the real data. Its numbers mean nothing. From the repository root:

```bash
python -c "from tests.synth import write_dataset; from multiturnserve.arena.data import MODELS; write_dataset('toy', n_conv=40, models=MODELS)"
export MTS_DATA_DIR=toy
multiturnserve train --router cor --out runs/cor                                  # about 10 seconds on CPU
multiturnserve arena replay --name toy-builtins                                   # the six built-in routers
multiturnserve arena replay --router examples/my_router.py --name toy-mine        # the example router
```

The toy data include test labels, so `arena replay` runs on their test split without `--dev` (`--dev` works too).
`--name` keeps the two runs apart: `runs/arena/toy-builtins/results.json` and `runs/arena/toy-mine/results.json`.
`models=MODELS` gives every turn the static track's 9 models, so no `--models` is needed. The toy embeddings are
16-dimensional; to serve a router trained on them, set `encoder: hash` in the config.

## Running the arena

MultiTurnServe Arena compares routers by answer quality and cost, in two tracks. The
[arena page](../arena/README.md) gives a short overview; the [leaderboard](../LEADERBOARD.md) holds the current
results.

### Static track

```bash
multiturnserve arena replay --router examples/my_router.py --dev                      # on a held-out part of the training split
multiturnserve arena submit examples/my_router.py --files examples/my_lab.py           # pack the router files for the organizers
```

#### arena replay

```bash
multiturnserve arena replay [--router NAME|FILE.py ...] [--hidden DIR] [--dev] [--folds K] [--bootstrap N]
                           [--name static] [--out runs/arena] [training options]
```

`replay` trains each router on the training split, replays it on the test split at 41 values of `lam`, and
writes `<out>/<name>/results.json`. The default routers are the six built-ins.

- Test labels come from the data itself (the organizers' full copy) or from `--hidden` / `MTS_HIDDEN_DIR`.
- Without test labels the replay stops and asks for `--dev`, which replays on a fifth of the training
  conversations instead.
- `--folds K` replays K folds over all conversations.
- `--bootstrap N` adds a paired bootstrap over test conversations.

`results.json`:

| Field | What |
|---|---|
| `policies[label]` | `AUDC`, `QNC`, `QNC_hull`, `peak_quality`, `curve` (41 points of `[lam, cost, quality]`), for every router and the oracle (true quality and cost) |
| `fixed_models`, `best_single_model` | `[cost, quality]` of every model used alone; the one with the highest quality |
| `n_test_turns` | turns replayed |
| `quality_mse_per_split` | squared error of every quality predictor, all turns and split by context dependence |
| `bootstrap` | with `--bootstrap N`: 95% intervals of AUDC and QNC per router, and pairwise win shares `AUDC_win[a][b]`, `QNC_win[a][b]` |
| `config`, `models` | the arguments of the run; the model pool |

[Cost accounting](architecture.md#cost-accounting) defines the normalized cost the curves use.

#### arena leaderboard

```bash
multiturnserve arena leaderboard RESULTS.json [--out LEADERBOARD.md] [--json site.json] [--note TEXT] [--source TEXT]
```

`leaderboard` ranks the routers by AUDC and adds, for each:
- QNC, the cost to reach the best single model's quality;
- that cost as a share of the best single model's cost;
- peak quality.

The best single model is the reference line. A second table gives each router's best `quality - w * cost` on its
curve for a few score weights `w` (cost as a share of the best single model's cost, as on the website). `w` is
not the router's cost weight `lam`: `lam` sets each pick and traces the curve; `w` picks a point on it afterwards. `--json`
writes the same rows, with the curves, for the website. `--note` adds a paragraph under the tables and `--source`
names the source in the text.

#### arena submit (static track)

```bash
multiturnserve arena submit my_router.py [--files my_lab.py ...] [--out submission.tar.gz] [--force]
```

`submit` checks that the file registers one `Router` and packs it into a tarball for the organizers' replay,
with:
- the other files it imports (`--files`);
- a `meta.json` with the router name, the sha256 of every file and the package version.

If the router file imports a module that sits next to it (a `.py` file or a package) and that module is not in
`--files`, `submit` prints a warning. It does not replace an existing tarball unless you pass `--force`. Nothing
is uploaded.

#### arena release (organizers)

```bash
multiturnserve arena release --data FULL_DIR --out PUBLIC_DIR [--hidden-out PRIVATE_DIR]
```

`release` writes the public data:
- training turns with every field;
- test turns without `labels` (`judge_quality`, `judge_expected_score`, `judge_score`, `judge_repaired`),
  without the response texts and without any other field named like a score, a quality or a label. The test
  turns keep their inputs (history, query, original answer) and every model's token counts;
- the embeddings and `prices.json`, copied.

`--hidden-out` gets `labels.jsonl` (`{"turn_id", "labels"}` per test turn), the file `MTS_HIDDEN_DIR` points at.
It may not be inside `--out`.

### Dynamic track

```bash
multiturnserve arena run --config POOL.yaml --tasks TASKS.jsonl --out LOG --user-url URL
                        [--router NAME|FILE.py] [--lams L ...] [--repeats 1] [--limit N] [--max-turns T]
                        [--user-model DeepSeek-V4.1-Flash-1017] [--user-key-env MTS_USER_API_KEY]
multiturnserve arena judge --log LOG --judge-url URL [--judge-model NVIDIA-Nemotron-3.5-Super]
                          [--judge-key-env MTS_JUDGE_API_KEY]
multiturnserve arena score --log LOG [--reference OTHER_LOG]
multiturnserve arena submit --log LOG [--out FILE.tar.gz] [--force]
```

`run` plays each conversation live:
- `POOL.yaml` is a serve config: your router, backends and models.
- `TASKS.jsonl` is a file you provide, in the `chat.jsonl` format (the test conversations are not public yet);
  only `conversation_id`, `turn` and `query` are read. Each conversation becomes a task: its goal is the first
  user message, and the simulated user also sees the original user messages.
- Every conversation is played once per cost weight and repeat. `--lams` defaults to the 5 values
  `k / (1 + k)` for k = 0.01, 0.1, 1, 10, 100.
- Turn 1 is the original first message. Then the user simulator writes each user turn, or ends the
  conversation, until `--max-turns` (default: the number of original user turns).
- `run` writes `LOG/meta.json` and `LOG/log.jsonl`, one line per answered turn: `conversation_id, chat_id,
  lam_index, lam, repeat, turn, query, model, text, n_prompt_tokens, n_cached_tokens, n_output_tokens, cost`.
- A stopped run continues where it stopped.

`judge` rates every logged answer in its own conversation, 1 to 9, and writes `LOG/judged.jsonl` (`chat_id,
turn, judge_score, judge_quality = (score - 1) / 8, judge_model, prompt_version`). It is resumable too.

`score` reports, per cost weight, the mean judge quality, the mean normalized cost per turn and the bill in USD;
then AUDC and peak quality over the cost weights. `--reference` (another scored log, for example a router that
always picks one model) adds QNC and QNC_hull against its peak quality. It writes `LOG/score.json`.

`submit --log` packs `meta.json`, `log.jsonl`, `judged.jsonl`, `score.json` (and the router file, if `run` was
given one) into `LOG/submission.tar.gz`.

The user simulator and the judge are any OpenAI-compatible endpoints. `--user-url` and `--judge-url` are base
URLs, for example `https://host/v1`, and the model names must be the ids those endpoints use. Both run at
temperature 0 with the fixed prompts in `multiturnserve/arena/dynamic.py` (`PROMPT_VERSION`).

## Track rules

Static track:

- Fixed for everyone: the 9 models and their order, the price table and the 41 cost weights
  (`multiturnserve/arena/static.yaml`, copied in `configs/arena/static.yaml`), the train/test split, the judge
  labels.
- Training uses the public training split only; the test labels stay with the organizers. A submission is a
  router file (`decide()`, optionally `fit(ctx)`). The organizers replay it with `multiturnserve arena replay`
  from this repository's main branch, not from the submission.
- Every model answered the same original history, so a pick does not change later turns. The quality of a pick
  is the judge's rating of that model's answer; its cost follows the KV cache along the routed path.
- Reported: AUDC, QNC_hull (cost to reach the best single model's quality, gemma-31b), peak quality; with
  `--bootstrap`, 95% intervals and pairwise win shares over conversations.

Dynamic track:

- The submitter chooses the pool, the backends and the data and pays for the calls. The user simulator is
  DeepSeek-V4.1-Flash-1017 and the judge NVIDIA Nemotron 3.5 Super, at temperature 0 with the fixed prompts.
- Conversations are compared per task, not per turn: after the first turn each router faces its own history.
- Cost is the real bill: the backend's reported tokens, cached tokens included, at the backend's prices.
- Only runs with the same tasks, pool and simulator are comparable. Results are self-reported.

## Reproducing the leaderboard

The repository has no script for the paper's tables. The paper describes the static track (Section 5) but has no
table or figure with its numbers; its main results (Section 4) use a different, five-model pool. The
[leaderboard](../LEADERBOARD.md) shows the
evaluation package's reported run of the six built-in routers (seed 0, default options). `arena replay` is a port
of that package: the same data, seed and options give the same `results.json` fields. `tests/test_equivalence.py`
checks this on synthetic data against the original `evaluate.py`, which is kept as a test fixture in
`tests/reference/`.

The run needs the full data with the test labels, which the organizers hold:

```bash
multiturnserve arena replay --seed 0                  # with MTS_DATA_DIR and MTS_HIDDEN_DIR set
multiturnserve arena leaderboard runs/arena/static/results.json
```
