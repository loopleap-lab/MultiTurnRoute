# MultiTurnServe reference

The full reference behind the [README](../README.md): install, every command and option, the serve config, the
router and serving interfaces, the data format, the cost accounting and the rules of both arena tracks.

## Install

```bash
git clone <this repository> && cd MultiTurnServe
pip install -e ".[serve,test]"      # the package, the `multiturnserve` command, the gateway (and pytest)
pip install -e ".[encoder]"         # transformers, for live embeddings with granite-4.2-3b
pytest -q                           # synthetic data, CPU, about 10 seconds
```

| Extra | Adds | Needed for |
|---|---|---|
| `serve` | fastapi, uvicorn | `multiturnserve serve` |
| `encoder` | transformers | serving the built-in routers (they read turn embeddings) |
| `test` | pytest, fastapi, httpx | the tests |

| Variable | What | Default |
|---|---|---|
| `MTS_DATA_DIR` | the data folder (below) | `./data` |
| `MTS_HIDDEN_DIR` | the hidden test labels, `labels.jsonl` (organizers only) | none |
| `MTS_RUNS` | where `serve` looks for a trained router, `<MTS_RUNS>/<router>/router.pt` | `./runs` |
| `VLLM_API_KEY`, `OPENROUTER_API_KEY` | API keys of the serving backends | none |
| `MTS_USER_API_KEY`, `MTS_JUDGE_API_KEY` | API keys of the dynamic track's user simulator and judge | none |

Keys are read from the environment when a request is sent; they are never written to a config, a log or a
submission.

## Commands

`multiturnserve --help` and `multiturnserve <command> --help` print the same.

### serve

```bash
multiturnserve serve CONFIG [--host 127.0.0.1] [--port 8080] [--router NAME|FILE.py] [--log DIR] [--check]
```

Loads the router, the backends and the encoder from the config and serves the gateway (needs `[serve]`).
`--check` loads everything, prints it and exits. `--router` and `--log` override the config.

| Endpoint | Body | Returns |
|---|---|---|
| `POST /v1/chat/completions` | OpenAI chat request; the chat is named by a `chat_id` field, an `X-Chat-Id` header or the `user` field | an OpenAI `chat.completion` (`model` = the chosen model, `usage` with `prompt_tokens_details.cached_tokens`) plus `multiturnserve: {chat_id, turn, model, cost, decision: {quality, cost}}` |
| `POST /v1/route` | `{chat_id, messages}` | `{model, quality, cost, turn}`; the state does not change |
| `POST /v1/observe` | `{chat_id, model, text, usage: {prompt_tokens, cached_tokens, output_tokens}[, cost][, query]}` | the chat's state |
| `GET /v1/sessions/{chat_id}` | | `{chat_id, turns, total_cost, kv_cache, models}` |
| `GET /v1/models`, `GET /health` | | the served models; the router and models |

A request carries the whole chat (`messages`), as OpenAI clients send it. The query of the turn is the last user
message. The gateway keeps one session per chat id in memory; a restart starts every chat afresh.

### train

```bash
multiturnserve train --router NAME|FILE.py --out DIR [training options]
```

Trains the router on every training conversation of `$MTS_DATA_DIR` (or `--data`) and writes `DIR/router.pt`: the
weights, the model list it was trained on, `lam` and the shared output-length heads. `serve` finds
`runs/<router>/router.pt` by default (`checkpoint:` in the config, or `MTS_RUNS`).

Training options (also taken by `arena replay`):

| Option | Default | Meaning |
|---|---|---|
| `--data` | `$MTS_DATA_DIR`, else `./data` | data folder |
| `--models` | the static track's 9 | model pool, in this order (ties go to the first; the predictors' outputs follow it) |
| `--batch` | all | use one sampling batch of the data, e.g. `pilot` |
| `--seed` | 0 | random seed |
| `--n-models` | 1 | average every trained predictor over this many seeds (`seed + 100 * i`) |
| `--epochs` | 60 | at most; early stopping on a 10% hold-out of the training conversations (patience 8) |
| `--loss` | `mse` | `mse` or `bce`, for every trained predictor |
| `--rank-w` | 0 | weight of a listwise ranking term over models |
| `--plugin FILE.py ...` | | extra files that register routers |
| `--dsp` | 1 | cor: 1 = DSP; 2 = a variant that projects every input to `--dsp-p` dims and also reads the history embedding |
| `--window` | 2 | cor: local attention window (turns) |
| `--dsp-p`, `--dsp-drop` | 128, 0.5 | cor, `--dsp 2` only: projection width, dropout |
| `--switch-penalty` | 0.1 | bao: penalty for changing models, in normalized-cost units |

### list

`multiturnserve list` prints the registered routers and serving backends.

### arena replay

```bash
multiturnserve arena replay [--router NAME|FILE.py ...] [--hidden DIR] [--dev] [--folds K] [--bootstrap N]
                           [--name static] [--out runs/arena] [training options]
```

The static track: trains each router on the training split, replays it on the test split at 41 values of `lam`,
and writes `<out>/<name>/results.json`. The default routers are the six built-ins. Test labels come from the data
itself (the organizers' full copy) or from `--hidden` / `MTS_HIDDEN_DIR`; without them the replay stops and asks
for `--dev`, which replays on a fifth of the training conversations instead. `--folds K` replays K folds over all
conversations. `--bootstrap N` adds a paired bootstrap over test conversations.

`results.json`:

| Field | What |
|---|---|
| `policies[label]` | `AUDC`, `QNC`, `QNC_hull`, `peak_quality`, `curve` (41 points of `[lam, cost, quality]`), for every router and the oracle (true quality and cost) |
| `fixed_models`, `best_single_model` | `[cost, quality]` of every model used alone; the one with the highest quality |
| `n_test_turns` | turns replayed |
| `quality_mse_per_split` | squared error of every quality predictor, all turns and split by context dependence |
| `bootstrap` | with `--bootstrap N`: 95% intervals of AUDC and QNC per router, and pairwise win shares `AUDC_win[a][b]`, `QNC_win[a][b]` |
| `config`, `models` | the arguments of the run; the model pool |

The numbers are those of the evaluation package this code was ported from: the same data, seed and options give the
same `results.json` fields (tested on synthetic data in `tests/test_equivalence.py`).

### arena leaderboard

```bash
multiturnserve arena leaderboard RESULTS.json [--out LEADERBOARD.md] [--json site.json] [--note TEXT] [--source TEXT]
```

Writes the leaderboard: routers by AUDC with QNC, the cost to match the best single model as a share of that model's
cost, and peak quality; the best single model as the reference line. A second table gives each router's best
`quality - w * cost` on its curve for a few cost weights `w` (cost as a share of the best single model's cost, as on
the website). `--json` writes the same rows, with the curves, for the website.

### arena release

```bash
multiturnserve arena release --data FULL_DIR --out PUBLIC_DIR [--hidden-out PRIVATE_DIR]
```

The public data: training turns with every field; test turns without `labels` (`judge_quality`,
`judge_expected_score`, `judge_score`, `judge_repaired`), without the response texts and without any other field
named like a score, a quality or a label. The test turns keep their inputs (history, query, original answer) and
every model's token counts. The embeddings and `prices.json` are copied. `--hidden-out` gets `labels.jsonl`
(`{"turn_id", "labels"}` per test turn), the file `MTS_HIDDEN_DIR` points at; it may not be inside `--out`.

### arena submit (static track)

```bash
multiturnserve arena submit my_router.py [--files my_lab.py ...] [--out submission.tar.gz]
```

Checks that the file registers one `Router` and packs it, the other files it imports (`--files`) and a `meta.json`
(router name, sha256 of every file, package version) into a tarball for the organizers' replay. Nothing is uploaded.

### arena run, judge, score, submit (dynamic track)

```bash
multiturnserve arena run --config POOL.yaml --tasks TASKS.jsonl --out LOG --user-url URL
                        [--router NAME|FILE.py] [--lams L ...] [--repeats 1] [--limit N] [--max-turns T]
                        [--user-model DeepSeek-V4.1-Flash-1017] [--user-key-env MTS_USER_API_KEY]
multiturnserve arena judge --log LOG --judge-url URL [--judge-model NVIDIA-Nemotron-3.5-Super]
                          [--judge-key-env MTS_JUDGE_API_KEY]
multiturnserve arena score --log LOG [--reference OTHER_LOG]
multiturnserve arena submit --log LOG [--out FILE.tar.gz]
```

- `run`: `POOL.yaml` is a serve config (your router, backends and models). `TASKS.jsonl` is in the `chat.jsonl`
  format; only `conversation_id`, `turn` and `query` are read. Every conversation is played once per cost weight
  (`--lams`, default the 5 values `k / (1 + k)` for k = 0.01, 0.1, 1, 10, 100) and repeat. Turn 1 is the
  original first message; then the user simulator writes each user turn (or ends the chat) until `--max-turns`
  (default: the number of original user turns). Writes `LOG/meta.json` and `LOG/log.jsonl`, one line per answered
  turn: `conversation_id, chat_id, lam_index, lam, repeat, turn, query, model, text, n_prompt_tokens,
  n_cached_tokens, n_output_tokens, cost`. A stopped run continues where it stopped.
- `judge`: rates every logged answer in its own conversation, 1 to 9; writes `LOG/judged.jsonl`
  (`chat_id, turn, judge_score, judge_quality = (score - 1) / 8, judge_model, prompt_version`). Resumable.
- `score`: per cost weight, the mean judge quality, the mean normalized cost per turn (below) and the bill in USD;
  then AUDC and peak quality over the cost weights. `--reference` (another scored log, e.g. a router that always
  picks one model) adds QNC and QNC_hull against its peak quality. Writes `LOG/score.json`.
- `submit --log`: packs `meta.json`, `log.jsonl`, `judged.jsonl`, `score.json` (and the router file, if `run` was
  given one) into `LOG/submission.tar.gz`.

The user simulator and the judge are any OpenAI-compatible endpoints (`--user-url`, `--judge-url` are base URLs,
e.g. `https://host/v1`); the model names must be the ids those endpoints use. Both run at temperature 0 with the
fixed prompts in `multiturnserve/arena/dynamic.py` (`PROMPT_VERSION`).

## Serve config

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

Any other key is an error. Files in `routers/` and `backends/` next to the config are loaded, so `router: my-router`
finds `routers/my_router.py`. Relative paths in `plugins` and `router` are relative to the config. Options of the
OpenAI-compatible backends: `base_url`, `api_key_env` (the variable holding the key), `model_ids` (our name -> the
server's model id), `prices`, `timeout`, `extra` (more request fields). A backend's prices default to the static
track's price table.

The log line: `chat_id, turn, model, n_prompt_tokens, n_cached_tokens, n_output_tokens, cost, quality_hat,
cost_hat, time`.

## Interfaces

### Router

```python
class Router:
    name: str            # registry name (dashes; single_turn also finds single-turn)
    label: str           # name in results.json (default: name)
    lam: float = 0.5     # cost weight; replay sweeps it
    quality, cost        # parts the default decide() uses
    def decide(self, turn, state) -> str             # the model for this turn
    def pick(self, q, c) -> str                      # argmax_k (1 - lam) * q[k] - lam * c.values[k] / c.scale
    def fit(self, ctx) -> None                       # train (arena.train.Context); default: nothing
    def state(self) -> dict; def load_state(self, state)   # what save() / load_router() keep
```

Register it with `@register_router`. `turn` (`multiturnserve.Turn`):

| Field | What |
|---|---|
| `chat_id`, `t`, `query`, `messages` | the chat, the turn index from 0, the user message, the whole request (online) |
| `models` | the candidate models, in pool order |
| `n_prompt[model]` | prompt tokens the model would read: recorded offline; online, 4 characters per token |
| `out_tokens[k]` | predicted output tokens of `models[k]` (the shared length heads; 256 without them) |
| `prices[model]` | USD per 1M tokens: `input`, `cached`, `output` |
| `conv`, `emb` | the conversation so far and its turn embeddings `turn_id -> {q, a, h}` (online: with an encoder) |

`state` (`SessionState`): `history`, a tuple of `Exchange(query, answer, model)`, and `kv_cache`, a
`CacheState` with `held[model]` tokens and `cached(model, n_prompt)`.

Parts (`multiturnserve.parts`): `CacheCost` (each model's cached prefix at the cached price), `StaticCost` (every
prompt token at the full price), `SwitchCost(penalty)` (full price plus a penalty for leaving the previous model),
`QualityHead(predict, models)` (a trained predictor), `LengthHead` (output length), `Encoder` (live embeddings).
A cost part returns `Costs(values, scale, models, usd)`.

### Serving

```python
class Serving:
    name: str
    models: list
    def run(self, model, messages) -> (text, usage)   # usage: Usage(prompt, cached, output), a dict, or OpenAI's usage
    def prices(self, model) -> dict                    # USD per 1M tokens: input, cached, output
```

Register it with `@register_serving`; the config's `models:` and `backends:` entries become its constructor
arguments.

### Sessions

`multiturnserve.open(chat_id, gateway=":8080")` talks to a gateway: `chat(msg) -> Reply(chat_id, turn, model, text,
cost, usage)`, `route(messages) -> Decision(model, quality, cost)`, `observe(outcome)`, `state()`.
`multiturnserve.open(chat_id, config="config.yaml")` makes a local session with the same calls plus
`run(decision, messages) -> Outcome(model, text, usage, cost)`.

## Data

`$MTS_DATA_DIR`:

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

Every model answered every turn on the original history. Only the longest prefix of consecutive turns where every
model in the pool has a label is used. The dynamic track's logs add `n_cached_tokens`, the cached prompt tokens the
backend reported. In the public release the test turns have no `labels` and no response `text`.

## Cost accounting

Prices are USD per 1M tokens. A turn that reads `L` prompt tokens, of which `C` are cached, and writes `G` tokens
costs

```
cost = input * (L - C) + cached * C + output * G
```

Offline (static track), a model that served a turn holds that turn's prompt and its answer, `n_prompt + n_output`
tokens, for the rest of the conversation; at a later turn its cached part is `min(held, n_prompt)` and every other
model keeps what it held. Online, `C` is what the backend reports (`cached_tokens`), and the session's KV cache
follows the same rule from the reported usage. Normalized cost divides a turn's cost by the most expensive model's
cost for that turn at the full input price (`C = 0`); curves and AUDC use mean normalized cost per turn. In the
static track every learned router is also charged its encoder's input tokens: `(query + answer + 2,000)
characters / 4` at the input price of `granite-3b`, per turn.

A router's own estimate (`CacheCost`, `StaticCost`) uses the same formula with the predicted output length.

## Track rules

Static track:

- Fixed for everyone: the 9 models and their order, the price table and the 41 cost weights
  (`configs/arena/static.yaml`), the train/test split, the judge labels.
- Training uses the public training split only; the test labels stay with the organizers. A submission is a router
  file (`decide()`, optionally `fit(ctx)`); the organizers replay it with `multiturnserve arena replay` from this
  repository's main branch, not from the submission.
- Every model answered the same original history, so a pick does not change later turns. The quality of a pick is
  the judge's rating of that model's answer; its cost follows the KV cache along the routed path.
- Reported: AUDC, QNC_hull (cost to reach the best single model's quality, gemma-31b), peak quality; with
  `--bootstrap`, 95% intervals and pairwise win shares over conversations.

Dynamic track:

- The submitter chooses the pool, the backends and the data and pays for the calls. The user simulator is
  DeepSeek-V4.1-Flash-1017 and the judge NVIDIA Nemotron 3.5 Super, at temperature 0 with the fixed prompts.
- Conversations are compared per task, not per turn: after the first turn each router faces its own history.
- Cost is the real bill: the backend's reported tokens, cached tokens included, at the backend's prices.
- Only runs with the same tasks, pool and simulator are comparable. Results are self-reported.
