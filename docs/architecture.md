# How MultiTurnServe works

MultiTurnServe routes every turn of a multi-turn session to one model of a pool. Each session keeps two states:
- the dialogue history: what was said, and which model answered each turn;
- the KV cache state: how many tokens of this session each model already holds.

A router reads both states and picks the model for the next turn. A serving backend runs the model and reports
its usage, new and cached tokens, which updates both states and gives the turn's cost. The arena runs the same
routers offline on recorded conversations, and online against a simulated user.

- [Layers](#layers)
- [One turn: route, run, observe](#one-turn-route-run-observe)
- [Cost accounting](#cost-accounting)
- [Offline and online](#offline-and-online)
- [Repository layout](#repository-layout)

## Layers

| Layer | Interface | Contents |
|---|---|---|
| App | `multiturnserve.open(chat_id, ...)`, or the gateway's OpenAI-compatible API | sends each user message of a session; gets the reply, the chosen model and the cost |
| Session | `Session`: `route`, `run`, `observe`, `chat` | the session state `SessionState(history, kv_cache)`: `history` is a tuple of `Exchange(query, answer, model)`, `kv_cache` a `CacheState` with the tokens `held[model]` |
| Router | `Router.decide(turn, state) -> model` | reads the turn (`Turn`) and the session state; the built-ins combine a quality part and a cost part with `pick` ([Routers](routers.md)) |
| Serving | `Serving.run(model, messages) -> (text, usage)`, `Serving.prices(model)` | runs one model call; reports prompt, cached and output tokens; gives USD prices per 1M tokens ([Serving backends](backends.md)) |
| Arena | `multiturnserve arena ...` | static track: replay on recorded conversations with hidden labels; dynamic track: live play with a user simulator and a judge ([User guide](user-guide.md#running-the-arena)) |

A `Runtime` holds what the sessions share: the router, the backends (model to backend), the encoder, the log
and the sessions by `chat_id`. The gateway serves one `Runtime`. Routers and backends are registered by name
(`@register_router`, `@register_serving`) and can come from plugin files.

## One turn: route, run, observe

Each turn of a session is three steps. `Session.chat(msg)` runs all three; the gateway's
`/v1/chat/completions` does the same per request.

1. **route(messages) -> Decision(model, quality, cost, t).** The session builds the `Turn` the router sees:
   - the query: the last user message;
   - the candidate models, in pool order, with each backend's prices;
   - `n_prompt[model]`, the prompt length, estimated at 4 characters per token;
   - `out_tokens[k]`, the predicted output tokens, from the router's length heads (256 without them, or without an encoder);
   - the turn embeddings, when the runtime has an encoder.

   The router's `decide(turn, state)` returns a model name. When the router chose with `pick`, the decision also
   carries the predicted quality and the estimated USD cost of that model. The state does not change.
2. **run(decision, messages) -> Outcome(model, text, usage, cost, query).** The chosen model's backend answers.
   The realized cost comes from the reported usage and the backend's prices.
3. **observe(outcome) -> SessionState.** The history gets the new exchange. In the KV cache, the chosen model
   now holds `prompt_tokens + output_tokens` of this session, and every other model keeps what it held. The
   session's total cost grows, and one line goes to the log.

An app that calls the model itself uses only route and observe: `POST /v1/route`, then its own call, then
`POST /v1/observe` with the reply and its usage.

## Cost accounting

Prices are USD per 1M tokens. A turn that reads `L` prompt tokens, of which `C` are cached, and writes `G` tokens
costs

```
cost = input * (L - C) + cached * C + output * G
```

- **Offline (static track).** A model that served a turn holds that turn's prompt and its answer,
  `n_prompt + n_output` tokens, for the rest of the conversation. At a later turn its cached part is
  `min(held, n_prompt)`, and every other model keeps what it held.
- **Online.** `C` is what the backend reports (`cached_tokens`). The session's KV cache follows the same rule
  from the reported usage.
- **Normalized cost.** A turn's cost divided by the most expensive model's cost for that turn at the full input
  price (`C = 0`). Curves and AUDC use the mean normalized cost per turn.
- **Encoder overhead.** In the static track every learned router is also charged its encoder's input tokens per
  turn: `(query + answer + 2,000) characters / 4` (query and answer capped at 2,000 characters each) at the
  input price of `granite-3b`.
- **A router's own estimate.** `CacheCost` and `StaticCost` use the same formula with the predicted output
  length. `CacheCost` charges each model's held prefix at the cached price; `StaticCost` charges every prompt
  token at the full price.

## Offline and online

The same `decide(turn, state)` runs in both settings. Only the source of the `Turn` and of the usage changes.

| | Offline replay (static track) | Online (gateway, local session, dynamic track) |
|---|---|---|
| prompt tokens | recorded per model | estimated at 4 characters per token |
| output tokens in the estimate | the shared length heads | the router's saved length heads (256 without them or without an encoder) |
| turn embeddings | precomputed (`embeddings/emb_chat_granite3b.npz`) | the live encoder (`encoder:` in the config) |
| answer of the turn | the original conversation's answer | the chosen model's reply |
| realized cost | recorded tokens, KV cache along the routed path | the backend's reported usage at its prices |
| quality | the judge's label of the chosen model's answer | the dynamic track's judge, after the run |

## Repository layout

| Path | Contents |
|---|---|
| `multiturnserve/__init__.py` | `open`, the public classes and the registries |
| `multiturnserve/session.py` | `Session` (route, run, observe, chat), `Runtime` (config loading), the turn log |
| `multiturnserve/router.py` | `Router` (`decide`, `pick`, `route`, `fit`, `save`) and `load_router` |
| `multiturnserve/types.py` | `Turn`, `SessionState`, `Exchange`, `Decision`, `Outcome`, `Reply`, `Usage`, `Costs` |
| `multiturnserve/pricing.py` | `turn_cost`, `static_cost`, `CacheState`, the static track's price table |
| `multiturnserve/registry.py` | `register_router`, `register_serving`, plugin loading |
| `multiturnserve/gateway.py` | the HTTP gateway (FastAPI) |
| `multiturnserve/client.py` | `RemoteSession`, a session on a running gateway |
| `multiturnserve/cli.py` | the `multiturnserve` command |
| `multiturnserve/parts/` | router parts: `CacheCost`, `StaticCost`, `SwitchCost`, `QualityHead`, `LengthHead`, `Encoder`, `HashEncoder` |
| `multiturnserve/routers/` | the six built-in routers, one package each, and the shared MLP router |
| `multiturnserve/serving/` | the `Serving` interface and the `vllm`, `openrouter` and `echo` backends |
| `multiturnserve/arena/` | data loading, training (`train.py`), the static replay, the dynamic track, the release, the leaderboard; `static.yaml`, the static track's fixed settings |
| `configs/` | `serve.yaml`, an example serve config; `arena/static.yaml`, a copy of the static track's settings |
| `examples/` | `my_router.py`, `my_backend.py`, `my_lab.py` (stand-ins for your code), `config.yaml`, `app.py` |
| `arena/README.md` | the arena's landing page |
| `LEADERBOARD.md` | static-track results of the built-in routers |
| `docs/` | the documentation: user guide, this page, routers, backends, FAQ; figures in `docs/assets/` |
| `tests/` | tests on synthetic data; `tests/reference/` holds the original evaluation code for the equivalence test |
