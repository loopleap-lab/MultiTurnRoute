# MultiTurnRoute FAQ

Short answers to the questions people ask when they start using MultiTurnRoute. Each answer links to the full
documentation.

- [The basics](#the-basics)
- [Installing](#installing)
- [Serving](#serving)
- [Routers](#routers)
- [Costs and cached tokens](#costs-and-cached-tokens)
- [The arena](#the-arena)
- [Troubleshooting](#troubleshooting)

## The basics

### What does MultiTurnRoute do?

It routes every turn of a multi-turn session to one model of a pool. For each session it keeps the dialogue
history and the KV cache state, the tokens each model already holds for the session. A router reads both and
picks the model; a serving backend runs it and reports its usage, which updates both states and gives the cost.
[How MultiTurnRoute works](architecture.md) has the details.

### Is it only for chat applications?

No. It serves any multi-turn LLM application whose requests carry the session's messages. Chat is one example.
The API keeps the OpenAI names (`/v1/chat/completions`, `chat_id`) so that existing clients work.

### Does my app need to change?

An OpenAI client works with the gateway once each request names its session: a `chat_id` field, an
`X-Chat-Id` header or the OpenAI `user` field. The reply's `model` field is the model the router chose. See
[The OpenAI-compatible API](user-guide.md#the-openai-compatible-api).

### Do I need a GPU or API keys?

Not to try it. `multiturnserve serve examples/config.yaml` runs an example router with the `echo` backend, and the
tests run on CPU with synthetic data. Serving real models needs a vLLM server or an OpenRouter key. The live
encoder of the built-in routers runs on CPU and is faster on a GPU.

## Installing

### How do I install it?

From source, with `pip install -e ".[serve]"`, plus `".[encoder]"` to serve the built-in routers. The package is
not on PyPI yet. See [Installation](user-guide.md#installation).

### Where do I get the data?

The routers train on the MultiTurnRoute data (`chat.jsonl`, the turn embeddings and `prices.json`); set
`MTS_DATA_DIR` to its folder. The data have no public download location yet, and no trained checkpoints are
published ([roadmap](../ROADMAP.md#6-data-and-checkpoints)). [The data](user-guide.md#the-data) describes the
format, and [Toy data](user-guide.md#toy-data) shows how to write a synthetic data folder to try `train` and
`arena replay`.

## Serving

### How does the gateway know which session a request belongs to?

From the `chat_id` field, the `X-Chat-Id` header or the `user` field, in that order. A request without any of
them gets HTTP 400.

### Does every request carry the whole session?

Yes. As with any OpenAI client, `messages` holds the session so far, and the last user message is the turn's
query. The gateway keeps the session's state (history, KV cache, total cost) by `chat_id`.

### What happens when the gateway restarts?

Sessions live in the gateway's memory, so every session starts afresh. There is no persistence and no expiry
yet ([roadmap](../ROADMAP.md#2-online-serving)).

### What if the model call fails?

The gateway answers HTTP 502 with the model and backend that failed. The session's state is unchanged, so the
request can be sent again.

### Can my app call the model itself and only ask which model to use?

Yes. `POST /v1/route` returns the model without changing the state. After your own call, `POST /v1/observe`
reports the reply and its usage, and the session's state is updated. In Python: `s.route(messages)`, your call,
then `s.observe(Outcome(model, text, usage))`. See [The Python session API](user-guide.md#the-python-session-api).

### What does the log hold?

One JSONL line per answered turn in `<log>/turns.jsonl`: the `chat_id`, the turn, the model, the prompt, cached
and output tokens, the cost, the router's predicted quality and cost, and the time. It holds no message texts.

### Where are API keys stored?

Nowhere. Each backend names an environment variable (`VLLM_API_KEY`, `OPENROUTER_API_KEY`, or `api_key_env`) and
reads it when a request is sent. See [API keys](backends.md#api-keys).

## Routers

### Which routers are built in?

`cor`, `single-turn`, `full-history`, `rag`, `knn` and `bao`. [Built-in routers](routers.md#built-in-routers)
says what each one reads. `multiturnserve list` prints them, with any plugin routers.

### Why does `serve` say the router needs training first?

The built-in routers learn from the data, so `serve` loads a checkpoint, `runs/<router>/router.pt` by default.
Run the command the error gives, for example `multiturnserve train --router cor --out runs/cor`, or point
`checkpoint:` in the config at a trained router.

### Can I serve a trained router with a different pool?

Only with models the router was trained on, in any order. A model outside them fails at the first turn.
`multiturnserve train --models ...` sets the pool for training.

### What does `lam` do?

It weighs cost against quality. `pick` takes the model with the highest `(1 - lam) * quality - lam * cost`, with
the cost normalized by the most expensive model. `lam = 0` picks on quality alone; the default is 0.5. Set it
with `lam:` in the config. The static replay sweeps 41 values and the dynamic track 5.

### Can a router ignore the parts and use its own rule?

Yes. `decide(turn, state)` returns any model name of the pool. The parts and `pick` are there to reuse. See
[the tutorial](routers.md#tutorial-adding-your-own-router).

## Costs and cached tokens

### Why is `cached_tokens` always 0?

The backend does not report cached tokens. Start vLLM with `--enable-prompt-tokens-details`. On OpenRouter,
reporting depends on the provider. See [Cached tokens](backends.md#cached-tokens).

### Why does the bill differ from the router's estimate?

The estimate is made before the turn, with prompt lengths at 4 characters per token, predicted output lengths,
and a KV cache that assumes no eviction. The bill uses the usage the backend reports. Both use the formula in
[Cost accounting](architecture.md#cost-accounting).

### Which prices are used?

A backend's `prices` option, else the static track's price table. A model in neither raises an error that says
what to add to the config. See [Prices](backends.md#prices).

## The arena

### Why does `arena replay` stop and ask for `--dev`?

The test labels are hidden in the public data. `--dev` replays on a fifth of the training conversations instead.
The organizers replay with the labels (`MTS_HIDDEN_DIR`).

### How do I submit?

Static track: `multiturnserve arena submit my_router.py [--files ...]`. Dynamic track: `arena run`, `judge` and
`score`, then `multiturnserve arena submit --log LOG`. Both write a tarball; nothing is uploaded. See
[Running the arena](user-guide.md#running-the-arena).

### Are dynamic-track results comparable across submitters?

Only between runs with the same tasks, pool and user simulator. Results are self-reported. See
[Track rules](user-guide.md#track-rules).

### Can a stopped dynamic run continue?

Yes. `arena run` and `arena judge` skip what their logs already hold. `run` writes one whole conversation at a
time, so a stop never leaves half of one.

## Troubleshooting

| Message | Cause |
|---|---|
| `unknown config key(s) [...]` | a key outside `router, serving, models, log, lam, checkpoint, encoder, backends, plugins` |
| `router '...' needs training first: no .../router.pt` | the router is trained and has no checkpoint; run the `multiturnserve train` command in the message |
| `unknown router '...'` | the name is not registered; check the file is in `routers/` next to the config or under `plugins:` |
| `... registers no router; add @register_router to its class` | the file has no registered `Router` |
| `router '...' is built in` | a plugin router reuses a built-in name; pick another |
| `router '...' picked '...', which no backend serves` | `decide` returned a model outside the pool |
| `model '...' is served by both ...` | one model is listed under two backends |
| `no prices for model '...'` | the model is neither in the backend's `prices` nor in the static track's table |
| `this router reads turn embeddings; serve it with an encoder` | the config sets `encoder: none` for a router that reads embeddings |
| `takes no option(s)` or `unexpected keyword argument` | an option under `backends:` that the backend does not take |
