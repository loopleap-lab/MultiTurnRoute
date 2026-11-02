# Serving backends

A serving backend runs one model call and reports its usage: the prompt tokens, how many of them were cached, and
the output tokens. With the backend's prices this gives each turn's cost, and the usage keeps the session's KV
cache state up to date. This page lists the built-in backends and their options, explains how cached tokens are
read, and shows how to add a backend.

- [Built-in backends](#built-in-backends)
- [Options](#options)
- [API keys](#api-keys)
- [Cached tokens](#cached-tokens)
- [Prices](#prices)
- [Tutorial: adding a backend](#tutorial-adding-a-backend)
- [Known limitations](#known-limitations)

## Built-in backends

| Backend | What it runs | Default `base_url` | Key from |
|---|---|---|---|
| `vllm` | a vLLM OpenAI-compatible server (`vllm serve ... --enable-prompt-tokens-details`) | `http://localhost:8000/v1` | `VLLM_API_KEY`, if the server wants one |
| `openrouter` | OpenRouter | `https://openrouter.ai/api/v1` | `OPENROUTER_API_KEY` |
| `echo` | repeats the last user message, with a prefix cache: for tests and dry runs | | |

A config lists the backends under `serving:` and each backend's models under `models:`:

```yaml
serving: [vllm, openrouter]
models:
  vllm: [gemma-e4b, gemma-26b-a4b]
  openrouter: [gemma-31b, mistral-medium-128b]
```

## Options

The options of each backend go under `backends:` in the config. `vllm` and `openrouter` are OpenAI-compatible
backends and take:

| Option | What | Default |
|---|---|---|
| `base_url` | the server's base URL | see the table above |
| `api_key_env` | the environment variable holding the key | see the table above |
| `model_ids` | our model name -> the server's model id | the model name |
| `prices` | USD per 1M tokens per model: `input`, `cached`, `output` | the static track's price table |
| `timeout` | seconds per request | 120 |
| `extra` | more fields for every request | none; `openrouter` sends `usage: {include: true}` |

```yaml
backends:
  vllm: {base_url: "http://localhost:8000/v1", model_ids: {gemma-e4b: my-org/gemma-e4b}, timeout: 120}
  openrouter: {prices: {gemma-31b: {input: 0.2, cached: 0.04, output: 0.4}}}
```

`echo` takes `prices` only. Any other option is an error.

## API keys

Keys are read from the environment variable named by `api_key_env` each time a request is sent, and sent as a
`Bearer` token. They are never written to a config, a log or a submission. Without the variable, the request is
sent without a key.

## Cached tokens

A backend's `run` returns `(text, usage)`. The usage can be:
- `Usage(prompt_tokens, cached_tokens, output_tokens)`;
- a tuple `(prompt, cached, output)`;
- a dict with `prompt_tokens`, `cached_tokens`, `output_tokens`;
- OpenAI's `usage` field, where the cached tokens are `prompt_tokens_details.cached_tokens` and the output tokens
  `completion_tokens`.

`vllm` and `openrouter` read the cached tokens from `usage.prompt_tokens_details.cached_tokens`, and count 0 when
the server does not report them:
- vLLM reports them only when started with `--enable-prompt-tokens-details`.
- On OpenRouter, the backend asks for usage in every request (`usage: {include: true}`); whether cached tokens are
  reported depends on the provider.

`echo` counts whitespace-separated words as tokens. A model's cache holds every prompt it read followed by its
reply, so the cached part of a new prompt is the longest prefix it shares with one of them. The cache belongs to
the backend, not to a session: every session of a gateway (one `Runtime`) shares it, so two sessions that start
with the same message get cached tokens on the second one, as on a serving engine with prefix caching.

## Prices

`prices(model)` returns USD per 1M tokens: `input`, `cached`, `output`. A backend uses its `prices` option first,
then the static track's price table (`multiturnserve/arena/static.yaml`). A model in neither is an error that
names the config entry to add. The static track's table:

| Model | Input | Cached | Output |
|---|---|---|---|
| `granite-3b` | 0.017 | 0.0034 | 0.112 |
| `gemma-e4b` | 0.02 | 0.004 | 0.04 |
| `gpt-oss-20b` | 0.018 | 0.009 | 0.09 |
| `devstral-24b` | 0.1 | 0.02 | 0.3 |
| `gemma-26b-a4b` | 0.0765 | 0.0425 | 0.255 |
| `granite-30b` | 0.16 | 0.04 | 0.65 |
| `gemma-31b` | 0.2 | 0.04 | 0.4 |
| `nemotron-30b-a3b` | 0.07 | 0.04 | 0.2 |
| `mistral-medium-128b` | 1.5 | 0.3 | 7.5 |

`static.yaml` gives the source of every price.

## Tutorial: adding a backend

### Any client

A backend is a class in a `.py` file. This one is `examples/my_backend.py`:

```python
from multiturnserve import register_serving, Serving
from my_lab import my_client, PRICES    # your code; examples/my_lab.py stands in for it

@register_serving
class MyBackend(Serving):
    name = "my-backend"
    models = ["gemma-26b-a4b", "gemma-31b"]

    def run(self, model, messages):
        reply = my_client.chat(model, messages)
        return reply.text, reply.usage    # usage: new and cached tokens

    def prices(self, model):
        return PRICES[model]              # $ per 1M tokens, cached and not
```

- `run(model, messages)` gets the session's messages in OpenAI format and returns the reply text and the usage
  (see [Cached tokens](#cached-tokens)).
- `prices(model)` returns USD per 1M tokens. Without it, the backend uses its `prices` option and then the
  static track's table.
- The config's `models:` entry for the backend replaces the class's `models`, and its `backends:` entry becomes
  constructor arguments. The base class takes `models` and `prices`; add your own options to `__init__`.

Put the file in a `backends/` folder in the config's folder, or list it under `plugins:` (paths relative to the
config), and name the backend under `serving:`. A file next to the config but not listed under `plugins:` is not
loaded. `examples/config.yaml` does this:

```yaml
router: my-router
serving: [echo, my-backend]
models:
  echo: [gemma-e4b]
  my-backend: [gemma-26b-a4b, gemma-31b]
lam: 0.2                  # cost weight: 0.2 weighs quality over cost, so the router moves to a larger model
plugins: [my_router.py, my_backend.py]
log: logs/
```

```bash
multiturnserve serve examples/config.yaml --check
```

### An OpenAI-compatible server

For a server with an OpenAI-compatible chat completions API, subclass `OpenAICompatible`. It sends the request,
reads the cached tokens, and takes every option in [Options](#options):

```python
from multiturnserve import register_serving
from multiturnserve.serving.openai_compat import OpenAICompatible


@register_serving
class MyServer(OpenAICompatible):
    name = "my-server"
    base_url = "http://localhost:9000/v1"
    api_key_env = "MY_SERVER_API_KEY"
```

## Known limitations

- Before a turn, the router sees prompt lengths estimated at 4 characters per token for every model; there is no
  per-model tokenizer. The bill uses the usage the backend reports.
- The KV cache state assumes a model keeps a session's prefix for the whole session, with no eviction or TTL.
  Eviction shows up in the bill, through the reported cached tokens, but not in the router's estimate.
- Model names map to provider ids only through `model_ids`; there is no built-in table of provider ids for the 9
  models.
- The tests run `vllm` and `openrouter` against an in-process fake server, not a live one.
- Sessions live in the gateway's memory: no persistence across restarts, no expiry, no streaming responses.

[ROADMAP.md](../ROADMAP.md) lists what comes next.
