# Routers

A router picks the model that answers each turn of a session. This page lists the built-in routers, describes
the router interface, and shows how to add your own router, from a file that serves at once to a router that
trains on the data.

- [Built-in routers](#built-in-routers)
- [The router interface](#the-router-interface)
- [Parts](#parts)
- [Tutorial: adding your own router](#tutorial-adding-your-own-router)

## Built-in routers

| Router | Quality from | Cost in the decision |
|---|---|---|
| `cor` | Dialogue State Predictor: local attention over past turns, a GRU state and a gate | each model's cached prefix at the cached price (`CacheCost`) |
| `single-turn` | MLP on the current query | every prompt token at the full input price (`StaticCost`) |
| `full-history` | MLP on one embedding of the history and the query | full input price (`StaticCost`) |
| `rag` | MLP on the query and the mean of the 2 past turns most similar to it | full input price (`StaticCost`) |
| `knn` | similarity-weighted labels of the 20 most similar training queries | full input price (`StaticCost`) |
| `bao` | MLP on one embedding of the history and the query (Bao et al., simplified) | full input price plus a penalty for switching models (`SwitchCost`) |

- All six train on the data and read turn embeddings. Train one with `multiturnserve train` before serving it
  (see the [user guide](user-guide.md#training-the-built-in-routers)); serve it with an encoder
  (`encoder: granite`, the default).
- `cor` is Chain-of-Route. Its Dialogue State Predictor is in `multiturnserve/routers/cor/dsp.py`; `--dsp 2`
  selects a variant that projects every input and also reads the history embedding.
- `single-turn`, `full-history` and `rag` share one MLP router (`multiturnserve/routers/mlp.py`) and differ in
  the features they read.
- `bao` is a simplified reimplementation of the idea in Bao et al., not the authors' algorithm or code: one
  frozen embedding of history and query, a two-layer MLP, and a fixed switch penalty (`--switch-penalty`, 0.1
  by default) in normalized-cost units.
- In `results.json` the routers appear under their labels, such as `CoR (DSP + SST)` or
  `kNN query + static`; the leaderboard maps them back to the router names.

## The router interface

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

Register a router with `@register_router`. The default `decide` is `pick(quality(turn, state.history),
cost(turn, state.kv_cache))`. In `pick`, ties go to the model listed first; `lam = 0` picks on quality alone.

Class attributes that change how the router is used:

| Attribute | Default | Effect |
|---|---|---|
| `trained` | `False` | `True`: `serve` loads a checkpoint (`router.pt`) and fails with the `train` command to run if there is none |
| `uses_embeddings` | `False` | `True`: `serve` loads the `granite` encoder unless the config sets `encoder:` |
| `encoder_overhead` | `True` | `False`: the static replay does not charge the encoder's input tokens to this router |
| `models` | `None` | the models the router was trained on; set by `fit` or by the checkpoint |

`turn` (`multiturnserve.Turn`):

| Field | What |
|---|---|
| `chat_id`, `t`, `query`, `messages` | the session, the turn index from 0, the user message, the whole request (online) |
| `models` | the candidate models, in pool order |
| `n_prompt[model]` | prompt tokens the model would read: recorded offline; online, 4 characters per token |
| `out_tokens[k]` | predicted output tokens of `models[k]` (the shared length heads; 256 without them) |
| `prices[model]` | USD per 1M tokens: `input`, `cached`, `output` |
| `conv`, `emb` | the conversation so far and its turn embeddings `turn_id -> {q, a, h}` (online: with an encoder) |

`state` (`SessionState`):
- `history`, a tuple of `Exchange(query, answer, model)`;
- `kv_cache`, a `CacheState` with `held[model]` tokens and `cached(model, n_prompt)`, the cached part of a
  prompt of `n_prompt` tokens.

## Parts

`multiturnserve.parts` holds the pieces the built-in routers are made of:

| Part | What it returns |
|---|---|
| `CacheCost()` | each model's estimated cost with its cached prefix at the cached price |
| `StaticCost()` | each model's estimated cost with every prompt token at the full price |
| `SwitchCost(penalty)` | full price plus a penalty for leaving the model that served the previous turn |
| `QualityHead(predict, models)` | a trained predictor's quality for every model |
| `LengthHead(W, models)` | predicted output tokens of every model |
| `Encoder()`, `HashEncoder(dim)` | live turn embeddings (granite-4.2-3b), or stand-ins from a hash for tests |

A cost part returns `Costs(values, scale, models, usd)`: `pick` divides `values` by `scale`, the most expensive
model's full-price estimate, and `usd` is the estimate in USD. [Cost accounting](architecture.md#cost-accounting)
gives the formula.

## Tutorial: adding your own router

### 1. Write the router

A router is one file. This one is `examples/my_router.py`:

```python
from multiturnserve import register_router, Router
from multiturnserve.parts import CacheCost
from my_lab import MyQuality      # your code

@register_router
class MyRouter(Router):
    name = "my-router"
    quality = MyQuality()      # reads the history
    cost = CacheCost()         # reads the KV cache

    def decide(self, turn, state):
        q = self.quality(turn, state.history)
        c = self.cost(turn, state.kv_cache)
        return self.pick(q, c)

# after the reply, MultiTurnServe updates both states
```

`MyQuality` is your quality estimate: called with `(turn, history)`, it returns one number per model of
`turn.models`, in that order. `examples/my_lab.py` gives a toy one, so the example runs anywhere. `decide` may
also return a model name by any rule of its own; it must be a model of the pool.

The name uses dashes and may not be the name of a built-in router.

### 2. Serve it

Put the file in `routers/` next to the config and name it in the config, or list it under `plugins:`:

```yaml
router: my-router
plugins: [my_router.py]      # not needed when the file is in routers/
```

`examples/config.yaml` serves `my-router` with the `echo` backend and `examples/my_backend.py`, with no GPU and
no keys:

```bash
multiturnserve serve examples/config.yaml --check     # loads the router and the backends, then exits
multiturnserve serve examples/config.yaml
python examples/app.py
```

### 3. Compare it in the static track

The same `decide` runs in the arena's replay:

```bash
multiturnserve arena replay --router examples/my_router.py cor --dev
```

`--dev` replays on a fifth of the training conversations, since the test labels are hidden. The router reads
the recorded prompt lengths and the precomputed embeddings, and `lam` sweeps 41 values.

### 4. Train it

A router that learns from the data implements `fit(ctx)`, and `state` / `load_state` so that a checkpoint can
hold it. `trained = True` makes `serve` load that checkpoint. This router predicts each model's mean quality on
the training turns:

```python
import numpy as np
import torch

from multiturnserve import register_router, Router
from multiturnserve.parts import CacheCost


@register_router
class MeanQuality(Router):
    name = "mean-quality"
    trained = True             # serve loads runs/mean-quality/router.pt
    cost = CacheCost()

    def fit(self, ctx):
        # every model's mean quality over the training turns
        q = [[t.quality[m] for m in ctx.models] for c in ctx.train_all for t in c.turns]
        self.mean = np.mean(q, axis=0)

    def decide(self, turn, state):
        q = np.array([self.mean[self.models.index(m)] for m in turn.models])
        c = self.cost(turn, state.kv_cache)
        return self.pick(q, c)

    def state(self):
        return {"mean": torch.tensor(self.mean)}

    def load_state(self, state):
        self.mean = state["mean"].numpy()
```

```bash
multiturnserve train --router routers/mean_quality.py --out runs/mean-quality
multiturnserve serve config.yaml                       # with router: mean-quality
```

`state()` returns tensors and plain values; the checkpoint is read with `torch.load(..., weights_only=True)`.
`fit` receives a `Context`:

| Field | What |
|---|---|
| `train_all` | all training conversations |
| `train` | the training conversations without the early-stopping hold-out |
| `val` | the early-stopping hold-out, 10% of the training conversations |
| `test` | the conversations to replay (empty in `multiturnserve train`) |
| `emb` | turn embeddings, `turn_id -> {"q", "a", "h"}` |
| `models` | the model pool, in order |
| `out_hat`, `length` | the shared output-length predictions and their weights |
| `args` | the run's options (`epochs`, `seed`, `loss`, `rank_w`, router options) |
| `seeds` | the seeds to average over (`--n-models`) |

Each conversation has `conv_id` and `turns`. Each turn has `turn_id`, `t`, `query`, `orig_answer`,
`context_dependent`, and per model `quality`, `n_prompt` and `n_output`.

### 5. Submit it

```bash
multiturnserve arena submit routers/my_router.py --files routers/my_lab.py
```

This packs the router, the files it imports and a `meta.json` into `submission.tar.gz` for the organizers'
replay. [Track rules](user-guide.md#track-rules) gives the rules.
