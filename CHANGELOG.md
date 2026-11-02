# Changelog

## 0.1.0 (2026-11-02)

First release. The code went in on 2026-10-30, ported from the evaluation package of the Chain-of-Route paper.

- Sessions (`multiturnserve.Session`, `Runtime`): one per `chat_id`, with two states, the dialogue history and
  the KV cache each model holds. Each turn is route, run and observe; the state follows the usage the backend
  reports, and every answered turn writes one log line without message texts.
- The router interface, `decide(turn, state)` and `pick(q, c)`, with parts for quality (`QualityHead`), cost
  (`CacheCost`, `StaticCost`, `SwitchCost`), output length (`LengthHead`) and live embeddings (`Encoder`).
- Six built-in routers: `cor`, `single-turn`, `full-history`, `rag`, `knn` and `bao` (a simplified Bao et al.).
  `multiturnserve train` trains one and writes the checkpoint `serve` loads.
- The serving interface, `run(model, messages)` and `prices(model)`, with the `vllm`, `openrouter` and `echo`
  backends. The API backends read the cached tokens from `usage.prompt_tokens_details.cached_tokens`; keys come
  from environment variables.
- The gateway (`multiturnserve serve`): an OpenAI-compatible `/v1/chat/completions`, plus `/v1/route`,
  `/v1/observe`, `/v1/sessions/{chat_id}`, `/v1/models` and `/health`. The Python client,
  `multiturnserve.open(chat_id, gateway=":8080")`, and local sessions from a config.
- Routers and backends from plugin files: `routers/` and `backends/` next to the config, or `plugins:`.
- MultiTurnServe Arena. Static track: `arena replay` over 41 cost weights with AUDC, QNC, QNC_hull, peak quality
  and a paired bootstrap; `arena release` for the public data with the test labels held apart; `arena submit` for
  a router file; `arena leaderboard`. Dynamic track: `arena run` with a user simulator, `arena judge`,
  `arena score`, `arena submit --log`.
- `LEADERBOARD.md`: the static-track results of the six built-in routers from the evaluation package's run.
- Examples in `examples/`: a router, a backend, a config and an app that run with the `echo` backend.
- Tests on synthetic data, including an equivalence test against the original evaluation code
  (`tests/reference/`).
- Documentation: `docs/user-guide.md`, `docs/architecture.md`, `docs/routers.md`, `docs/backends.md`,
  `docs/FAQ.md`, `ROADMAP.md` and this changelog.

Renamed from SessionRouteX to MultiTurnServe on 2026-11-02: the Python package and the command
`sessionroutex` became `multiturnserve`, the environment variables `SRX_*` became `MTS_*`, and `docs/static/`
became `docs/assets/`.
