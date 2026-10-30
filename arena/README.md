# SessionRouteX Arena

Compares LLM routers on real multi-turn chats, by answer quality and cost, in two tracks. The code is in
[`sessionroutex/arena/`](../sessionroutex/arena); every command and option is in
[docs/REFERENCE.md](../docs/REFERENCE.md#arena-replay); the current results are in [LEADERBOARD.md](../LEADERBOARD.md).

- **Static track**: a fixed pool of 9 models and recorded chats. Every model has answered every turn on the chat's
  real history and each answer is scored, so a router's picks are looked up and scored turn by turn on hidden test
  chats. The training split is released in full.

  ```bash
  sessionroutex arena replay --router routers/my_router.py --dev   # on a held-out part of the training split
  sessionroutex arena submit routers/my_router.py                  # pack the router file for the organizers
  ```

- **Dynamic track**: the chat runs live. A fixed simulated user reads each reply and writes the next message, and
  routers are compared on whole chats within the same model pool. You run your own models with your own API keys.

  ```bash
  sessionroutex arena run    --config pool.yaml --tasks test.jsonl --out runs/dyn/mine --user-url URL
  sessionroutex arena judge  --log runs/dyn/mine --judge-url URL
  sessionroutex arena score  --log runs/dyn/mine
  sessionroutex arena submit --log runs/dyn/mine
  ```

Rules for both tracks are in [docs/REFERENCE.md](../docs/REFERENCE.md#track-rules).
