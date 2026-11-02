# MultiTurnServe Arena

Compares LLM routers on real multi-turn conversations, by answer quality and cost, in two tracks. The code is in
[`multiturnserve/arena/`](../multiturnserve/arena); every command and option is in the
[user guide](../docs/user-guide.md#running-the-arena); the current results are in
[LEADERBOARD.md](../LEADERBOARD.md).

- **Static track**: a fixed pool of 9 models and recorded conversations. Every model has answered every turn on
  the conversation's real history and each answer is scored, so a router's picks are looked up and scored turn
  by turn on hidden test conversations. The training split is released in full.

  ```bash
  multiturnserve arena replay --router routers/my_router.py --dev   # on a held-out part of the training split
  multiturnserve arena submit routers/my_router.py                  # pack the router file for the organizers
  ```

- **Dynamic track**: the session runs live. A fixed simulated user reads each reply and writes the next message,
  and routers are compared on whole sessions within the same model pool. You run your own models with your own
  API keys.

  ```bash
  multiturnserve arena run    --config pool.yaml --tasks test.jsonl --out runs/dyn/mine --user-url URL
  multiturnserve arena judge  --log runs/dyn/mine --judge-url URL
  multiturnserve arena score  --log runs/dyn/mine
  multiturnserve arena submit --log runs/dyn/mine
  ```

Rules for both tracks are in the [user guide](../docs/user-guide.md#track-rules). To write a router for the
arena, see [Routers](../docs/routers.md#tutorial-adding-your-own-router).
