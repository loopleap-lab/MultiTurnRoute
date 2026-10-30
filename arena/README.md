# SessionRouteX Arena

Compares LLM routers on real multi-turn chats, by answer quality and cost, in two tracks.

- **Static track**: a fixed pool of models and recorded chats. Every model has answered every turn on the chat's real history, and each answer is scored, so a router's picks are scored turn by turn, offline, on hidden test chats. The training split is released.
- **Dynamic track**: the chat runs live. A fixed simulated user reads each reply and writes the next message; routers are compared on whole chats within the same model pool.

**Status:** data, scoring code and the leaderboard will be published here.
