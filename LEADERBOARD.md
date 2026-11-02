# Leaderboard

Static track of MultiTurnServe Arena: 3,671 test turns, 9 models (`configs/arena/static.yaml`). Every router is trained on the train split and replayed on the test split over 41 values of the cost weight. Source: the evaluation package's reported run (reported/chat_s0/results.json).

- **AUDC**: area under the cost-quality curve (cost normalized by the most expensive model). Higher is better.
- **QNC**: the lowest normalized cost at which a router reaches the quality of the best single model, gemma-31b (0.7061); interpolated on the curve's upper envelope. Lower is better; - means never.
- **Cost vs gemma-31b**: QNC as a share of what gemma-31b alone costs (0.0462).
- **Peak**: the highest mean quality on the curve.

| Rank | Router | AUDC | QNC | Cost vs gemma-31b | Peak |
|---|---|---|---|---|---|
| 1 | **CoR (ours)** (`cor`) | 0.7389 | 0.0344 | 74% | 0.7449 |
| 2 | RAG history (`rag`) | 0.6998 | - | - | 0.7048 |
| 3 | Single-turn (`single-turn`) | 0.6963 | - | - | 0.7016 |
| 4 | Bao et al. (simplified) (`bao`) | 0.6909 | - | - | 0.6955 |
| 5 | Full history (`full-history`) | 0.6905 | - | - | 0.6952 |
| 6 | kNN (`knn`) | 0.6858 | - | - | 0.6905 |
| | gemma-31b alone (reference) | | 0.0462 | 100% | 0.7061 |

### Score at a cost weight

How much cost matters is a choice. For a weight w, each router is taken at its best point of the curve for quality - w x cost, with cost as a share of what gemma-31b alone costs.

| Router | w = 0 | w = 0.05 | w = 0.1 | w = 0.2 | w = 0.5 |
|---|---|---|---|---|---|
| CoR (ours) | 0.7449 | 0.6938 | 0.6427 | 0.5904 | 0.5302 |
| RAG history | 0.7048 | 0.6555 | 0.6112 | 0.5863 | 0.5296 |
| Bao et al. (simplified) | 0.6955 | 0.6523 | 0.6114 | 0.5863 | 0.5301 |
| Single-turn | 0.7016 | 0.6485 | 0.6050 | 0.5861 | 0.5296 |
| Full history | 0.6952 | 0.6478 | 0.6075 | 0.5865 | 0.5302 |
| kNN | 0.6905 | 0.6351 | 0.6066 | 0.5871 | 0.5291 |
| gemma-31b alone | 0.7061 | 0.6561 | 0.6061 | 0.5061 | 0.2061 |

One run (seed 0) of the six built-in routers with the default options; there is no variance across seeds yet. Bao et al. is a simplified reimplementation (`multiturnserve/routers/bao`), not the authors' code. Reproduce with `multiturnserve arena replay --seed 0` on the full data (the test labels are held by the organizers).
