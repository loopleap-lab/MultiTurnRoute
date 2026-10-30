"""What every method's build() receives: the split, embeddings, the model pool, predicted output lengths, the
per-turn encoder overhead and the run arguments."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Context:
    train_all: list    # all training conversations, original order
    train: list        # training conversations (early-stopping hold-out removed)
    val: list          # early-stopping hold-out
    test: list         # evaluation conversations
    emb: dict          # turn_id -> {"q", "a", "h"}
    models: list       # model pool (names, fixed order)
    out_hat: dict      # conv_id -> [T, K] predicted output tokens (shared length heads)
    overhead: dict     # conv_id -> [T] USD per turn of running the frozen encoder
    args: object       # run arguments (epochs, seed, n_models, loss, rank_w, method-specific options)

    @property
    def seeds(self) -> list:
        """Every trained predictor is averaged over n_models seeds."""
        return [self.args.seed + 100 * m for m in range(getattr(self.args, "n_models", 1))]
