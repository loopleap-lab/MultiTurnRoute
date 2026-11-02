"""Simplified Bao et al. multi-turn router: its own history predictor and a switch penalty in the decision."""
from ...parts import QualityHead, SwitchCost
from ...registry import register_router
from ...router import Router
from .bao import BaoModel


@register_router
class Bao(Router):
    name = "bao"
    label = "full history + switch penalty (Bao-style)"
    trained = True
    uses_embeddings = True

    def __init__(self, switch_penalty: float = 0.1):
        self.quality, self.cost = QualityHead(), SwitchCost(switch_penalty)

    def decide(self, turn, state):
        """argmax of (1 - lam) * q_hat - lam * (static cost estimate / most expensive + switch penalty)."""
        prev = state.history[-1].model if state.history else None
        q = self.quality(turn, state.history)
        c = self.cost(turn, state.kv_cache, prev=prev)
        return self.pick(q, c)

    def fit(self, ctx) -> None:
        self.cost = SwitchCost(getattr(ctx.args, "switch_penalty", 0.1))
        model = BaoModel(ctx.models, loss=getattr(ctx.args, "loss", "mse"), epochs=ctx.args.epochs,
                         seed=ctx.args.seed).fit(ctx.train, ctx.val, ctx.emb)
        self.models = list(ctx.models)
        self.quality = QualityHead(model.predict, ctx.models)

    def predictors(self, emb) -> dict:
        return {"bao": lambda c: self.quality.predict(c, emb)}

    def state(self) -> dict:
        return {"net": self.quality.predict.__self__.state(), "switch_penalty": self.cost.penalty}

    def load_state(self, state: dict) -> None:
        model = BaoModel(self.models).load_state(state["net"])
        self.cost = SwitchCost(state["switch_penalty"])
        self.quality = QualityHead(model.predict, self.models)
