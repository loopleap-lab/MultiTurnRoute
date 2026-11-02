"""Shared shape of the routers whose quality predictor is an MLP on per-turn features (single-turn, full-history,
rag): averaged over seeds, decision by quality-cost utility with the cost at full input price."""
from ..parts import QualityHead, StaticCost
from ..parts.nn import Averaged, MLPPredictor, train_mlp
from ..router import Router


class MLPRouter(Router):
    trained = True
    uses_embeddings = True
    predictor_name = "mlp"

    @staticmethod
    def features(c, emb):
        raise NotImplementedError

    def __init__(self):
        self.quality, self.cost = QualityHead(), StaticCost()

    def decide(self, turn, state):
        q = self.quality(turn, state.history)
        c = self.cost(turn, state.kv_cache)
        return self.pick(q, c)

    def fit(self, ctx) -> None:
        pred = Averaged([train_mlp(ctx.train, ctx.val, self.features, ctx.emb, ctx.models, ctx.args.epochs, sd,
                                   ctx.args.rank_w) for sd in ctx.seeds])
        self.models = list(ctx.models)
        self.quality = QualityHead(pred, ctx.models)

    def predictors(self, emb) -> dict:
        return {self.predictor_name: lambda c: self.quality.predict(c, emb)}

    def state(self) -> dict:
        return {"heads": [f.state() for f in self.quality.predict.fs]}

    def load_state(self, state: dict) -> None:
        self.quality = QualityHead(Averaged([MLPPredictor.from_state(s, self.features) for s in state["heads"]]),
                                   self.models)
