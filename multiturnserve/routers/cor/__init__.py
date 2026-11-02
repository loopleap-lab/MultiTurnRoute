"""Chain-of-Route (CoR): the Dialogue State Propagator (DSP, dsp.py) estimates every model's quality from the dialogue
state, and the decision charges each model's cached conversation prefix at the cached price (parts.CacheCost, the
Serving State Tracker)."""
from ...parts import CacheCost, QualityHead
from ...parts.nn import Averaged
from ...registry import register_router
from ...router import Router
from .dsp import DSPPredictor, train_dsp


@register_router
class CoR(Router):
    name = "cor"
    label = "CoR (DSP + SST)"
    trained = True
    uses_embeddings = True

    def __init__(self):
        self.quality, self.cost = QualityHead(), CacheCost()

    def decide(self, turn, state):
        q = self.quality(turn, state.history)
        c = self.cost(turn, state.kv_cache)
        return self.pick(q, c)

    def fit(self, ctx) -> None:
        a = ctx.args
        dsps = [train_dsp(ctx.train, ctx.val, ctx.emb, ctx.models, a.epochs, getattr(a, "window", 2), sd,
                          getattr(a, "rank_w", 0.0), getattr(a, "dsp", 1), getattr(a, "dsp_p", 128),
                          getattr(a, "dsp_drop", 0.5)) for sd in ctx.seeds]
        self.models = list(ctx.models)
        self.quality = QualityHead(Averaged(dsps), ctx.models)

    def predictors(self, emb) -> dict:
        return {"dsp": lambda c: self.quality.predict(c, emb)}

    def state(self) -> dict:
        return {"dsps": [f.state() for f in self.quality.predict.fs]}

    def load_state(self, state: dict) -> None:
        self.quality = QualityHead(Averaged([DSPPredictor.from_state(s) for s in state["dsps"]]), self.models)
