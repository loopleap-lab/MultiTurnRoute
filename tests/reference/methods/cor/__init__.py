"""Chain-of-Route (CoR): the Dialogue State Predictor (dsp.py) estimates every model's quality from the dialogue
state, and the decision charges each model's cached conversation prefix at the cached price (SST, policy.py)."""
import numpy as np

from methods.cor.dsp import train_dsp
from methods.cor.policy import CacheAwarePolicy

NAME = "cor"


def build(ctx) -> tuple:
    a = ctx.args
    dsps = [train_dsp(ctx.train, ctx.val, ctx.emb, ctx.models, a.epochs, getattr(a, "window", 2), sd,
                      getattr(a, "rank_w", 0.0), getattr(a, "dsp", 1), getattr(a, "dsp_p", 128),
                      getattr(a, "dsp_drop", 0.5)) for sd in ctx.seeds]

    def dsp(c):
        return np.mean([f(c) for f in dsps], axis=0)
    q_hat = {c.conv_id: dsp(c) for c in ctx.test}
    return [CacheAwarePolicy("CoR (DSP + SST)", q_hat, ctx.out_hat, ctx.overhead)], {"dsp": dsp}
