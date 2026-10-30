"""Simplified Bao et al. multi-turn router: its own history predictor and a switch penalty in the decision."""
from methods.bao.bao import BaoRouter

NAME = "bao"


def build(ctx) -> tuple:
    router = BaoRouter(ctx.models, ctx.emb, switch_penalty=getattr(ctx.args, "switch_penalty", 0.1),
                       loss=getattr(ctx.args, "loss", "mse"), epochs=ctx.args.epochs, seed=ctx.args.seed)
    router.fit(ctx.train, ctx.val).policy(ctx.test, ctx.out_hat, ctx.overhead)
    return [router], {"bao": router.predict}
