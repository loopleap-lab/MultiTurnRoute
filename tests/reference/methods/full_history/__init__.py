"""Full-history routing: an MLP predicts every model's quality from one embedding of the history + query text;
decision by quality-cost utility with the cost at full input price."""
from common.embeddings import conv_arrays
from common.nn import averaged, train_mlp
from common.simulate import UtilityPolicy

NAME = "full_history"


def build(ctx) -> tuple:
    inputs = lambda c: conv_arrays(c, ctx.emb)[2]  # noqa: E731
    pred = averaged([train_mlp(ctx.train, ctx.val, inputs, ctx.models, ctx.args.epochs, sd, ctx.args.rank_w)
                     for sd in ctx.seeds])
    q_hat = {c.conv_id: pred(c) for c in ctx.test}
    return [UtilityPolicy("full history + static", q_hat, ctx.out_hat, ctx.overhead)], {"full_history": pred}
