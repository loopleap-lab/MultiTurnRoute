"""Single-turn routing: an MLP predicts every model's quality from the current query embedding only;
decision by quality-cost utility with the cost at full input price."""
from ...arena.embeddings import conv_arrays
from ...registry import register_router
from ..mlp import MLPRouter


@register_router
class SingleTurn(MLPRouter):
    name = "single-turn"
    label = "single-turn (query-only + static)"
    predictor_name = "query_only"

    @staticmethod
    def features(c, emb):
        return conv_arrays(c, emb)[0]
