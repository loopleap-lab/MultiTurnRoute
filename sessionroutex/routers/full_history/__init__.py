"""Full-history routing: an MLP predicts every model's quality from one embedding of the history + query text;
decision by quality-cost utility with the cost at full input price."""
from ...arena.embeddings import conv_arrays
from ...registry import register_router
from ..mlp import MLPRouter


@register_router
class FullHistory(MLPRouter):
    name = "full-history"
    label = "full history + static"
    predictor_name = "full_history"

    @staticmethod
    def features(c, emb):
        return conv_arrays(c, emb)[2]
