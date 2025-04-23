from framework3 import F3Pipeline, Cached
from framework3.plugins.optimizer.grid_optimizer import GridOptimizer

from ..filters.transformer_embeddings import TransformersEmbedder
from ..filters.per_topic_mean_sim import PerTopicMeanSimilarity
from ..metrics.corr_spearman import CorrSpearman


def get_pipeline():
    return F3Pipeline(
        filters=[
            Cached(
                TransformersEmbedder("microsoft/mpnet-base", input_embs=True).grid(
                    {
                        "model_path": [
                            "microsoft/mpnet-base",
                            "bert-base-uncased",
                            "roberta-base",
                            "albert-base-v2",
                        ],
                    }
                )
            ),
            PerTopicMeanSimilarity("COSINE").grid({"sim_f_name": ["COSINE", "LINEAR"]}),
        ],
        metrics=[CorrSpearman()],
    ).optimizer(GridOptimizer(scorer=CorrSpearman()))
