from labchain import F3Pipeline, Cached
from labchain.plugins.optimizer.grid_optimizer import GridOptimizer

from src.filters.contextual_embeddings import ContextualTransformersEmbedder
from src.filters.per_topic_mean_sim import PerTopicMeanSimilarity
from src.metrics.corr_spearman import CorrSpearman
from src.metrics.stat_metric import BootstrapSpearmanCI, CorrPearsonWithPValue, CorrSpearmanWithPValue


def get_pipeline():
    return F3Pipeline(
        filters=[
            Cached(
                ContextualTransformersEmbedder(
                    "bert-base-uncased", 
                    context_strategy="template"
                ).grid({
                    "model_path": [
                        "bert-base-uncased",
                        "roberta-base",
                        "microsoft/mpnet-base",
                    ],
                    "context_strategy": [
                        "template",
                        "pairwise", 
                        "layer_pooling"
                    ]
                }),
                cache_filter=False
            ),
            PerTopicMeanSimilarity("COSINE").grid({
                "sim_f_name": ["COSINE", "LINEAR"]
            }),
        ],
        metrics=[CorrSpearman(),CorrSpearmanWithPValue(), BootstrapSpearmanCI()],
    ).optimizer(GridOptimizer(scorer=CorrSpearman()))