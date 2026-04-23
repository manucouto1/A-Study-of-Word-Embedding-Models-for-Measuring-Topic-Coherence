from labchain import F3Pipeline, Cached
from labchain.plugins.optimizer import GridOptimizer

from src.filters.transformer_embeddings import TransformersEmbedder
from src.filters.per_topic_mean_sim import PerTopicMeanSimilarity
from src.metrics.corr_spearman import CorrSpearman
from src.metrics.stat_metric import BootstrapSpearmanCI, CorrSpearmanWithPValue


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
                ),
                cache_filter=False,
            ),
            PerTopicMeanSimilarity("COSINE").grid({"sim_f_name": ["COSINE", "LINEAR"]}),
        ],
        metrics=[CorrSpearman(),CorrSpearmanWithPValue(), BootstrapSpearmanCI()],
    ).optimizer(GridOptimizer(scorer=CorrSpearman()))
