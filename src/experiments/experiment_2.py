from src.filters.gensim_embeddings import GensimEmbedder

from framework3 import F3Pipeline, Cached
from framework3.plugins.optimizer.grid_optimizer import GridOptimizer

from ..filters.per_topic_mean_sim import PerTopicMeanSimilarity
from ..metrics.corr_spearman import CorrSpearman


def get_pipeline():
    return F3Pipeline(
        filters=[
            Cached(
                GensimEmbedder("word2vec-google-news-300").grid(
                    {
                        "model_path": [
                            "word2vec-google-news-300",
                            "glove-wiki-gigaword-300",
                            "fasttext-wiki-news-subwords-300",
                        ],
                    }
                )
            ),
            PerTopicMeanSimilarity("COSINE").grid({"sim_f_name": ["COSINE", "LINEAR"]}),
        ],
        metrics=[CorrSpearman()],
    ).optimizer(GridOptimizer(scorer=CorrSpearman()))
