from src.filters.gensim_embeddings import GensimEmbedder

from labchain import F3Pipeline, Cached
from labchain.plugins.optimizer.grid_optimizer import GridOptimizer

from src.filters.per_topic_mean_sim import PerTopicFixedCosineMeanSim, PerTopicMeanSimilarity
from src.metrics.corr_spearman import CorrSpearman
from src.metrics.stat_metric import BootstrapSpearmanCI, CorrPearsonWithPValue, CorrSpearmanWithPValue


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
                ),
                cache_filter=False,
                overwrite=True
            ),
            
            PerTopicFixedCosineMeanSim("COSINE").grid({"sim_f_name": ["COSINE", "LINEAR"]}),
        ],
        metrics=[CorrSpearman(),CorrSpearmanWithPValue(), BootstrapSpearmanCI()],
    ).optimizer(GridOptimizer(scorer=CorrSpearman()))
