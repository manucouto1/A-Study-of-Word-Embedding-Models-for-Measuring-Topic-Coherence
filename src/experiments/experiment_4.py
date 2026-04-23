# src/experiments/experiment_5_tbuckets.py

from labchain import F3Pipeline, Cached
from labchain.plugins.optimizer.grid_optimizer import GridOptimizer

from src.filters.gensim_embeddings import GensimEmbedder
from src.filters.t_buckets import TBuckets
from src.metrics.corr_spearman import CorrSpearman
from src.metrics.stat_metric import BootstrapSpearmanCI, CorrPearsonWithPValue, CorrSpearmanWithPValue


def get_pipeline():
    """
    Experimento 5: TBuckets (Ramrakhiyani et al., 2017)
    
    Algoritmo:
    1. Obtener embeddings de palabras del tópico
    2. SVD para encontrar k direcciones principales (buckets)
    3. Integer Linear Programming para asignar palabras a buckets óptimamente
    4. Score = tamaño del bucket más grande / total palabras
    
    Embeddings evaluados:
    - word2vec-google-news-300 (usado en paper original)
    - glove-wiki-gigaword-300
    - fasttext-wiki-news-subwords-300
    
    Hiperparámetros:
    - k_buckets: número de buckets [2, 3, 4, 5]
    """
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
                cache_filter=False
            ),
            TBuckets().grid({
                "fixer":["nothing"],
            })
        ],
        metrics=[CorrSpearman(),CorrSpearmanWithPValue(), BootstrapSpearmanCI()],
    ).optimizer(GridOptimizer(scorer=CorrSpearman()))