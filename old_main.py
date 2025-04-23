from src.experiments.configs import experiment_setup
from src.filters.transformer_embeddings import TransformersEmbedder, XYData
from src.filters.per_topic_mean_sim import PerTopicMeanSimilarity
from src.filters.gensim_embeddings import GensimEmbedder
from framework3 import Cached, F3Pipeline, MonoPipeline
from itertools import product
import pandas as pd
import torch
from rich import print


transformers_models = [
    "microsoft/mpnet-base",
    "bert-base-uncased",
    "roberta-base",
    "albert-base-v2",
]

gensim_models = [
    "word2vec-google-news-300",
    "glove-wiki-gigaword-300",
    "fasttext-wiki-news-subwords-300",
]


f_metrics = ["COSINE", "LINEAR"]

use_input_embeddingd = [True]

transformers_combinations = list(
    product(transformers_models, use_input_embeddingd, f_metrics)
)
gensim_combinations = list(product(gensim_models, f_metrics))

print(gensim_combinations)

all_dfs = pd.DataFrame()
seen_models = set()


def overwrite_first(model, apply=False):
    if model in seen_models or not apply:
        return False
    else:
        seen_models.add(model)
        return True


for name, config in experiment_setup.items():
    df = pd.read_csv(config["topics"], header=None, index_col=0)
    X = XYData(_hash=name, _path="datasets", _value=df.loc[:, 2:].values)  # type: ignore

    thePipeline2 = MonoPipeline(
        filters=[
            *[
                F3Pipeline(
                    filters=[
                        Cached(
                            filter=TransformersEmbedder(model, input_embs=input_embs),
                            overwrite=overwrite_first(model, apply=False),
                            cache_data=True,
                            cache_filter=False,
                        ),
                        PerTopicMeanSimilarity(metric),  # type: ignore
                    ]
                )
                for model, input_embs, metric in transformers_combinations
            ],
            *[
                F3Pipeline(
                    filters=[
                        Cached(
                            filter=GensimEmbedder(model),
                            overwrite=overwrite_first(model, apply=False),
                            cache_data=True,
                            cache_filter=False,
                        ),
                        PerTopicMeanSimilarity(metric),  # type: ignore
                    ]
                )
                for model, metric in gensim_combinations
            ],
        ],
    )

    thePipeline2.init()
    preduction2 = thePipeline2.predict(X)

    coh_results = pd.read_csv(config["results"], index_col=0)

    groun_truth = XYData.mock(torch.tensor(coh_results["humans"].values))

    df = pd.DataFrame(
        preduction2.value,
        columns=list(
            map(
                lambda x: f"{x[0]}_{x[1]}{'' if len(x) == 2 else '_' + x[2]}",
                transformers_combinations + gensim_combinations,
            )
        ),
    )

    df_combined = pd.concat([coh_results.iloc[:, 1:], df], axis=1)

    all_dfs[name] = df_combined.corr(method="spearman")["humans"]

    all_dfs = all_dfs.drop(index="humans")

all_dfs["avg"] = all_dfs.mean(axis=1)

print(all_dfs)

all_dfs.to_csv("data/all_results.csv")
