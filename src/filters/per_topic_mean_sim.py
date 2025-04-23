from typing import Literal
from framework3.container.container import Container
from framework3.base.base_clases import BaseFilter
from framework3.base.base_types import XYData

import torch
from torchmetrics.functional.pairwise import (
    pairwise_cosine_similarity,
    pairwise_euclidean_distance,
    pairwise_manhattan_distance,
    pairwise_minkowski_distance,
    pairwise_linear_similarity,
)

Fsim = {
    "COSINE": lambda x: pairwise_cosine_similarity(
        x, reduction="mean", zero_diagonal=True
    ),
    "LINEAR": lambda x: pairwise_linear_similarity(
        x, reduction="mean", zero_diagonal=True
    ),
    "EUCLIDEAN": lambda x: 1
    / (pairwise_euclidean_distance(x, reduction="mean", zero_diagonal=True) + 1e-8),
    "MANHATTAN": lambda x: 1
    / (pairwise_manhattan_distance(x, reduction="mean", zero_diagonal=True) + 1e-8),
    "MINKOWSKI": lambda x: 1
    / (pairwise_minkowski_distance(x, reduction="mean", zero_diagonal=True) + 1e-8),
}


@Container.bind()
class PerTopicMeanSimilarity(BaseFilter):
    def __init__(
        self,
        sim_f_name: Literal[
            "COSINE", "EUCLIDEAN", "MANHATTAN", "MINKOWSKI", "LINEAR", "SPEARMAN_CORR"
        ] = "COSINE",
    ):
        super().__init__(sim_f_name=sim_f_name)
        self._sim_f = Fsim[sim_f_name]

    def predict(self, x: XYData) -> XYData:
        topic_tensors = x.value
        topic_results = []
        for i in range(topic_tensors.size(0)):
            res = self._sim_f(topic_tensors[i])
            topic_results.append(res)

        value = torch.mean(torch.stack(topic_results), dim=1)

        return XYData.mock(value)
