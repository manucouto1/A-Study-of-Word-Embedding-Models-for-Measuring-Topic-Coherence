from typing import Literal
from labchain.container.container import Container
from labchain.base.base_clases import BaseFilter
from labchain.base.base_types import XYData

import torch
from torchmetrics.functional.pairwise import (
    pairwise_cosine_similarity,
    pairwise_euclidean_distance,
    pairwise_manhattan_distance,
    pairwise_minkowski_distance,
    pairwise_linear_similarity,
)

import torch
import torch.nn.functional as F


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


@Container.bind()
class PerTopicFixedCosineMeanSim(BaseFilter):
    def __init__(
        self,
        sim_f_name: Literal[
            "COSINE", "LINEAR"
        ] = "COSINE",
    ):
        super().__init__(sim_f_name=sim_f_name)
        
    def predict(self, x: XYData) -> XYData:
        E = x.value["embeddings"]
        M = x.value["mask"]
        
        if self.sim_f_name == "COSINE":
            E = F.normalize(E, dim=-1, eps=1e-8)
            sim = torch.matmul(E, E.transpose(1, 2))
        elif self.sim_f_name == "LINEAR":
            sim = torch.matmul(E, E.transpose(1, 2))

        
        pair_mask = M.unsqueeze(2) & M.unsqueeze(1)

        diag = torch.eye(M.shape[1], device=M.device).bool()
        pair_mask &= ~diag

        masked_cos = torch.where(pair_mask, sim, torch.zeros_like(sim))

        sum_sim = masked_cos.sum(dim=(1, 2))   # (T,)
        count = pair_mask.sum(dim=(1, 2))      # (T,)

        coherence = torch.zeros_like(sum_sim)

        valid = count > 0
        coherence[valid] = sum_sim[valid] / count[valid]


        return XYData.mock(coherence)