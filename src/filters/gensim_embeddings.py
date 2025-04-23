from typing import Callable
from framework3 import Container, XYData
from framework3.base import BaseFilter
from tqdm import tqdm

import gensim.downloader as api
import numpy as np
import torch
import warnings

warnings.filterwarnings(
    "ignore",
    message="Metric `SpearmanCorrcoef` will save all targets and predictions in the buffer.",
)


@Container.bind()
class GensimEmbedder(BaseFilter):
    def __init__(self, model_path: str):
        super().__init__(model_path=model_path)  # Initialize the base class
        self._embeddings: Callable = lambda: api.load(model_path)

    def predict(self, x: XYData) -> XYData:
        if callable(self._embeddings):
            self._embeddings = (
                self._embeddings()
            )  # Load the embeddings if they haven't been loaded yet
        all_m = []
        for topic in tqdm(x.value):
            topic_m = []
            for word in topic:
                try:
                    topic_m.append(self._embeddings[str(word)])  # type: ignore
                except KeyError:
                    topic_m.append([0] * self._embeddings.vector_size)  # type: ignore
            all_m.append(torch.tensor(np.array(topic_m)))

        all_stack = torch.stack(all_m)
        return XYData.mock(all_stack.squeeze(2))
