from typing import Callable
from labchain import Container, XYData
from labchain.base import BaseFilter
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
        valid_mask = []

        for topic in tqdm(x.value):
            topic_m = []
            word_mask = []
            for word in topic:
                try:
                    topic_m.append(self._embeddings[str(word)])  # type: ignore
                    word_mask.append(True)
                except KeyError:
                    topic_m.append(np.zeros(self._embeddings.vector_size))  # type: ignore
                    word_mask.append(True)

            all_m.append(torch.tensor(np.array(topic_m)))
            valid_mask.append(torch.tensor(word_mask))

        all_stack = torch.stack(all_m)
        mask_stack = torch.stack(valid_mask)
        
        return XYData.mock({
            "embeddings":all_stack.squeeze(2), 
            "mask": mask_stack
        })
