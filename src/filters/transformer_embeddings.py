import types
from transformers import AutoModel, AutoTokenizer  # type: ignore

from tqdm import tqdm

from framework3.base import BaseFilter
from framework3 import Container
from framework3.base.base_types import XYData

import torch


def is_lambda(v):
    return isinstance(v, types.LambdaType) and v.__name__ == "<lambda>"


@Container.bind()
class TransformersEmbedder(BaseFilter):
    def __init__(self, model_path: str, input_embs=False):
        super().__init__(model_path=model_path)
        self.model_path = (
            model_path  # Path to the pre-trained model (e.g., 'bert-base-uncased')
        )
        self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        self._model = AutoModel.from_pretrained(model_path)
        self.input_embs = input_embs
        if input_embs:
            self._model = self._model.get_input_embeddings()

    def predict(self, x: XYData) -> XYData:
        all_m = []
        for topic in tqdm(x.value):
            topic_m = []
            for word in topic:
                encoded_input = self._tokenizer.encode(str(word), return_tensors="pt")
                if self.input_embs:
                    out = self._model(encoded_input)
                    mean_embeddings = torch.mean(out[0].detach().cpu(), axis=0)  # type: ignore
                else:
                    out = self._model(input_ids=encoded_input)
                    mean_embeddings = torch.mean(
                        out.last_hidden_state[0].detach().cpu(), axis=0
                    )  # type: ignore
                topic_m.append(mean_embeddings)

            all_m.append(torch.stack(topic_m))
        all_stack = torch.stack(all_m)
        return XYData.mock(all_stack.squeeze(2))
