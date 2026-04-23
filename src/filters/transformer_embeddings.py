from transformers import AutoModel, AutoTokenizer # type: ignore
from tqdm import tqdm
from labchain.base import BaseFilter
from labchain import Container
from labchain.base.base_types import XYData
import torch
import random
import numpy as np


@Container.bind()
class TransformersEmbedder(BaseFilter):
    def __init__(self, model_path: str, input_embs=True, layer_num=None, seed=42):
        """
        Args:
            model_path: Path al modelo de Hugging Face
            input_embs: Si True, usa input embeddings. Si False, usa hidden states
            layer_num: Número de capa a extraer (0-indexed). 
                      None = última capa
                      -1 = input embeddings
                      0, 1, 2, ... = capas específicas
            seed: Semilla aleatoria
        """
        super().__init__(model_path=model_path)
        
        self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        self._model = AutoModel.from_pretrained(model_path, output_hidden_states=True)
        self._model.eval()
        
        self.model_path = model_path 
        self.seed = seed
        self.input_embs = input_embs
        self.layer_num = layer_num
        
        # Determinar número de capas del modelo
        self._num_layers = getattr(self._model.config, "num_hidden_layers", 12)
        
        # Si usamos input embeddings, extraer la capa
        if input_embs:
            self._embedding_layer = self._model.get_input_embeddings()
            
    def set_seed(self, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def predict(self, x: XYData) -> XYData:
        self.set_seed(self.seed)
        
        with torch.no_grad():
            all_m = []
            for topic in tqdm(x.value):
                topic_m = []
                for word in topic:
                    encoded_input = self._tokenizer.encode(str(word), return_tensors="pt")
                    
                    if self.input_embs:
                        # Input embeddings (capa -1)
                        out = self._embedding_layer(encoded_input)
                        mean_embeddings = torch.mean(out[0][1:-1].detach().cpu(), axis=0) # type: ignore
                    else:
                        # Hidden states de una capa específica
                        out = self._model(input_ids=encoded_input)
                        
                        if self.layer_num is None:
                            # Última capa (igual que last_hidden_state)
                            hidden_state = out.last_hidden_state
                        else:
                            # Capa específica (layer_num + 1 porque index 0 son los embeddings)
                            hidden_state = out.hidden_states[self.layer_num + 1]
                        
                        mean_embeddings = torch.mean(
                            hidden_state[0][1:-1].detach().cpu(), axis=0
                        ) # type: ignore
                    
                    topic_m.append(mean_embeddings)

                all_m.append(torch.stack(topic_m))
            
            all_stack = torch.stack(all_m)
            return XYData.mock(all_stack.squeeze(2))
