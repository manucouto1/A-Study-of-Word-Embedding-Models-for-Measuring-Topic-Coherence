# src/filters/contextual_transformer_embeddings.py

from labchain import Container, XYData
from labchain.base.base_clases import BaseFilter
from transformers import AutoModel, AutoTokenizer # type: ignore
import torch
from typing import Literal
from tqdm import tqdm
import gc
import warnings
from typeguard import InstrumentationWarning

warnings.filterwarnings(
    "ignore",
    category=InstrumentationWarning
)


@Container.bind()
class ContextualTransformersEmbedder(BaseFilter):
    def __init__(
        self,
        model_path: str,
        context_strategy: Literal["template", "pairwise", "layer_pooling"] = "template",
        layer_range: list = [9, 12],
    ):
        super().__init__(model_path=model_path)
        self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        self._model = AutoModel.from_pretrained(model_path)
        self._model.eval()
        
        for param in self._model.parameters():
            param.requires_grad = False
        
        self.context_strategy = context_strategy
        self.layer_range = layer_range

    def _create_template_context(self, word: str) -> str:
        return f"The topic discusses {word}."

    def _create_pairwise_context(self, word1: str, word2: str) -> str:
        """Contexto que enfatiza la relación entre dos palabras"""
        return f"{word1} and {word2} are semantically related."

    @torch.no_grad()
    def _get_embedding_batch(self, texts: list) -> torch.Tensor:
        """Procesa batch de textos SIN memory leak"""
        inputs = self._tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=64
        )

        if self.context_strategy == "layer_pooling":
            outputs = self._model(**inputs, output_hidden_states=True, use_cache=False)
            layers = [
                outputs.hidden_states[i].detach().clone() 
                for i in range(self.layer_range[0], self.layer_range[1] + 1)
            ]
            del outputs
            
            pooled = torch.stack(layers).mean(dim=0)
            result = pooled[:, 0, :].clone()
            del layers, pooled
            
        else:
            outputs = self._model(**inputs, use_cache=False)
            result = outputs.last_hidden_state[:, 0, :].detach().clone()
            del outputs

        del inputs
        gc.collect()
        
        return result

    @torch.no_grad()
    def predict(self, x: XYData) -> XYData:
        n_topics = len(x.value)
        
        if n_topics == 0:
            return XYData.mock(torch.empty(0))
        
        sample_topic = x.value[0]
        n_words = len(sample_topic)
        
        # Obtener dimensión de embedding
        test_emb = self._get_embedding_batch([self._create_template_context("test")])
        emb_dim = test_emb.shape[-1]
        del test_emb
        gc.collect()
        
        # Pre-alocar tensor final
        all_embeddings = torch.zeros(n_topics, n_words, emb_dim, dtype=torch.float32)

        batch_size = 16
        
        for topic_idx, topic in enumerate(tqdm(x.value, desc=f"{self.context_strategy}")):
            
            if self.context_strategy == "pairwise":
                # CORRECCIÓN: Cada palabra se empareja con la siguiente (circularmente)
                contexts = []
                for i in range(len(topic)):
                    word1 = str(topic[i])
                    word2 = str(topic[(i + 1) % len(topic)])  # Circular: última con primera
                    context = self._create_pairwise_context(word1, word2)
                    contexts.append(context)
                
                # Procesar en batches
                word_embs = []
                for batch_start in range(0, len(contexts), batch_size):
                    batch_contexts = contexts[batch_start:batch_start + batch_size]
                    batch_emb = self._get_embedding_batch(batch_contexts)
                    word_embs.append(batch_emb.clone())
                    del batch_emb, batch_contexts
                
                # Cada palabra tiene su embedding único
                topic_emb = torch.cat(word_embs, dim=0)
                all_embeddings[topic_idx] = topic_emb.clone()
                
                del word_embs, topic_emb, contexts
                
            else:  # template, layer_pooling
                word_embs = []
                
                for batch_start in range(0, len(topic), batch_size):
                    batch_words = topic[batch_start:batch_start + batch_size]
                    contexts = [self._create_template_context(str(w)) for w in batch_words]
                    
                    batch_emb = self._get_embedding_batch(contexts)
                    word_embs.append(batch_emb.clone())
                    del batch_emb, contexts, batch_words
                
                topic_emb = torch.cat(word_embs, dim=0)
                all_embeddings[topic_idx] = topic_emb.clone()
                del word_embs, topic_emb
            
            if (topic_idx + 1) % 10 == 0:
                gc.collect()

        return XYData.mock(all_embeddings)