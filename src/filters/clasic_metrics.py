import swifter  # noqa: F401
from typing import List
from framework3 import Container, XYData
from framework3.base import BaseFilter
from gensim.corpora.dictionary import Dictionary
from gensim.models import CoherenceModel
from gensim.utils import simple_preprocess
from ..experiments.configs import sources_setup
from nltk.corpus import stopwords

import pandas as pd
import warnings

warnings.filterwarnings(
    "ignore",
    message="Metric `SpearmanCorrcoef` will save all targets and predictions in the buffer.",
)

stop_words = list(set(stopwords.words("english")))


@Container.bind()
class ClasicMetric(BaseFilter):
    def __init__(self, model_path: str, sim_f_name: str) -> None:
        super().__init__(model_path=model_path)  # Initialize the base class
        self._metric = model_path
        self._dictionary: Dictionary | None = None  # Initialize the dictionary
        self._texts: List[str] | None = None
        self.sim_f_name = sim_f_name

    def build_corpus(self, dataset: str) -> "ClasicMetric":
        if Container.storage.check_if_exists(
            hashcode=f"{dataset}_dict",
            context=f"{Container.storage.get_root_path()}/aux_data",
        ) and Container.storage.check_if_exists(
            hashcode=f"{dataset}_corpus",
            context=f"{Container.storage.get_root_path()}/aux_data",
        ):
            self._texts = Container.storage.download_file(
                hashcode=f"{dataset}_corpus",
                context=f"{Container.storage.get_root_path()}/aux_data",
            )
            self._dictionary = Container.storage.download_file(
                hashcode=f"{dataset}_dict",
                context=f"{Container.storage.get_root_path()}/aux_data",
            )
        else:
            texts: pd.DataFrame = sources_setup[dataset]["texts"]()
            self._texts = texts.text.swifter.apply(
                lambda x: simple_preprocess(x)
            ).values.tolist()
            self._dictionary = Dictionary(self._texts)

            Container.storage.upload_file(
                file=self._texts,
                file_name=f"{dataset}_corpus",
                context=f"{Container.storage.get_root_path()}/aux_data",
            )
            Container.storage.upload_file(
                file=self._dictionary,
                file_name=f"{dataset}_dict",
                context=f"{Container.storage.get_root_path()}/aux_data",
            )

        return self

    def predict(self, x: XYData) -> XYData:
        self.build_corpus(self.sim_f_name)
        if self._dictionary is None or self._texts is None:
            raise ValueError(
                "Dictionary is not initialized. Please call fit method first."
            )
        topics = [[str(word) for word in topic] for topic in x.value]
        self._dictionary.add_documents(topics)

        cm = CoherenceModel(
            topics=topics,
            texts=self._texts,
            dictionary=self._dictionary,
            coherence=self._metric,
            topn=10,
        )

        return XYData.mock(cm.get_coherence_per_topic())
