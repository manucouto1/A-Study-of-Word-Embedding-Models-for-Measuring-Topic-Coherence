import swifter  # noqa: F401
from typing import List, cast
from framework3 import Container, XYData
from framework3.base import BaseFilter
from gensim.corpora.dictionary import Dictionary
from gensim.models import CoherenceModel
from gensim.utils import simple_preprocess
from ..experiments.configs import sources_setup

from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer

import pandas as pd
import warnings
import re
import string

warnings.filterwarnings(
    "ignore",
    message="Metric `SpearmanCorrcoef` will save all targets and predictions in the buffer.",
)

stop_words = list(set(stopwords.words("english")))


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[{}0-9]".format(string.punctuation), " ", text)
    text = re.sub(r"[^A-Za-z0-9 ]+", " ", text)
    text = word_tokenize(text)
    text = [word for word in text if word not in stop_words]
    text = [WordNetLemmatizer().lemmatize(word) for word in text]
    text = " ".join(text)
    return text


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
                lambda x: preprocess_text(x)
            ).values.tolist()

            self._dictionary = Dictionary(
                list(map(lambda x: simple_preprocess(x), cast(List[str], self._texts)))
            )

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

        cleared_topics = []
        for topic in topics:
            cleared_topic = []
            for word in topic:
                if word in list(self._dictionary.token2id.keys()):
                    cleared_topic.append(word)
            cleared_topics.append(cleared_topic)

        topics_df = pd.DataFrame({"topics": cleared_topics})

        cm = CoherenceModel(
            topics=topics_df.loc[topics_df.topics.apply(len) > 0].topics,
            texts=list(map(lambda x: simple_preprocess(x), self._texts)),
            dictionary=self._dictionary,
            coherence=self._metric,
            topn=10,
        )

        topics_df.loc[topics_df.topics.apply(len) > 0, "score"] = (
            cm.get_coherence_per_topic()
        )

        return XYData.mock(topics_df.score.values.tolist())
