from datasets.load import load_dataset
from sklearn.datasets import fetch_20newsgroups

import random
import pandas as pd
import kagglehub


def wiki_get_texts():
    dataset = load_dataset("wikipedia", language="en", date="20220301", split="train")
    dataset = dataset.select(random.sample(range(0, len(dataset)), 300000))  # type: ignore
    print(dataset)
    return pd.DataFrame({"text": dataset["text"]})  # type: ignore


def ng20_get_texts():
    newsgroups = fetch_20newsgroups(
        subset="all", remove=("headers", "footers", "quotes")
    )
    return pd.DataFrame({"text": newsgroups["data"]})  # type: ignore


def genomics_get_texts():
    docs: pd.DataFrame = load_dataset(
        "irds/medline_2004", "docs", trust_remote_code=True
    ).to_pandas()  # type: ignore

    qrels: pd.DataFrame = load_dataset(
        "irds/medline_2004_trec-genomics-2005", "qrels", trust_remote_code=True
    ).to_pandas()  # type: ignore
    print(qrels)  # type: ignore
    sample = docs.loc[docs.doc_id.isin(qrels.doc_id.tolist())]  # type: ignore

    return pd.DataFrame({"text": sample.title + sample.abstract})


def nyt_get_texts():
    path = kagglehub.dataset_download("tumanovalexander/nyt-articles-data")
    nyt = pd.read_parquet(path, engine="pyarrow")
    return pd.DataFrame({"text": (nyt.title + nyt.excerpt).sample(frac=0.01)})
