# A-Study-of-Word-Embedding-Models-for-Measuring-Topic-Coherence
This repository provides code for evaluating topic coherence using word embeddings. It compares models like Word2Vec, FastText, GloVe, BERT, RoBERTa, ALBERT, and MPNET, showing that embedding-based metrics can outperform traditional methods in assessing the quality of topic model outputs.
---
This repository contains the code and experiments for the paper __"A Study of Word Embedding Models for Measuring Topic Coherence"__. The project investigates the effectiveness of embedding-based approaches for evaluating topic model coherence—a key challenge in topic modeling. While traditional metrics rely on word co-occurrence statistics or human judgments, this study focuses on semantic similarity between word embeddings as a richer and more scalable alternative.

We systematically compare a wide range of word embedding models, including Word2Vec, FastText, GloVe, BERT, RoBERTa, ALBERT, and MPNET, analyzing their ability to measure the coherence of the top words in a topic. Our results show that embedding-based methods are not only competitive with, but often outperform, classical coherence metrics. This work provides a unified and comprehensive perspective on how modern word representations can be leveraged to improve topic evaluation.
## Code
### Experiment 1: Transformer-based embedding models  
This experiment evaluates the ability of **transformer-based models** (such as BERT, RoBERTa, ALBERT, and MPNET) to measure topic coherence through **semantic similarity** between the embeddings of each topic's top words. Various similarity metrics (COSINE and LINEAR) are tested, and their correlation with human judgments is measured using Spearman correlation.  
📂 Code available in `src/experiments/experiment_1`.
```python
F3Pipeline(
    filters=[
        Cached(
            TransformersEmbedder("microsoft/mpnet-base", input_embs=True).grid(
                {
                    "model_path": [
                        "microsoft/mpnet-base",
                        "bert-base-uncased",
                        "roberta-base",
                        "albert-base-v2",
                    ],
                }
            )
        ),
        PerTopicMeanSimilarity("COSINE").grid({"sim_f_name": ["COSINE", "LINEAR"]}),
    ],
    metrics=[CorrSpearman()],
).optimizer(GridOptimizer(scorer=CorrSpearman()))

```
### Experiment 2: Classic word embeddings (Gensim)  
This experiment evaluates **classic word embedding models** such as Word2Vec, GloVe, and FastText. Similar to Experiment 1, it analyzes how similarity-based metrics between embeddings correlate with human evaluations of topic coherence.  
📂 Code available in `src/experiments/experiment_2`.
```python
F3Pipeline(
    filters=[
        Cached(
            GensimEmbedder("word2vec-google-news-300").grid(
                {
                    "model_path": [
                        "word2vec-google-news-300",
                        "glove-wiki-gigaword-300",
                        "fasttext-wiki-news-subwords-300",
                    ],
                }
            )
        ),
        PerTopicMeanSimilarity("COSINE").grid({"sim_f_name": ["COSINE", "LINEAR"]}),
    ],
    metrics=[CorrSpearman()],
).optimizer(GridOptimizer(scorer=CorrSpearman()))

```
### Experiment 3: Traditional coherence metrics  
This experiment compares **traditional coherence metrics** based on word co-occurrence statistics (e.g., c_npmi, u_mass, c_v, c_uci) across different datasets. It serves as a baseline against the embedding-based methods explored in the previous experiments.  
📂 Code available in `src/experiments/experiment_3`.
```python
F3Pipeline(
    filters=[
        Cached(
            ClasicMetric(model_path="c_npmi", sim_f_name=dataset).grid(
                {
                    "model_path": ["c_npmi", "u_mass", "c_v", "c_uci"],
                    "sim_f_name": [dataset],
                }
            )
        ),
    ],
    metrics=[CorrSpearman()],
).optimizer(GridOptimizer(scorer=CorrSpearman()))
```
🔁 All experiments can be executed together using the `main.py` script, enabling a full evaluation in a single pass.
## Results

---

### 📊 Classic Metrics

| Dataset   | Metric  | Corpus     | Spearman |
|-----------|---------|------------|----------|
| 20ng      | c_npmi  | 20ng       | 0.116993 |
|           |         | wiki       | -0.057289|
|           | c_uci   | 20ng       | 0.116309 |
|           |         | wiki       | -0.057837|
|           | c_v     | 20ng       | 0.622631 |
|           |         | wiki       | 0.728601 |
|           | u_mass  | 20ng       | 0.605081 |
|           |         | wiki       | 0.628105 |
| genomics  | c_npmi  | genomics   | -0.220658|
|           |         | wiki       | -0.150980|
|           | c_uci   | genomics   | -0.224564|
|           |         | wiki       | -0.147625|
|           | c_v     | genomics   | 0.448102 |
|           |         | wiki       | 0.690653 |
|           | u_mass  | genomics   | 0.403029 |
|           |         | wiki       | 0.468284 |
| nyt       | c_npmi  | nyt        | 0.556984 |
|           |         | wiki       | 0.728555 |
|           | c_uci   | nyt        | 0.522173 |
|           |         | wiki       | 0.715550 |
|           | c_v     | nyt        | 0.578331 |
|           |         | wiki       | 0.772017 |
|           | u_mass  | nyt        | 0.651695 |
|           |         | wiki       | 0.560366 |

---

### 🤖 Embedding-Based Metrics

| Dataset   | Model                             | Similarity | Spearman |
|-----------|-----------------------------------|------------|----------|
| 20ng      | albert-base-v2                    | COSINE     | 0.607150 |
|           |                                   | LINEAR     | 0.708662 |
|           | bert-base-uncased                 | COSINE     | 0.569558 |
|           |                                   | LINEAR     | 0.666155 |
|           | fasttext-wiki-news-subwords-300   | COSINE     | 0.853550 |
|           |                                   | LINEAR     | 0.584980 |
|           | glove-wiki-gigaword-300           | COSINE     | 0.857683 |
|           |                                   | LINEAR     | 0.831864 |
|           | microsoft/mpnet-base              | COSINE     | 0.571055 |
|           |                                   | LINEAR     | 0.658294 |
|           | roberta-base                      | COSINE     | 0.254252 |
|           |                                   | LINEAR     | 0.404697 |
|           | word2vec-google-news-300          | COSINE     | 0.581495 |
|           |                                   | LINEAR     | 0.503270 |
| genomics  | albert-base-v2                    | COSINE     | 0.052710 |
|           |                                   | LINEAR     | 0.072437 |
|           | bert-base-uncased                 | COSINE     | 0.082923 |
|           |                                   | LINEAR     | 0.254924 |
|           | fasttext-wiki-news-subwords-300   | COSINE     | 0.720706 |
|           |                                   | LINEAR     | 0.580357 |
|           | glove-wiki-gigaword-300           | COSINE     | 0.796962 |
|           |                                   | LINEAR     | 0.741014 |
|           | microsoft/mpnet-base              | COSINE     | 0.039612 |
|           |                                   | LINEAR     | 0.083391 |
|           | roberta-base                      | COSINE     | -0.306558|
|           |                                   | LINEAR     | -0.226572|
|           | word2vec-google-news-300          | COSINE     | 0.540738 |
|           |                                   | LINEAR     | 0.646789 |
| nyt       | albert-base-v2                    | COSINE     | 0.613154 |
|           |                                   | LINEAR     | 0.436060 |
|           | bert-base-uncased                 | COSINE     | 0.376616 |
|           |                                   | LINEAR     | 0.108985 |
|           | fasttext-wiki-news-subwords-300   | COSINE     | 0.521773 |
|           |                                   | LINEAR     | 0.510231 |
|           | glove-wiki-gigaword-300           | COSINE     | 0.752312 |
|           |                                   | LINEAR     | 0.819134 |
|           | microsoft/mpnet-base              | COSINE     | 0.493901 |
|           |                                   | LINEAR     | 0.467015 |
|           | roberta-base                      | COSINE     | 0.383620 |
|           |                                   | LINEAR     | 0.374012 |
|           | word2vec-google-news-300          | COSINE     | 0.460657 |
|           |                                   | LINEAR     | 0.623999 |

---
