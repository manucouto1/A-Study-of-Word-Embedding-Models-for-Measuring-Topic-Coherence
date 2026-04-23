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

Spearman correlation between automated coherence metrics and human quality judgments. Correlations statistically significant at $p < 0.05$ are marked with $\dagger$, and those at $p < 0.01$ with $\ddagger$. **Bold** values indicate the best score per column within each model family.

---

### 📊 Classical Models

**Co-occurrence computed from the same corpus**

| Metric | 20NG             | NYT              | Geno             | Avg.   |
|--------|------------------|------------------|------------------|--------|
| NPMI   | 0.473‡           | 0.786‡           | 0.485‡           | 0.581  |
| UMASS  | 0.544‡           | 0.648‡           | 0.409‡           | 0.534  |
| CV     | 0.502‡           | 0.562‡           | 0.475‡           | 0.513  |
| UCI    | 0.487‡           | 0.758‡           | 0.466‡           | 0.570  |

**Co-occurrence estimated from Wikipedia**

| Metric | 20NG             | NYT              | Geno             | Avg.   |
|--------|------------------|------------------|------------------|--------|
| NPMI   | 0.766‡           | 0.777‡           | 0.663‡           | 0.735  |
| UMASS  | 0.662‡           | 0.496‡           | 0.510‡           | 0.556  |
| CV     | 0.730‡           | 0.777‡           | 0.660‡           | 0.722  |
| UCI    | 0.731‡           | 0.753‡           | 0.653‡           | 0.712  |

**Ramrakhiyani et al. [1]**

| Metric    | 20NG       | NYT        | Geno       | Avg.      |
|-----------|------------|------------|------------|-----------|
| Tbuckets  | **0.870**  | **0.819**  | **0.729**  | **0.806** |

---

### 🤖 Embedding Models

**Similarity $m_{cos}$ with aggregation $\sigma_a$**

| Model    | 20NG             | NYT              | Geno             | Avg.   |
|----------|------------------|------------------|------------------|--------|
| BERT     | −0.365‡          | −0.041           | 0.014            | −0.131 |
| RoBERTa  | 0.040            | 0.432‡           | 0.099            | 0.190  |
| ALBERT   | 0.414‡           | 0.623‡           | 0.322‡           | 0.453  |
| MPNet    | 0.369‡           | 0.721‡           | 0.274            | 0.455  |
| FastText | 0.701‡           | 0.612‡           | 0.670‡           | 0.661  |
| Word2Vec | 0.523‡           | 0.589‡           | 0.620‡           | 0.578  |
| GloVe    | **0.825‡**       | **0.755‡**       | **0.745‡**       | **0.775** |

**Similarity $m_{dot}$ with aggregation $\sigma_a$**

| Model    | 20NG             | NYT              | Geno             | Avg.   |
|----------|------------------|------------------|------------------|--------|
| BERT     | −0.208†          | −0.111           | 0.242†           | −0.026 |
| RoBERTa  | 0.551‡           | 0.576‡           | 0.012            | 0.380  |
| ALBERT   | 0.578‡           | 0.520‡           | 0.370            | 0.489  |
| MPNet    | 0.592‡           | 0.733‡           | 0.454            | 0.593  |
| FastText | 0.585‡           | 0.510‡           | 0.580‡           | 0.559  |
| Word2Vec | 0.503‡           | 0.623‡           | 0.646‡           | 0.591  |
| GloVe    | **0.832‡**       | **0.819‡**       | **0.741‡**       | **0.797** |

---

### 📌 Key findings

- **GloVe** is the strongest embedding model overall, reaching an average correlation of **0.797** with $m_{dot}$ and **0.775** with $m_{cos}$, outperforming all co-occurrence-based metrics and approaching the performance of the specialized Tbuckets metric.
- Classical embedding models (GloVe, FastText, Word2Vec) consistently outperform contextual transformer-based models (BERT, RoBERTa, ALBERT, MPNet) when embeddings are used out-of-the-box for coherence estimation.
- Wikipedia-based co-occurrence substantially improves the classical metrics compared to same-corpus estimation, especially on smaller corpora such as Genomics.
- BERT shows negative or near-zero correlation in several settings, confirming that contextual representations require adaptation before being useful as coherence signals.

---

[1] Ramrakhiyani et al., *Measuring Topic Coherence through Optimal Word Buckets*, 2017.

---
