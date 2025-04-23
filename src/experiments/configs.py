from .sources import genomics_get_texts, ng20_get_texts, nyt_get_texts, wiki_get_texts

experiment_setup = {
    "20ng": {
        "topics": "data/topics20NG.txt",
        "gold": "data/gold20NG.txt",
        "texts": ng20_get_texts,
        "results": "notebooks/data/coherence_resutls.csv",
    },
    "nyt": {
        "topics": "data/topicsNYT.txt",
        "gold": "data/goldNYT.txt",
        "texts": nyt_get_texts,
        "results": "notebooks/data/coherence_resutls_NYT.csv",
    },
    "genomics": {
        "topics": "data/topicsGenomics.txt",
        "gold": "data/goldGenomics.txt",
        "texts": genomics_get_texts,
        "results": "notebooks/data/coherence_resutls_genomics.csv",
    },
}

sources_setup = {
    "20ng": {
        "texts": ng20_get_texts,
    },
    "nyt": {
        "texts": nyt_get_texts,
    },
    "genomics": {
        "texts": genomics_get_texts,
    },
    "wiki": {
        "texts": wiki_get_texts,
    },
}
