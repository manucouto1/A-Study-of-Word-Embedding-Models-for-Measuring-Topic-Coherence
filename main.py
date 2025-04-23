from src.experiments.experiment_1 import get_pipeline as pipe1
from src.experiments.experiment_2 import get_pipeline as pipe2
from src.experiments.experiment_3 import get_pipeline as pipe3
from src.experiments.configs import experiment_setup

from framework3.plugins.pipelines.parallel.parallel_mono_pipeline import MonoPipeline
from framework3 import XYData
from rich import print
import pandas as pd


def get_pipeline(dataset):
    return MonoPipeline(filters=[pipe1(), pipe2(), pipe3(dataset), pipe3("wiki")])


def experiments():
    results = []
    for k, v in experiment_setup.items():
        X_data = XYData(
            _hash=f"{k}",
            _path="dataset",
            _value=pd.read_csv(v["topics"], header=None, index_col=0).loc[:, 2:].values,
        )

        Y_data = XYData(
            _hash=f"{k} Y",
            _path="/dataset",
            _value=pd.read_csv(v["gold"], header=None).iloc[:, 0].values.tolist(),
        )

        gs_pipeline = get_pipeline(k)
        gs_pipeline.fit(X_data, Y_data)

        for filter in gs_pipeline.filters:
            res = filter._results

            params_dfs = []
            for col in res.drop(columns=["score"]).columns:
                params_dfs.append(res[col].apply(pd.Series))

            df_flat = pd.concat(
                [
                    *params_dfs,
                    res.drop(columns=res.columns[~res.columns.isin(["score"])]),
                ],
                axis=1,
            )
            df_flat["dataset"] = k
            results.append(df_flat)
    return pd.concat(results)


final = experiments()

clasic = final.loc[~final["sim_f_name"].isin(["COSINE", "LINEAR"])]
embeds = final.loc[final["sim_f_name"].isin(["COSINE", "LINEAR"])]

print("Classic metrics:")
print(clasic.groupby(["dataset", "model_path", "sim_f_name"])["score"].mean())
print("Embedding based mertics:")
print(embeds.groupby(["dataset", "model_path", "sim_f_name"])["score"].mean())
