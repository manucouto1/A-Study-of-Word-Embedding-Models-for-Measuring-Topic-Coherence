from operator import index
from pathlib import Path
from src.experiments.experiment_1 import get_pipeline as pipe1
from src.experiments.experiment_2 import get_pipeline as pipe2
from src.experiments.experiment_3 import get_pipeline as pipe3
from src.experiments.configs import experiment_setup

from labchain.plugins.pipelines.parallel.parallel_mono_pipeline import MonoPipeline
from labchain import XYData
from rich import print
import pandas as pd

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)  # o un número grande


def get_pipeline(dataset):
    return MonoPipeline(filters=[pipe1(),  pipe2(), pipe3(dataset), pipe3("wiki")])


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

        for pipe_id, filter in enumerate(gs_pipeline.filters):
            res = filter._results
            path = Path(f"results/{k}/{pipe_id}.csv")
            path.parent.mkdir(parents=True, exist_ok=True)
            res.to_csv(f"results/{k}/{pipe_id}.csv", index=False)                
            


experiments()


