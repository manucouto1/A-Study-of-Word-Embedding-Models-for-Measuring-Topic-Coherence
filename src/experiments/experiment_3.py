from src.filters.clasic_metrics import ClasicMetric

from framework3 import F3Pipeline, Cached
from framework3.plugins.optimizer.grid_optimizer import GridOptimizer

from ..metrics.corr_spearman import CorrSpearman


def get_pipeline(dataset: str):
    return F3Pipeline(
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
