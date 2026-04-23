from src.filters.clasic_metrics import ClasicMetric

from labchain import F3Pipeline, Cached
from labchain.plugins.optimizer.grid_optimizer import GridOptimizer

from src.metrics.corr_spearman import CorrSpearman
from src.metrics.stat_metric import BootstrapSpearmanCI, CorrPearsonWithPValue, CorrSpearmanWithPValue


def get_pipeline(dataset: str):
    return F3Pipeline(
        filters=[
            Cached(
                ClasicMetric(model_path="c_npmi", sim_f_name=dataset).grid(
                    {
                        "model_path": ["c_npmi", "u_mass", "c_v", "c_uci"],
                        "sim_f_name": [dataset],
                    }
                ),
                cache_filter=False
            ),
        ],
        metrics=[CorrSpearman(),CorrSpearmanWithPValue(), BootstrapSpearmanCI()],
    ).optimizer(GridOptimizer(scorer=CorrSpearman()))
