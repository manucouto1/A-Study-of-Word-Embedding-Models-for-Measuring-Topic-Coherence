from framework3 import Container, BaseMetric, XYData
from scipy.stats import spearmanr
import numpy as np


@Container.bind()
class CorrSpearman(BaseMetric):
    def evaluate(self, x_data: XYData, y_true: XYData | None, y_pred: XYData) -> float:
        if y_true is None:
            raise ValueError(
                "y_true cannot be None for Spearman correlation calculation"
            )

        y_true_values = np.ravel(y_true.value)
        y_pred_values = np.ravel(y_pred.value)

        # Calculate Spearman correlation
        correlation, _ = spearmanr(y_true_values, y_pred_values, nan_policy="omit")

        return float(np.asarray(correlation).item())
