# src/metrics/statistical_metrics.py

from labchain import Container, BaseMetric, XYData
from scipy.stats import spearmanr, pearsonr
import numpy as np
from typing import Dict, Any


@Container.bind()
class CorrSpearmanWithPValue(BaseMetric):
    """
    Calcula correlación de Spearman + p-value
    Retorna el p-value (no la correlación)
    """
    
    def evaluate(self, x_data: XYData, y_true: XYData | None, y_pred: XYData) -> float:
        if y_true is None:
            raise ValueError("y_true cannot be None for Spearman correlation calculation")

        y_true_values = np.ravel(y_true.value)
        y_pred_values = np.ravel(y_pred.value)

        # Calculate Spearman correlation with p-value
        res = spearmanr(y_true_values, y_pred_values, nan_policy="omit")

        # Retornar p-value (menor = más significativo)
        return float(res.pvalue) # type: ignore



@Container.bind()
class CorrPearsonWithPValue(BaseMetric):
    """
    Calcula correlación de Pearson + p-value
    Útil para comparar con Spearman
    """
    
    def evaluate(self, x_data: XYData, y_true: XYData | None, y_pred: XYData) -> float:
        if y_true is None:
            raise ValueError("y_true cannot be None for Pearson correlation calculation")

        y_true_values = np.ravel(y_true.value)
        y_pred_values = np.ravel(y_pred.value)

        # Calculate Pearson correlation with p-value
        res = pearsonr(y_true_values, y_pred_values)

        return float(res.pvalue) # type: ignore


@Container.bind()
class BootstrapSpearmanCI(BaseMetric):
    """
    Calcula intervalo de confianza de Spearman usando Bootstrap
    Retorna el ancho del IC (más estrecho = más confianza)
    """
    
    def __init__(self, n_bootstrap: int = 1000, confidence: float = 0.95):
        super().__init__(n_bootstrap=n_bootstrap, confidence=confidence)
        self.n_bootstrap = n_bootstrap
        self.confidence = confidence
    
    def evaluate(self, x_data: XYData, y_true: XYData | None, y_pred: XYData) -> float:
        if y_true is None:
            raise ValueError("y_true cannot be None")

        y_true_values = np.ravel(y_true.value)
        y_pred_values = np.ravel(y_pred.value)
        
        n_samples = len(y_true_values)
        correlations = []
        
        # Bootstrap
        for _ in range(self.n_bootstrap):
            # Resample con reemplazo
            indices = np.random.choice(n_samples, n_samples, replace=True)
            y_true_boot = y_true_values[indices]
            y_pred_boot = y_pred_values[indices]
            
            # Calcular Spearman
            corr, _ = spearmanr(y_true_boot, y_pred_boot, nan_policy="omit")
            correlations.append(corr)
        
        correlations = np.array(correlations)
        alpha = 1 - self.confidence
        
        ci_lower = np.percentile(correlations, 100 * alpha / 2)
        ci_upper = np.percentile(correlations, 100 * (1 - alpha / 2))
        
        # Retornar ancho del intervalo de confianza
        # (más estrecho = más preciso)
        ci_width = ci_upper - ci_lower
        
        return float(ci_width)

