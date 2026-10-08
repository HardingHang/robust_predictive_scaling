"""Common interface for probabilistic workload forecasters.

Paper mapping: Section III-B (Probabilistic Workload Forecaster).
A quantile forecaster is trained on a historical workload series
(Definition 1) and produces quantile forecasts (Definition 2): for each
requested quantile level tau, an array of shape (H,) over the forecast
horizon.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

# Quantile levels used by the paper's robust scalers (Section IV-C):
# A = {0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99}.
DEFAULT_QUANTILES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)


class QuantileForecaster(ABC):
    """Trains on a workload series and forecasts quantiles for a horizon."""

    def __init__(
        self,
        context_length: int = 72,
        prediction_length: int = 72,
        quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
    ):
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.quantiles = quantiles

    @abstractmethod
    def fit(self, train: np.ndarray) -> "QuantileForecaster":
        """Train the forecaster on the historical workload series."""

    @abstractmethod
    def predict_quantiles(self, context: np.ndarray) -> dict[float, np.ndarray]:
        """Return {tau: array of shape (prediction_length,)} for the horizon
        following the given context series (length >= context_length)."""

    def predict_std(self, context: np.ndarray) -> np.ndarray | None:
        """Predicted per-step standard deviation, if the model exposes a
        parametric distribution (used as U for the adaptive scaler, paper
        Section III-C.2). Quantile-grid models return None."""
        return None
