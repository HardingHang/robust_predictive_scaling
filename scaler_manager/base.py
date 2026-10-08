"""Common interface for auto-scaling strategies.

Paper mapping: Section III-C. A scaler turns (quantile) workload forecasts
into an allocation of compute nodes for each step of the decision horizon.

`quantile_forecasts` maps a quantile level tau (0 < tau < 1) to an array of
shape (H,) with the forecasted workload at that quantile for each future
step. All scalers return an int array of shape (H,) with node counts.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from metrics.scaling import min_nodes_required

QuantileForecasts = dict[float, np.ndarray]


class Scaler(ABC):
    """Base class for predictive scalers driven by quantile forecasts."""

    def __init__(self, theta: float):
        if theta <= 0:
            raise ValueError("theta must be positive")
        self.theta = theta

    @abstractmethod
    def allocate(self, quantile_forecasts: QuantileForecasts) -> np.ndarray:
        """Return the node allocation for each step of the horizon."""

    def _nodes_for(self, forecast: np.ndarray) -> np.ndarray:
        """Closed-form optimum of Eq. 6: smallest c with w / c < theta."""
        return min_nodes_required(forecast, self.theta)

    @staticmethod
    def _check_horizon(quantile_forecasts: QuantileForecasts) -> int:
        horizons = {np.asarray(q).shape for q in quantile_forecasts.values()}
        if len(horizons) != 1:
            raise ValueError("all quantile forecasts must share the horizon")
        return horizons.pop()[0]
