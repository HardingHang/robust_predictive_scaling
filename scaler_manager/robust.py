"""Robust auto-scaler with a fixed quantile level.

Paper mapping: Definition 4 / Eq. 4-6 (Robust Auto-Scaling Optimization).
The uncertain constraint "for all w in the uncertainty set" is reformulated
with its robust counterpart (sup over the set), and the supremum is replaced
by the tau-quantile forecast:

    min sum_t c_t   s.t.  w_hat^tau_t / c_t <= theta_t

which decomposes per time step and has the closed-form optimum
c_t = floor(w_hat^tau_t / theta) + 1. The quantile level tau controls the
conservatism of the strategy: higher tau -> more robust against
under-provisioning, at the cost of lower utilization.
"""

from __future__ import annotations

import numpy as np

from scaler_manager.base import QuantileForecasts, Scaler


class RobustScaler(Scaler):
    """Allocate nodes from the forecast at one fixed quantile level tau."""

    def __init__(self, theta: float, tau: float):
        super().__init__(theta)
        if not 0.0 < tau < 1.0:
            raise ValueError("tau must be in (0, 1)")
        self.tau = tau

    def allocate(self, quantile_forecasts: QuantileForecasts) -> np.ndarray:
        self._check_horizon(quantile_forecasts)
        if self.tau not in quantile_forecasts:
            raise KeyError(f"missing forecast for quantile level {self.tau}")
        return self._nodes_for(np.asarray(quantile_forecasts[self.tau]))
