"""Uncertainty-aware adaptive robust auto-scaler.

Paper mapping: Section III-C.2 (Adaptive Extension), Definition 5 / Eq. 7-8
and Algorithm 1.

Instead of one fixed quantile level for the whole decision horizon, the
adaptive strategy picks the quantile level per time step based on the
forecast uncertainty U_t:

    U_t < rho_tau  -> use the optimistic quantile tau1
    U_t >= rho_tau -> use the conservative quantile tau2   (tau1 < tau2)

Uncertainty metric (Eq. 8): a pinball-style deviation of every quantile
forecast from the median (0.5) forecast,

    U = sum_i (w^0.5 - w^tau_i) * (tau_i - I(w^tau_i > w^0.5))

which is non-negative and grows as the predictive distribution widens.

For forecasters that output a parametric distribution (e.g. DeepAR), the
predicted standard deviation can be used as U directly (paper Section
III-C.2); pass `std_forecast` to the constructor for that variant.
"""

from __future__ import annotations

import numpy as np

from scaler_manager.base import QuantileForecasts, Scaler

MEDIAN = 0.5


def uncertainty_level(quantiles_at_t: dict[float, float]) -> float:
    """Uncertainty U of one forecasting step (paper Eq. 8).

    Args:
        quantiles_at_t: mapping quantile level -> forecast value at step t.
            Must include the median level 0.5.

    Returns:
        Non-negative U; 0 when all quantile forecasts equal the median.
    """
    if MEDIAN not in quantiles_at_t:
        raise KeyError("uncertainty metric requires the 0.5-quantile forecast")
    median = quantiles_at_t[MEDIAN]
    u = 0.0
    for tau, value in quantiles_at_t.items():
        if tau == MEDIAN:
            continue
        u += (median - value) * (tau - float(value > median))
    return float(u)


class AdaptiveScaler(Scaler):
    """Uncertainty-aware adaptive scaling (paper Algorithm 1).

    Args:
        theta: workload threshold for resource scaling.
        tau1: optimistic quantile level, used when U_t < rho_tau.
        tau2: conservative quantile level, used when U_t >= rho_tau.
        rho_tau: uncertainty threshold for adjusting quantile levels.
        std_forecast: optional array of shape (H,) with the predicted
            standard deviation per step (parametric forecasters). When
            given, it replaces the Eq. 8 quantile-spread metric.
    """

    def __init__(
        self,
        theta: float,
        tau1: float,
        tau2: float,
        rho_tau: float,
        std_forecast: np.ndarray | None = None,
    ):
        super().__init__(theta)
        if not 0.0 < tau1 < tau2 < 1.0:
            raise ValueError("require 0 < tau1 < tau2 < 1")
        if rho_tau < 0:
            raise ValueError("rho_tau must be non-negative")
        self.tau1 = tau1
        self.tau2 = tau2
        self.rho_tau = rho_tau
        self.std_forecast = std_forecast

    def allocate(self, quantile_forecasts: QuantileForecasts) -> np.ndarray:
        horizon = self._check_horizon(quantile_forecasts)
        for tau in (self.tau1, self.tau2, MEDIAN):
            if tau not in quantile_forecasts:
                raise KeyError(f"missing forecast for quantile level {tau}")

        uncertainties = self._uncertainties(quantile_forecasts, horizon)
        nodes_low = self._nodes_for(np.asarray(quantile_forecasts[self.tau1]))
        nodes_high = self._nodes_for(np.asarray(quantile_forecasts[self.tau2]))

        out = np.empty(horizon, dtype=int)
        for t in range(horizon):
            if uncertainties[t] < self.rho_tau:
                out[t] = nodes_low[t]
            else:
                out[t] = nodes_high[t]
        return out

    def _uncertainties(
        self, quantile_forecasts: QuantileForecasts, horizon: int
    ) -> np.ndarray:
        if self.std_forecast is not None:
            std = np.asarray(self.std_forecast, dtype=float)
            if std.shape != (horizon,):
                raise ValueError("std_forecast must match the horizon")
            return std
        return np.array(
            [
                uncertainty_level({tau: float(q[t]) for tau, q in quantile_forecasts.items()})
                for t in range(horizon)
            ]
        )
