"""Forecasting accuracy metrics.

Paper mapping: Section IV-B.1.
All functions operate on one forecasting window: `y` has shape (H,) and
`quantile_forecasts` maps quantile level tau -> array of shape (H,).
"""

from __future__ import annotations

import numpy as np


def quantile_loss(y: np.ndarray, y_quantile: np.ndarray, tau: float) -> float:
    """Pinball (quantile) loss, summed over the horizon (paper Eq. 1-2).

    rho_tau(y, y_hat) = (tau - I(y < y_hat)) * (y - y_hat), which is
    non-negative: underestimation is weighted by tau, overestimation by
    (1 - tau). At tau = 0.5 it reduces to half the L1 (MAE) loss.
    """
    y = np.asarray(y, dtype=float)
    y_quantile = np.asarray(y_quantile, dtype=float)
    diff = y - y_quantile
    return float(np.sum(diff * (tau - (diff < 0))))


def weighted_quantile_loss(y: np.ndarray, y_quantile: np.ndarray, tau: float) -> float:
    """wQL[tau] = 2 * QL_tau / sum(y) (paper Section IV-B.1)."""
    y = np.asarray(y, dtype=float)
    denom = np.sum(np.abs(y))
    if denom == 0:
        raise ValueError("sum of targets is zero; wQL undefined")
    return 2.0 * quantile_loss(y, y_quantile, tau) / float(denom)


def coverage(y: np.ndarray, y_quantile: np.ndarray) -> float:
    """Coverage[tau]: fraction of targets below the tau-quantile forecast.

    For a perfectly calibrated forecaster, coverage[tau] == tau.
    """
    y = np.asarray(y, dtype=float)
    y_quantile = np.asarray(y_quantile, dtype=float)
    return float(np.mean(y <= y_quantile))


def mean_weighted_quantile_loss(
    y: np.ndarray, quantile_forecasts: dict[float, np.ndarray]
) -> float:
    """mean wQL over a set of quantile levels A (paper Section IV-B.1)."""
    return float(
        np.mean(
            [weighted_quantile_loss(y, q, tau) for tau, q in quantile_forecasts.items()]
        )
    )


def mse_from_quantiles(y: np.ndarray, quantile_forecasts: dict[float, np.ndarray]) -> float:
    """MSE of a point estimate derived by averaging the quantile forecasts.

    Paper Section IV-B.1: "we derive the mean value from the forecast
    obtained at the predefined quantiles and utilize it as the point
    prediction".
    """
    y = np.asarray(y, dtype=float)
    point = np.mean(list(quantile_forecasts.values()), axis=0)
    return float(np.mean((y - point) ** 2))
