"""Reactive auto-scalers (baselines).

Paper mapping: Section IV-A.2 (Resource Scalers) — conceptually similar to
Google Autopilot / Kubernetes HPA: scale from statistics of a recent window
of *observed* workload, without any forecasting.

Unlike predictive scalers, reactive ones operate step-by-step on the actual
series: at time t they decide c_t from workload values up to t-1.
"""

from __future__ import annotations

import numpy as np

from metrics.scaling import min_nodes_required


class ReactiveMax:
    """Scale based on the maximum workload within the recent window."""

    def __init__(self, theta: float, window: int = 6):
        self.theta = theta
        self.window = window

    def allocate_series(self, actual: np.ndarray) -> np.ndarray:
        """Allocation for each step t of `actual`, using actuals up to t-1."""
        actual = np.asarray(actual, dtype=float)
        out = np.empty(len(actual), dtype=int)
        for t in range(len(actual)):
            lo = max(0, t - self.window)
            history = actual[lo:t]
            reference = history.max() if len(history) else actual[t]
            out[t] = min_nodes_required(np.asarray([reference]), self.theta)[0]
        return out


class ReactiveAvg:
    """Scale based on an exponentially-decaying weighted average of the
    recent window (paper: half-life of 6 time intervals)."""

    def __init__(self, theta: float, window: int = 6, half_life: float = 6.0):
        self.theta = theta
        self.window = window
        self.half_life = half_life

    def allocate_series(self, actual: np.ndarray) -> np.ndarray:
        actual = np.asarray(actual, dtype=float)
        out = np.empty(len(actual), dtype=int)
        for t in range(len(actual)):
            lo = max(0, t - self.window)
            history = actual[lo:t]
            if len(history) == 0:
                reference = actual[t]
            else:
                # Most recent observation gets the largest weight.
                ages = np.arange(len(history) - 1, -1, -1, dtype=float)
                weights = 0.5 ** (ages / self.half_life)
                reference = float(np.average(history, weights=weights))
            out[t] = min_nodes_required(np.asarray([reference]), self.theta)[0]
        return out
