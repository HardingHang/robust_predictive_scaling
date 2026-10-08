"""Non-robust predictive scaler based on point forecasts.

Paper mapping: Section IV-A.2 — scalers driven by point forecasts
(e.g. QB5000, TFT-point). A point forecast is the 0.5-quantile forecast,
so this is equivalent to the robust scaler at tau=0.5 and serves as the
non-robust baseline.
"""

from __future__ import annotations

import numpy as np

from scaler_manager.base import QuantileForecasts, Scaler

POINT_QUANTILE = 0.5


class PointScaler(Scaler):
    """Allocate nodes from the median (0.5-quantile) forecast only."""

    def allocate(self, quantile_forecasts: QuantileForecasts) -> np.ndarray:
        self._check_horizon(quantile_forecasts)
        if POINT_QUANTILE not in quantile_forecasts:
            raise KeyError("point scaler requires the 0.5-quantile forecast")
        return self._nodes_for(np.asarray(quantile_forecasts[POINT_QUANTILE]))
