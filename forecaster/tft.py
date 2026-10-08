"""TFT forecaster: learns a pre-specified grid of quantiles.

Paper mapping: Section III-B.2 "Learn Pre-specified Grid of Quantiles".
The Temporal Fusion Transformer [30] is trained by jointly minimizing the
quantile loss summed across all quantile outputs, and directly outputs the
values of the specified quantile levels.
"""

from __future__ import annotations

import shutil
import tempfile

import numpy as np
import pandas as pd
from gluonts.dataset.common import ListDataset
from gluonts.torch.model.tft import TemporalFusionTransformerEstimator

from forecaster.base import QuantileForecaster

FREQ = "10min"


class TFTForecaster(QuantileForecaster):
    def __init__(self, *args, epochs: int = 5, **kwargs):
        super().__init__(*args, **kwargs)
        self.epochs = epochs
        self._predictor = None

    def fit(self, train: np.ndarray) -> "TFTForecaster":
        dataset = ListDataset(
            [{"start": pd.Period("2000-01-01", freq=FREQ), "target": np.asarray(train)}],
            freq=FREQ,
        )
        workdir = tempfile.mkdtemp(prefix="tft_")
        estimator = TemporalFusionTransformerEstimator(
            freq=FREQ,
            context_length=self.context_length,
            prediction_length=self.prediction_length,
            quantiles=list(self.quantiles),  # pre-specified grid of quantiles
            hidden_dim=32,
            lr=1e-3,  # paper Section IV-A.2
            batch_size=32,
            trainer_kwargs={
                "max_epochs": self.epochs,
                "enable_progress_bar": False,
                "logger": False,
                "default_root_dir": workdir,
            },
        )
        self._predictor = estimator.train(dataset)
        shutil.rmtree(workdir, ignore_errors=True)
        return self

    def predict_quantiles(self, context: np.ndarray) -> dict[float, np.ndarray]:
        dataset = ListDataset(
            [
                {
                    "start": pd.Period("2000-01-01", freq=FREQ),
                    "target": np.asarray(context, dtype=np.float32),
                }
            ],
            freq=FREQ,
        )
        forecast = next(iter(self._predictor.predict(dataset)))
        return {tau: forecast.quantile(tau) for tau in self.quantiles}
