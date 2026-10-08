"""DeepAR forecaster: learns a parametric distribution.

Paper mapping: Section III-B.2 "Learn parametric distributions". DeepAR
[29] outputs distribution parameters with an autoregressive recurrent
network; the paper uses the Student-t distribution for its heavier tails.
Quantile forecasts are obtained by sampling from the learned distribution.
"""

from __future__ import annotations

import shutil
import tempfile

import numpy as np
import pandas as pd
from gluonts.dataset.common import ListDataset
from gluonts.torch.distributions import StudentTOutput
from gluonts.torch.model.deepar import DeepAREstimator

from forecaster.base import QuantileForecaster

FREQ = "10min"  # 10-minute intervals (paper Section IV-A.1)


class DeepARForecaster(QuantileForecaster):
    def __init__(self, *args, epochs: int = 5, num_samples: int = 200, **kwargs):
        super().__init__(*args, **kwargs)
        self.epochs = epochs
        self.num_samples = num_samples
        self._predictor = None

    def fit(self, train: np.ndarray) -> "DeepARForecaster":
        dataset = ListDataset(
            [{"start": pd.Period("2000-01-01", freq=FREQ), "target": np.asarray(train)}],
            freq=FREQ,
        )
        workdir = tempfile.mkdtemp(prefix="deepar_")
        estimator = DeepAREstimator(
            freq=FREQ,
            context_length=self.context_length,
            prediction_length=self.prediction_length,
            distr_output=StudentTOutput(),  # paper: Student-t for heavy tails
            num_parallel_samples=self.num_samples,
            hidden_size=40,
            num_layers=2,
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

    def _forecast(self, context: np.ndarray):
        dataset = ListDataset(
            [{"start": pd.Period("2000-01-01", freq=FREQ), "target": np.asarray(context)}],
            freq=FREQ,
        )
        return next(iter(self._predictor.predict(dataset)))

    def predict_quantiles(self, context: np.ndarray) -> dict[float, np.ndarray]:
        forecast = self._forecast(np.asarray(context, dtype=np.float32))
        return {tau: forecast.quantile(tau) for tau in self.quantiles}

    def predict_std(self, context: np.ndarray) -> np.ndarray | None:
        forecast = self._forecast(np.asarray(context, dtype=np.float32))
        # SampleForecast: std over samples per time step (paper Section
        # III-C.2: variance/std of the predicted distribution as U).
        return forecast.samples.std(axis=0)
