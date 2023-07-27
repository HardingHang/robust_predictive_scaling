import os
from pathlib import Path

from gluonts.model.predictor import Predictor
from gluonts.torch.model.tft import TemporalFusionTransformerEstimator
from pytorch_lightning.callbacks import EarlyStopping as TorchEarlyStopping
from gluonts.evaluation import Evaluator, make_evaluation_predictions

from forecaster.model import ForecastModel
from utils import dataentry_to_dataframe

class TFT(ForecastModel):
    def __init__(self, training, validation, prediction_length, context_length, quantiles=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], benchmark ='alibaba'):
        super().__init__(training, validation, prediction_length, context_length, quantiles=quantiles)
        early_stop_callback = TorchEarlyStopping(
            monitor='val_loss',
            patience=10,
            verbose=True,
            mode='min'
        )

        self.estimator =  TemporalFusionTransformerEstimator(
            prediction_length=self.prediction_length,
            context_length=self.context_length,
            freq=self.training_dataset.freq,
            hidden_dim=128,
            batch_size=128,
            lr=1e-3,
            quantiles=quantiles,
            trainer_kwargs={
                "max_epochs": 300,
                "callbacks": [early_stop_callback],
            }
        )

        self.benchmark = benchmark

        if len(quantiles) == 1 and quantiles[0] == 0.5:
            # point forecasting model
            self.path = "saved_models/%s/tft_context%s_prediction%s_point" % (
                self.benchmark, self.context_length, self.prediction_length)
        else:
            self.path = "saved_models/%s/tft_context%s_prediction%s" % (
                self.benchmark, self.context_length, self.prediction_length)
    

