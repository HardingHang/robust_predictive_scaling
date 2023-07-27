import os
from pathlib import Path
from gluonts.mx import Trainer, DeepAREstimator
from gluonts.model.predictor import Predictor
from gluonts.model.forecast import QuantileForecast

from forecaster.model import ForecastModel
from forecaster.models.utils import EarlyStopping
from gluonts.evaluation import Evaluator, make_evaluation_predictions

from utils import dataentry_to_dataframe

class DeepAR(ForecastModel):
    def __init__(self, training, validation,prediction_length, context_length, quantiles=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], benchmark ='alibaba'):
        super().__init__(training, validation, prediction_length, context_length, quantiles=quantiles)
        self.estimator =  DeepAREstimator(
            prediction_length=self.prediction_length,
            context_length=self.context_length,
            freq=self.training_dataset.freq,
            batch_size=128,
            num_cells=128,
            num_layers=3,
            trainer=Trainer(
                learning_rate=1e-3,
                epochs=200,
                callbacks=[EarlyStopping()],
            )

        )

        self.benchmark = benchmark

        if len(quantiles) == 1 and quantiles[0] == 0.5:
            # point forecasting model
            self.path = "saved_models/%s/deepar_context%s_prediction%s_point" % (
                self.benchmark, self.context_length, self.prediction_length)
        else:
            self.path = "saved_models/%s/deepar_context%s_prediction%s" % (
                self.benchmark, self.context_length, self.prediction_length)

        
        