#from gluonts.ext.r_forecast import
from gluonts.ext.statsforecast import AutoARIMAPredictor
#from statsforecast.models import
from gluonts.evaluation import Evaluator
from gluonts.model.forecast import QuantileForecast
from utils import dataentry_to_dataframe


class ARIMA():
    def __init__(self, training, validation, prediction_length, context_length, quantiles=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]) -> None:
        self.training = training
        self.validation = validation
        self.training_dataset = self.training.dataset
        self.validation_dataset = self.validation.dataset
        self.freq = self.training_dataset.freq
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.quantiles = quantiles
        
        self.estimator = None

        self.predictor = AutoARIMAPredictor(prediction_length=self.prediction_length, quantile_levels=quantiles)


    def predict(self, test):
        self.test_input = [entry[0] for entry in test]
        self.test_label = [
            dataentry_to_dataframe(entry[1]) for entry in test
        ]
        forecasts = []
        for i in self.test_input:
            forecasts.append(self.predictor.predict_item(i))

        evaluator = Evaluator(quantiles=self.quantiles)
        agg_metrics, item_metrics = evaluator(
            self.test_label[:], forecasts[:],
        )
        print(agg_metrics)