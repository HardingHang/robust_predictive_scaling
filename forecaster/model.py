import os

from pathlib import Path

from gluonts.model.predictor import Predictor
from gluonts.evaluation import Evaluator

from gluonts.model.forecast import QuantileForecast

from utils import dataentry_to_dataframe
class ForecastModel:
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
        self.path = None

    def train(self):
        predictor = self.estimator.train(
            self.training_dataset, validation_data=self.validation_dataset, cache_data=True)
        
        
        os.makedirs(self.path, exist_ok=True)
        predictor.serialize(Path(self.path))

    def predict(self, test):
        import time
        
        self.test_input = [entry[0] for entry in test]
        self.test_label = [
            dataentry_to_dataframe(entry[1]) for entry in test
        ]
        if os.path.exists(self.path):
            predictor = Predictor.deserialize(Path(self.path))
            start_time = time.perf_counter()
            forecast_it = predictor.predict(self.test_input)
        else:
            print("No model found, train a new one")

        forecasts = list(forecast_it)
        end_time = time.perf_counter()
        execution_time = end_time - start_time

        # 打印运行时间
        print("代码运行时间：", execution_time*1000, "ms")

        evaluator = Evaluator(quantiles=self.quantiles)
        agg_metrics, item_metrics = evaluator(
            self.test_label, forecasts,
        )
        f = open("results.txt", "a")
        print(agg_metrics, file = f)
        f.close()
        print(agg_metrics)

        results = []
        for forecast in forecasts:
            if isinstance(forecast, QuantileForecast):
                results.append(forecast)
            else:
                # convert sampleforecast into quantileforecast
                quantileforecast = forecast.to_quantile_forecast([str(q) for q in self.quantiles])
                results.append(quantileforecast)
        labels = []
        for entry in test:
            labels.append(entry[1])

        return labels, results
