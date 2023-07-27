from utils import dataentry_to_dataframe

import numpy as np
import mxnet as mx
from gluonts.dataset.common import Dataset
from gluonts.dataset.split import split
from gluonts.mx import SimpleFeedForwardEstimator, DeepAREstimator, MQRNNEstimator, MQCNNEstimator, CanonicalRNNEstimator, Trainer, context
from gluonts.mx.trainer.callback import Callback
from gluonts.evaluation import make_evaluation_predictions, Evaluator
from gluonts.mx import copy_parameters, GluonPredictor
from gluonts.mx.distribution import StudentTOutput
from gluonts.torch.model.tft import TemporalFusionTransformerEstimator
from pytorch_lightning.callbacks import EarlyStopping as TorchEarlyStopping
import pytorch_lightning as pl

import optuna
import time
trainer = pl.Trainer


class EarlyStopping(Callback):
    def __init__(self,
                 patience: int = 10,
                 min_delta: float = 1e4,
                 mode: str = 'min',
                 restore_best_network: bool = True,):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.restore_best_network = restore_best_network

        if self.mode == 'min':
            self.best_score = float('inf')
        else:
            self.best_score = -float('inf')

    def on_validation_epoch_end(
        self,
        epoch_no: int,
        epoch_loss: float,
        training_network: mx.gluon.nn.HybridBlock,
        trainer: mx.gluon.Trainer,
    ) -> bool:
        should_continue = True

        if self.mode == 'min':
            score_improved = epoch_loss < self.best_score
        else:
            score_improved = epoch_loss > self.best_score-self.min_delta

        if score_improved:
            self.best_score = epoch_loss

            if self.restore_best_network:
                training_network.save_parameters("best_network.params")

            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                should_continue = False
                print(
                    f"EarlyStopping callback initiated stop of training at epoch {epoch_no}."
                )

                if self.restore_best_network:
                    print(
                        f"Restoring best network from epoch {epoch_no - self.patience}."
                    )
                    training_network.load_parameters("best_network.params")

        return should_continue


class MetricInferenceEarlyStopping(Callback):
    """
    Callback that implements early stopping based on a metric on the validation dataset.
    """

    def __init__(
        self,
        validation_dataset: Dataset,
        predictor: GluonPredictor,
        evaluator: Evaluator = Evaluator(num_workers=4),
        metric: str = "mean_wQuantileLoss",
        patience: int = 10,
        min_delta: float = 1e4,
        verbose: bool = True,
        minimize_metric: bool = True,
        restore_best_network: bool = True,
        num_samples: int = 100,
    ):
        assert patience >= 0, "EarlyStopping Callback patience needs to be >= 0"
        assert min_delta >= 0, "EarlyStopping Callback min_delta needs to be >= 0.0"
        assert num_samples >= 1, "EarlyStopping Callback num_samples needs to be >= 1"

        self.validation_dataset = list(validation_dataset)
        self.predictor = predictor
        self.evaluator = evaluator
        self.metric = metric
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_network = restore_best_network
        self.num_samples = num_samples

        if minimize_metric:
            self.best_metric_value = np.inf
            self.is_better = np.less
        else:
            self.best_metric_value = -np.inf
            self.is_better = np.greater

        self.validation_metric_history: list[float] = []
        self.best_network = None
        self.n_stale_epochs = 0

    def on_epoch_end(
        self,
        epoch_no: int,
        epoch_loss: float,
        training_network: mx.gluon.nn.HybridBlock,
        trainer: mx.gluon.Trainer,
        best_epoch_info: dict,
        ctx: mx.Context,
    ) -> bool:
        should_continue = True
        copy_parameters(training_network, self.predictor.prediction_net)

        forecast_it, ts_it = make_evaluation_predictions(
            dataset=self.validation_dataset,
            predictor=self.predictor,
            num_samples=self.num_samples,
        )

        agg_metrics, item_metrics = self.evaluator(ts_it, forecast_it)
        current_metric_value = agg_metrics[self.metric]

        if self.is_better(current_metric_value, self.best_metric_value):
            self.best_metric_value = current_metric_value

            if self.restore_best_network:
                training_network.save_parameters("best_network.params")

            self.n_stale_epochs = 0
        else:
            self.n_stale_epochs += 1
            if self.n_stale_epochs == self.patience:
                should_continue = False
                print(
                    f"EarlyStopping callback initiated stop of training at epoch {epoch_no} with epoch loss{epoch_loss}."
                )

                if self.restore_best_network:
                    print(
                        f"Restoring best network from epoch {epoch_no - self.patience}."
                    )
                    training_network.load_parameters("best_network.params")

        return should_continue


class HyperparameterTuningObjective:
    def __init__(self, training, validation, context_length, prediction_length, freq, quantiles = [0.025, 0.1, 0.2, 0.5, 0.8, 0.9, 0.975], metric_type="mean_wQuantileLoss") -> None:
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.freq = freq
        self.metric_type = metric_type
        self.quantiles = quantiles

        self.trainining = training
        self.validation = validation
        # self.train, test_template = split(
        #     dataset, offset=-24*self.prediction_length)

        # self.validation = test_template.generate_instances(
        #     prediction_length=prediction_length,
        #     windows=24,
        # )

        self.validation_input = [entry[0] for entry in self.validation]
        self.validation_label = [
            dataentry_to_dataframe(entry[1]) for entry in self.validation
        ]

    def _get_params(self) -> dict:
        raise NotImplementedError("_get_params missing")

    def __call__(self, trial) -> float:
        raise NotImplementedError("__call__ missing")


class DeepARTuningObjective(HyperparameterTuningObjective):
    # def __init__(self, dataset, context_length, prediction_length, freq, metric_type="mean_wQuantileLoss") -> None:
    #     super().__init__(dataset, context_length,
    #                    prediction_length, freq, metric_type)

    def _get_params(self, trial) -> dict:
        return {
            "num_layers": trial.suggest_int("num_layers", 1, 3),
            # "hidden_size": trial.suggest_int("hidden_size", 16, 128),
            "hidden_size": trial.suggest_categorical("hidden_size", [16, 32, 64, 128]),
            # "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 1e-2),
            # "batch_size": trial.suggest_int("batch_size", 32, 256),
            "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
        }

    def __call__(self, trial) -> float:
        params = self._get_params(trial)

        # train with early stopping
        estimator = DeepAREstimator(
            prediction_length=self.prediction_length,
            context_length=self.context_length,
            freq=self.freq,
            batch_size=params["batch_size"],
            num_cells=params["hidden_size"],
            num_layers=params["num_layers"],
            trainer=Trainer(
                ctx="cpu",
                learning_rate=1e-3,
                callbacks=[EarlyStopping()],
            )

        )
        predictor = estimator.train(
            self.trainining.dataset, validation_data=self.validation.dataset, cache_data=True)
        forecast_it = predictor.predict(self.validation_input)

        forecasts = list(forecast_it)

        evaluator = Evaluator(
            quantiles=self.quantiles)
        agg_metrics, item_metrics = evaluator(
            self.validation_label, forecasts,
        )
        print(agg_metrics)

        return agg_metrics[self.metric_type]


class FeedForwardTuningObjective(HyperparameterTuningObjective):
    def _get_params(self, trial) -> dict:
        return {
            "num_layers": trial.suggest_int("num_layers", 1, 3),
            # "hidden_size": trial.suggest_int("hidden_size", 16, 128),
            "hidden_size": trial.suggest_categorical("hidden_size", [16, 32, 64, 128]),
            # "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 1e-2),
            # "batch_size": trial.suggest_int("batch_size", 32, 256),
            "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
        }

    def __call__(self, trial) -> float:
        params = self._get_params(trial)

        estimator = SimpleFeedForwardEstimator(
            num_hidden_dimensions=[params["hidden_size"]]*params["num_layers"],
            prediction_length=self.prediction_length,
            context_length=self.context_length,
            # freq=self.freq,
            batch_size=params["batch_size"],
            trainer=Trainer(learning_rate=1e-3,
                            callbacks=[EarlyStopping()],),
        )

        predictor = estimator.train(
            self.trainining.dataset, validation_data=self.validation.dataset, cache_data=True)
        forecast_it = predictor.predict(self.validation_input)

        forecasts = list(forecast_it)

        evaluator = Evaluator(
            quantiles=self.quantiles)
        agg_metrics, item_metrics = evaluator(
            self.validation_label, forecasts,
        )
        print(agg_metrics)

        return agg_metrics[self.metric_type]


class CanonicalRNNTuningObjective(HyperparameterTuningObjective):
    def _get_params(self, trial) -> dict:
        return {
            "num_layers": trial.suggest_int("num_layers", 1, 4),
            "hidden_size": trial.suggest_categorical("hidden_size", [16, 32, 64, 128]),
        }

    def __call__(self, trial) -> float:
        params = self._get_params(trial)

        estimator = CanonicalRNNEstimator(
            prediction_length=self.prediction_length,
            context_length=self.context_length,
            freq=self.freq,
            num_cells=params["hidden_size"],
            num_layers=params["num_layers"],
            trainer=Trainer(learning_rate=1e-3,
                            callbacks=[EarlyStopping()]),
        )

        predictor = estimator.train(
            self.trainining.dataset, validation_data=self.validation.dataset, cache_data=True)
        forecast_it = predictor.predict(self.validation_input)

        forecasts = list(forecast_it)

        evaluator = Evaluator(
            quantiles=self.quantiles)
        agg_metrics, item_metrics = evaluator(
            self.validation_label, forecasts,
        )
        print(agg_metrics)

        return agg_metrics[self.metric_type]


class MQRNNTuningObjective(HyperparameterTuningObjective):
    def _get_params(self, trial) -> dict:
        return {
            "num_layers": trial.suggest_int("num_layers", 1, 3),
            "hidden_size": trial.suggest_categorical("hidden_size", [16, 32, 64, 128]),
            "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
        }

    def __call__(self, trial) -> float:
        params = self._get_params(trial)

        estimator = MQRNNEstimator(
            decoder_mlp_dim_seq=[params["hidden_size"]]*params["num_layers"],
            prediction_length=self.prediction_length,
            context_length=self.context_length,
            freq=self.freq,
            batch_size=params["batch_size"],
            quantiles=self.quantiles,
            # distr_output = StudentTOutput(),
            trainer=Trainer(learning_rate=1e-3,
                            callbacks=[EarlyStopping()]),
        )

        predictor = estimator.train(
            self.trainining.dataset, validation_data=self.validation.dataset, cache_data=True)
        forecast_it = predictor.predict(self.validation_input)

        forecasts = list(forecast_it)

        evaluator = Evaluator(
            quantiles=self.quantiles)
        agg_metrics, item_metrics = evaluator(
            self.validation_label, forecasts,
        )
        print(agg_metrics)

        return agg_metrics[self.metric_type]


class MQCNNTuningObjective(HyperparameterTuningObjective):
    def _get_params(self, trial) -> dict:
        return {
            "num_layers": trial.suggest_int("num_layers", 1, 3),
            "hidden_size": trial.suggest_categorical("hidden_size", [16, 32, 64, 128]),
            "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
        }

    def __call__(self, trial) -> float:
        params = self._get_params(trial)

        estimator = MQCNNEstimator(
            decoder_mlp_dim_seq=[params["hidden_size"]]*params["num_layers"],
            prediction_length=self.prediction_length,
            context_length=self.context_length,
            freq=self.freq,
            batch_size=params["batch_size"],
            quantiles=self.quantiles,
            # distr_output = StudentTOutput(),
            trainer=Trainer(learning_rate=1e-3,
                            callbacks=[EarlyStopping()]),
        )

        predictor = estimator.train(
            self.trainining.dataset, validation_data=self.validation.dataset, cache_data=True)
        forecast_it = predictor.predict(self.validation_input)

        forecasts = list(forecast_it)

        evaluator = Evaluator(
            quantiles=self.quantiles)
        agg_metrics, item_metrics = evaluator(
            self.validation_label, forecasts,
        )
        print(agg_metrics)

        return agg_metrics[self.metric_type]


class TFTTuningObjective(HyperparameterTuningObjective):
    def _get_params(self, trial) -> dict:
        return {
            # "num_layers": trial.suggest_int("num_layers", 1, 4),
            "hidden_size": trial.suggest_categorical("hidden_size", [16,32,64,128]),
            "batch_size": trial.suggest_categorical("batch_size", [16,32,64,128]),
        }

    def __call__(self, trial) -> float:
        params = self._get_params(trial)

        # estimator = TemporalFusionTransformerEstimator(
        #     prediction_length=self.prediction_length,
        #     context_length=self.context_length,
        #     freq=self.freq,
        #     hidden_dim=params["hidden_size"],
        #     batch_size=params["batch_size"],
        #     quantiles=self.quantiles,
        #     trainer=Trainer(learning_rate=1e-3,
        #                     callbacks=[EarlyStopping()]),
        # )

        early_stop_callback = TorchEarlyStopping(
            monitor='val_loss',
            patience=10,
            verbose=True,
            mode='min'
        )
        estimator = TemporalFusionTransformerEstimator(
            prediction_length=self.prediction_length,
            context_length=self.context_length,
            freq=self.freq,
            hidden_dim=params["hidden_size"],
            batch_size=params["batch_size"],
            lr=1e-3,
            quantiles=self.quantiles,
            trainer_kwargs={
                "max_epochs": 200,
                "callbacks": [early_stop_callback],
            }
        )

        predictor = estimator.train(
            self.trainining.dataset, validation_data=self.validation.dataset, cache_data=True)
        forecast_it = predictor.predict(self.validation_input)

        forecasts = list(forecast_it)

        evaluator = Evaluator(
            quantiles=self.quantiles)
        agg_metrics, item_metrics = evaluator(
            self.validation_label, forecasts,
        )
        print(agg_metrics)

        return agg_metrics[self.metric_type]
