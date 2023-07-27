import time
import unittest
import os
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

from gluonts.dataset.pandas import PandasDataset
from gluonts.model.predictor import Predictor
from gluonts.evaluation import make_evaluation_predictions, Evaluator
from gluonts.dataset.split import split
from gluonts.model.forecast import QuantileForecast
from pathlib import Path
from gluonts.torch.model.tft import TemporalFusionTransformerEstimator
from pytorch_lightning.callbacks import EarlyStopping as TorchEarlyStopping
from gluonts.dataset.util import to_pandas

from forecaster.hyparameter_tuning import DeepARTuningObjective, EarlyStopping
from utils import dataentry_to_dataframe

from forecaster.models.tft import TFT
from forecaster.models.deepar import DeepAR

from scale_manager.autoscale_manager import AutoScaleManager
from scale_manager.autoscale_manager_evaluator import AutoScaleManagerEvaluator


class AlibabaResourceScalingTestTestCase(unittest.TestCase):
    def setUp(self) -> None:
        data_length = 1150
        data = pd.read_csv("./data/data_10min.csv")
        df = data[lambda x: x.time_idx < data_length]
        df = df.astype(dict(series=str))

        date_range = pd.date_range(pd.to_datetime(
            '2017-12-21 15:00:22'), periods=df['time_idx'].max()+1, freq='10T')

        df['time_idx'] = df['time_idx'].map(lambda x: date_range[x])

        ds = PandasDataset.from_long_dataframe(
            df, timestamp='time_idx', item_id="series", target="value", freq='10T'
        )

        self.prediction_length = 72
        self.context_length = 72

        train_test_offset = int(data_length*0.3)
        self.training, test_template = split(ds, offset=-train_test_offset)

        self.test = test_template.generate_instances(
            prediction_length=self.prediction_length,
            windows=(train_test_offset-self.prediction_length)//72,
            distance=72,
        )
        # self.test = test_template.generate_instances(
        #     prediction_length=self.prediction_length,
        #     windows=train_test_offset//36,
        #     distance=12,
        # )

        self.training_dataset = self.training.dataset
        self.test_dataset = self.test.dataset

        train_val_offset = int(data_length*0.7*0.25)
        self.training, validation_template = split(
            self.training_dataset, offset=-train_val_offset)

        self.validation = validation_template.generate_instances(
            prediction_length=self.prediction_length,
            windows=(train_val_offset-self.prediction_length)//6,
            distance=6,
        )

        self.training_dataset = self.training.dataset
        self.validation_dataset = self.validation.dataset

        self.threshold = 70

    def test_reactive_scaling(self):
        self.test_input = [entry[0] for entry in self.test]
        self.test_label = [
            dataentry_to_dataframe(entry[1]) for entry in self.test
        ]

        num_test_windows = self.test.windows
        selected_item_ids = 'cpu_util_percent'
        upd_total = 0
        upi_total = 0
        under_utilization_duration_total = 0

        for input, label in zip(self.test_input, self.test_label):
            if input['item_id'] == selected_item_ids:
                input = input['target']
                label = label['cpu_util_percent'].values
                result = np.concatenate((input, label))
                threshold = np.full((self.prediction_length), self.threshold)
                manager = AutoScaleManager()

                plan = manager.reactive_solution(
                    result, thresholds=threshold, metric="max")

                manager_evaluator = AutoScaleManagerEvaluator()
                upd, upi, under_utilization_duration = manager_evaluator.evaluation(
                    plan, self.threshold, label)

                upd_total += upd
                upi_total += upi
                under_utilization_duration_total += under_utilization_duration

        print("Under-Provisioning Duration")
        print("reactive version", upd_total /
              (num_test_windows*self.prediction_length))
        print("Under-Provisioning Intensity")
        print("reactive version", upi_total/num_test_windows)
        print("Under-Utilization Duration")
        print("reactive version", under_utilization_duration_total /
              (num_test_windows*self.prediction_length))

    def test_TFT_non_robust_scaling(self):
        model = TFT(self.training, self.validation, self.prediction_length,
                    self.context_length, quantiles=[0.5])
        model.train()
        labels, results = model.predict(self.test)

        num_test_windows = self.test.windows
        selected_item_ids = 'cpu_util_percent'

        upd_total = 0
        upi_total = 0
        under_utilization_duration_total = 0

        for test_id in range(num_test_windows):
            selected_results = {}
            selected_labels = {}
            tmp_results = []
            tmp_labels = []
            for r, l in zip(results, labels):
                if r.item_id == selected_item_ids and l["item_id"] == selected_item_ids:
                    tmp_results.append(r)
                    tmp_labels.append(l)
            selected_results[selected_item_ids] = tmp_results
            selected_labels[selected_item_ids] = tmp_labels

            single_prediction = selected_results[selected_item_ids][test_id]
            single_observation = selected_labels[selected_item_ids][test_id]['target']

            num_intervals = single_prediction.forecast_array.shape[1]
            threshold = np.full((num_intervals), self.threshold)
            manager = AutoScaleManager()

            plan = manager.basic_solution(
                single_prediction, thresholds=threshold)

            manager_evaluator = AutoScaleManagerEvaluator()
            upd, upi, under_utilization_duration = manager_evaluator.evaluation(
                plan, self.threshold, single_observation)

            upd_total += upd
            upi_total += upi
            under_utilization_duration_total += under_utilization_duration

        print("Under-Provisioning Duration")
        print("vanilla version", upd_total /
              (num_test_windows*self.prediction_length))
        print("Under-Provisioning Intensity")
        print("vanilla version", upi_total/num_test_windows)
        print("Under-Utilization Duration")
        print("vanilla version", under_utilization_duration_total /
              (num_test_windows*self.prediction_length))

    def test_TFT_robust_scaling(self):

        model = TFT(self.training, self.validation, self.prediction_length,
                    self.context_length, quantiles=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
        # model.train()
        labels, results = model.predict(self.test)

        num_test_windows = self.test.windows
        selected_item_ids = 'cpu_util_percent'

        robust_upd_total = 0
        robust_upi_total = 0
        robust_under_utilization_duration_total = 0

        for test_id in range(num_test_windows):
            selected_results = {}
            selected_labels = {}
            tmp_results = []
            tmp_labels = []
            for r, l in zip(results, labels):
                if r.item_id == selected_item_ids and l["item_id"] == selected_item_ids:
                    tmp_results.append(r)
                    tmp_labels.append(l)
            selected_results[selected_item_ids] = tmp_results
            selected_labels[selected_item_ids] = tmp_labels

            single_prediction = selected_results[selected_item_ids][test_id]
            single_observation = selected_labels[selected_item_ids][test_id]['target']

            num_intervals = single_prediction.forecast_array.shape[1]
            threshold = np.full((num_intervals), self.threshold)
            manager = AutoScaleManager()

            quantile = 0.8
            plan = manager.robust_solution(
                single_prediction, quantile, thresholds=threshold)

            manager_evaluator = AutoScaleManagerEvaluator()
            robust_upd, robust_upi, robust_under_utilization_duration = manager_evaluator.evaluation(
                plan, self.threshold, single_observation)

            robust_upd_total += robust_upd
            robust_upi_total += robust_upi
            robust_under_utilization_duration_total += robust_under_utilization_duration

        print("Under-Provisioning Duration")
        print("robust version", robust_upd_total /
              (num_test_windows*self.prediction_length))
        print("Under-Provisioning Intensity")
        print("robust version", robust_upi_total/num_test_windows)
        print("Under-Utilization Duration")
        print("robust version", robust_under_utilization_duration_total /
              (num_test_windows*self.prediction_length))

    def test_tft_adaptive_robust_scaling(self):
        model = TFT(self.training, self.validation, self.prediction_length,
                    self.context_length, quantiles=[0.5, 0.6, 0.7, 0.8, 0.9])
        # model.train()
        labels, results = model.predict(self.test)

        num_test_windows = self.test.windows
        selected_item_ids = 'cpu_util_percent'

        adaptive_robust_upd_total = 0
        adaptive_robust_upi_total = 0
        adaptive_robust_under_utilization_duration_total = 0

        for test_id in range(num_test_windows):
            selected_results = {}
            selected_labels = {}
            tmp_results = []
            tmp_labels = []
            for r, l in zip(results, labels):
                if r.item_id == selected_item_ids and l["item_id"] == selected_item_ids:
                    tmp_results.append(r)
                    tmp_labels.append(l)
            selected_results[selected_item_ids] = tmp_results
            selected_labels[selected_item_ids] = tmp_labels

            single_prediction = selected_results[selected_item_ids][test_id]
            single_observation = selected_labels[selected_item_ids][test_id]['target']

            num_intervals = single_prediction.forecast_array.shape[1]
            threshold = np.full((num_intervals), self.threshold)
            manager = AutoScaleManager()

            adaptive_plan = manager.adaptive_robust_solution(
                single_prediction, quantiles=[0.9, 0.8], thresholds=threshold)

            manager_evaluator = AutoScaleManagerEvaluator()

            adaptive_robust_upd, adaptive_robust_upi, adaptive_robust_under_utilization_duration = manager_evaluator.evaluation(
                adaptive_plan, self.threshold, single_observation)

            adaptive_robust_upd_total += adaptive_robust_upd
            adaptive_robust_upi_total += adaptive_robust_upi
            adaptive_robust_under_utilization_duration_total += adaptive_robust_under_utilization_duration

        print("Under-Provisioning Duration")
        print("robust version", adaptive_robust_upd_total /
              (num_test_windows*self.prediction_length))
        print("Under-Provisioning Intensity")
        print("robust version", adaptive_robust_upi_total/num_test_windows)
        print("Under-Utilization Duration")
        print("robust version", adaptive_robust_under_utilization_duration_total /
              (num_test_windows*self.prediction_length))

    def test_deepar_robust_scaling(self):
        model = DeepAR(self.training, self.validation, self.prediction_length,
                       self.context_length, quantiles=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])

        # model.train()
        labels, results = model.predict(self.test)

        num_test_windows = self.test.windows
        selected_item_ids = 'cpu_util_percent'

        robust_upd_total = 0
        robust_upi_total = 0
        robust_under_utilization_duration_total = 0

        for test_id in range(num_test_windows):
            selected_results = {}
            selected_labels = {}
            tmp_results = []
            tmp_labels = []
            for r, l in zip(results, labels):
                if r.item_id == selected_item_ids and l["item_id"] == selected_item_ids:
                    tmp_results.append(r)
                    tmp_labels.append(l)
            selected_results[selected_item_ids] = tmp_results
            selected_labels[selected_item_ids] = tmp_labels

            single_prediction = selected_results[selected_item_ids][test_id]
            single_observation = selected_labels[selected_item_ids][test_id]['target']

            num_intervals = single_prediction.forecast_array.shape[1]
            threshold = np.full((num_intervals), self.threshold)
            manager = AutoScaleManager()

            quantile = 0.9
            plan = manager.robust_solution(
                single_prediction, quantile, thresholds=threshold)

            manager_evaluator = AutoScaleManagerEvaluator()
            robust_upd, robust_upi, robust_under_utilization_duration = manager_evaluator.evaluation(
                plan, self.threshold, single_observation)

            robust_upd_total += robust_upd
            robust_upi_total += robust_upi
            robust_under_utilization_duration_total += robust_under_utilization_duration

        print("Under-Provisioning Duration")
        print("robust version", robust_upd_total /
              (num_test_windows*self.prediction_length))
        print("Under-Provisioning Intensity")
        print("robust version", robust_upi_total/num_test_windows)
        print("Under-Utilization Duration")
        print("robust version", robust_under_utilization_duration_total /
              (num_test_windows*self.prediction_length))

    def test_adaptive_robust_scaling(self):
        import numpy as np
        model = DeepAR(self.training, self.validation, self.prediction_length,
                    self.context_length, quantiles=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])
        # model = TFT(self.training, self.validation, self.prediction_length,
        #             self.context_length, quantiles=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])
        # model.train()
        labels, results = model.predict(self.test)

        num_test_windows = self.test.windows
        selected_item_ids = 'cpu_util_percent'

        import seaborn as sns
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt

      
        param_values = [0.99, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5]
        # param_values = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]


        matrix_metrics1 = np.empty((len(param_values), len(param_values)))
        matrix_metrics1[:] = np.nan

        matrix_metrics2 = np.empty((len(param_values), len(param_values)))
        matrix_metrics2[:] = np.nan

        for i in range(len(param_values)):
            for j in range(i, len(param_values)):
                param1 = param_values[i]
                param2 = param_values[j]
                robust_upd_total = 0
                robust_upi_total = 0
                robust_under_utilization_duration_total = 0

                for test_id in range(num_test_windows):
                    selected_results = {}
                    selected_labels = {}
                    tmp_results = []
                    tmp_labels = []
                    for r, l in zip(results, labels):
                        if r.item_id == selected_item_ids and l["item_id"] == selected_item_ids:
                            tmp_results.append(r)
                            tmp_labels.append(l)
                    selected_results[selected_item_ids] = tmp_results
                    selected_labels[selected_item_ids] = tmp_labels

                    single_prediction = selected_results[selected_item_ids][test_id]
                    single_observation = selected_labels[selected_item_ids][test_id]['target']

                    num_intervals = single_prediction.forecast_array.shape[1]
                    threshold = np.full((num_intervals), self.threshold)
                    manager = AutoScaleManager()

                    quantiles = [param2, param1]
                    plan = manager.adaptive_robust_solution(
                        single_prediction, quantiles, thresholds=threshold, uncertainty_threshold=6)

                    manager_evaluator = AutoScaleManagerEvaluator()
                    robust_upd, robust_upi, robust_under_utilization_duration = manager_evaluator.evaluation(
                        plan, self.threshold, single_observation)

                    robust_upd_total += robust_upd
                    robust_upi_total += robust_upi
                    robust_under_utilization_duration_total += robust_under_utilization_duration

                metric1_value = robust_upd_total / \
                    (num_test_windows*self.prediction_length) * 100
                metric2_value = robust_under_utilization_duration_total / \
                    (num_test_windows*self.prediction_length) * 100

                matrix_metrics1[i, j] = metric1_value
                matrix_metrics2[i, j] = metric2_value
                # matrix[j, i] = indicator_value


        df_metrics1 = pd.DataFrame(
            matrix_metrics1, index=param_values, columns=param_values)
        df_metrics2 = pd.DataFrame(
            matrix_metrics2, index=param_values, columns=param_values)

        sns.set(font_scale=1.3)
        fig, axs = plt.subplots(1, 2, figsize=(14.8, 6))
        plt.subplots_adjust(left=0.05)
        plt.subplots_adjust(right=0.995)
        sns.heatmap(df_metrics1, annot=True, fmt=".1f",
                    cmap="YlGnBu", ax=axs[0])
        axs[0].set_title('Under-Provisioning Rate (%)',fontdict={'fontsize': 20})
        axs[0].set_xlabel('Optional Quantile Levels',fontdict={'fontsize': 20})
        axs[0].set_ylabel('Optional Quantile Levels',fontdict={'fontsize': 20})

        sns.heatmap(df_metrics2, annot=True, fmt=".1f",
                    cmap="YlGnBu", ax=axs[1])
        axs[1].set_title('Over-Provisioning Rate (%)',fontdict={'fontsize': 20})
        axs[1].set_xlabel('Optional Quantile Levels',fontdict={'fontsize': 20})
        axs[1].set_ylabel('Optional Quantile Levels',fontdict={'fontsize': 20})

        # plt.tight_layout()
        plt.savefig("adaptive alibaba deepar.pdf")
        plt.show()

        # print("Under-Provisioning Duration")
        # print("robust version", robust_upd_total /
        #       (num_test_windows*self.prediction_length))
        # print("Under-Provisioning Intensity")
        # print("robust version", robust_upi_total/num_test_windows)
        # print("Under-Utilization Duration")
        # print("robust version", robust_under_utilization_duration_total /
        #       (num_test_windows*self.prediction_length))

    def test_sensitivity_analysis(self):
        import numpy as np
        model = DeepAR(self.training, self.validation, self.prediction_length,
                       self.context_length, quantiles=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])
        # model = TFT(self.training, self.validation, self.prediction_length,
        #             self.context_length, quantiles=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])
        # model.train()
        labels, results = model.predict(self.test)

        num_test_windows = self.test.windows
        selected_item_ids = 'cpu_util_percent'
        
        markers = ['o', 'v', 's', 'x']
        colors = ['#447293', '#E79B25', '#008A5B', '#A65080']
        cnt = -1
        fig = plt.figure(figsize=[6.4, 3.2])
        for quantiles in [[0.95, 0.9], [0.95, 0.8], [0.9, 0.8], [0.9, 0.7]]:
            upd = []
            opd = []
            cnt +=1
            for i in np.arange(2, 10, 0.5):
                robust_upd_total = 0
                robust_upi_total = 0
                robust_under_utilization_duration_total = 0
                for test_id in range(num_test_windows):
                    selected_results = {}
                    selected_labels = {}
                    tmp_results = []
                    tmp_labels = []
                    for r, l in zip(results, labels):
                        if r.item_id == selected_item_ids and l["item_id"] == selected_item_ids:
                            tmp_results.append(r)
                            tmp_labels.append(l)
                    selected_results[selected_item_ids] = tmp_results
                    selected_labels[selected_item_ids] = tmp_labels

                    single_prediction = selected_results[selected_item_ids][test_id]
                    single_observation = selected_labels[selected_item_ids][test_id]['target']

                    num_intervals = single_prediction.forecast_array.shape[1]
                    threshold = np.full((num_intervals), self.threshold)
                    manager = AutoScaleManager()


                    # plan = manager.robust_solution(
                    #     single_prediction, quantile, thresholds=threshold)
                    
                    
                    plan = manager.adaptive_robust_solution(
                        single_prediction, quantiles, thresholds=threshold, uncertainty_threshold=i)
                    # result = [x / y for x, y in zip(single_observation, plan)]
                    manager_evaluator = AutoScaleManagerEvaluator()
                    robust_upd, robust_upi, robust_under_utilization_duration = manager_evaluator.evaluation(
                        plan, self.threshold, single_observation)
                    robust_upd_total += robust_upd
                    robust_upi_total += robust_upi
                    robust_under_utilization_duration_total += robust_under_utilization_duration
                    
                metric1_value = robust_upd_total / \
                    (num_test_windows*self.prediction_length) * 100
                metric2_value = robust_under_utilization_duration_total / \
                    (num_test_windows*self.prediction_length) * 100
                upd.append(metric1_value)
                opd.append(metric2_value)
            plt.plot(np.arange(2, 10, 0.5), upd, color= colors[cnt], marker = markers[cnt],label=quantiles)
        # plt.plot(np.arange(4, 8, 0.5), opd, label="over-provisioning")
        plt.legend(fontsize=16)
        plt.ylim(0, 5)

        plt.xlabel('Uncertainty Threshold',fontdict={'fontsize': 16})
        # plt.ylabel('Over-Provisioning Rate (%)')
        plt.ylabel('Under-Provisioning Rate (%)', fontdict={'fontsize': 16})
        
        # plt.savefig("sensitivity google tft under.pdf")
        plt.show()
