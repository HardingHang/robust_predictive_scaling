import pandas as pd

from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split

from utils import dataentry_to_dataframe
from models.deepar import DeepAR

MODELS = {
    "FeedForwardNetwork": FeedForwardNetwork,
    "DeepAR":DeepAR,

}

class Forecaster():
    def __init__(self, benchmark = "AliClsuter", model = "DeepAR"):
        self.benchmark = benchmark,
        self.model_name = model

    def prepare_data(self, prediction_length, context_length):
        if self.benchmark == "AliCluster":
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

        
        self.prediction_length = prediction_length
        self.context_length = context_length

        train_test_offset = int(data_length*0.3)
        self.training, test_template = split(ds, offset = -train_test_offset)

        self.test = test_template.generate_instances(
            prediction_length=self.prediction_length,
            windows=(train_test_offset-self.prediction_length)//6,
            distance=6,
        )

        self.training_dataset=self.training.dataset
        self.test_dataset=self.test.dataset

        train_val_offset = int(data_length*0.7*0.25)
        self.training, validation_template=split(
            self.training_dataset, offset=-train_val_offset)

        self.validation=validation_template.generate_instances(
            prediction_length=self.prediction_length,
            windows=(train_val_offset-self.prediction_length)//6,
            distance = 6,
        )

        self.training_dataset=self.training.dataset
        self.validation_dataset=self.validation.dataset

        self.test_input=[entry[0] for entry in self.test]
        self.test_label=[
            dataentry_to_dataframe(entry[1]) for entry in self.test
        ]

    def train(self):
        pass