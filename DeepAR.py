import datetime
import pandas as pd
import json
import numpy as np
from gluonts.dataset.common import ListDataset
from gluonts.dataset.util import to_pandas
from gluonts.mx.model.deepar import DeepAREstimator
from gluonts.mx.distribution import ZeroInflatedNegativeBinomialOutput, StudentTOutput  # likelihood
from gluonts.mx.trainer.learning_rate_scheduler import LearningRateReduction
from gluonts.mx.trainer import Trainer
from gluonts.mx.trainer.model_averaging import ModelAveraging, SelectNBestSoftmax, SelectNBestMean

from gluonts.evaluation import Evaluator
from gluonts.evaluation.backtest import make_evaluation_predictions

# import matplotlib.pyplot as plt

df = pd.read_excel("C:\\Users\\vijayan.antonyganamu\\Documents\\Bit\\DS-Projects\\Call-Center\\dfcopy.xlsx")
df.columns = ["month", "noOfCalls"]
df = df[:-1]
df.month = pd.to_datetime(df.month)
df.set_index("month", drop=True, inplace=True)
# print(df.index[60])
training_data = ListDataset(
    [{"start": df.index[0], "target": df.noOfCalls[:"2022-12-01T00:00:00.000000"], }],
    freq="D"
)

entry = next(iter(training_data))
train_series = to_pandas(entry)

test_data = ListDataset(
    [{"start": df.index[60], "target": df.noOfCalls[:"2023-12-01T00:00:00.000000"]}],
    freq="D"
)

entry = next(iter(test_data))
test_series = to_pandas(entry)

callbacks = [
    LearningRateReduction(objective="min",
                          patience=10,
                          base_lr=1e-3,
                          decay_factor=0.5,
                          ),
    ModelAveraging(avg_strategy=SelectNBestMean(num_models=2))
]

estimator = DeepAREstimator(
    freq="D",
    prediction_length=7,
    context_length=36,
    num_layers=2,
    num_cells=40,
    distr_output=StudentTOutput(),
    dropout_rate=0.01,
    trainer=Trainer(  # ctx = mx.context.cpu(),
        epochs=5,
        callbacks=callbacks))

predictor = estimator.train(training_data)

forecast_it, ts_it = make_evaluation_predictions(
    dataset=test_data,  # test dataset
    predictor=predictor,  # predictor
    num_samples=1,  # number of sample paths we want for evaluation
)

forecasts = list(forecast_it)
tss = list(ts_it)
for item in forecasts:
    print(f"forecasts--{item}")
ts_entry = tss[0]
forecast_entry = forecasts[0]
# print(f"forecast_entry--{forecast_entry}")

evaluator = Evaluator()
agg_metrics, item_metrics = evaluator(iter(tss), iter(forecasts), num_series=len(test_data))

print(json.dumps(agg_metrics, indent=4))
