import json
from datetime import datetime, timedelta

import pandas as pd
from gluonts.dataset.common import ListDataset
from gluonts.evaluation import Evaluator
from gluonts.evaluation import make_evaluation_predictions
from gluonts.mx.distribution import StudentTOutput  # likelihood
from gluonts.mx.model.deepar import DeepAREstimator
from gluonts.mx.trainer import Trainer
from gluonts.mx.trainer.learning_rate_scheduler import LearningRateReduction
from gluonts.mx.trainer.model_averaging import SelectNBestMean, ModelAveraging

df = pd.read_excel('./Call-Center-Dataset.xlsx')

df["Date"] = pd.to_datetime(df["Date"])

df["Time"] = df["Time"] * 24
df['Time'] = df['Time'].apply(lambda x: (datetime.min + timedelta(hours=x)).strftime('%H:%M:%S'))

df["CallReceivedDateTime"] = pd.to_datetime(df['Date'].dt.strftime('%Y-%m-%d') + ' ' + df['Time'].astype(str))

df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')

df["Answered"] = df['Answered (Y/N)'].apply(lambda y: 1 if y == 'Y' else 0)

dfCopy = df[df['Answered'] == 1].groupby('Date').size().reset_index(name='sumofansweredcalls')

dfCopy.rename(columns={'Date': 'callsReceivedDate'}, inplace=True)

dfCopy["sumofansweredcalls"].astype(int)


dfCopy.set_index('callsReceivedDate', drop=True, inplace=True)

dfCopy.to_excel("factorized-data.xlsx")

training_data = ListDataset(
    [{"start": dfCopy.index[0], "target": dfCopy.sumofansweredcalls[:"2021-02-28T00:00:00.000000"], }],
    freq="D"
)

test_data = ListDataset(
    [{"start": dfCopy.index[60], "target": dfCopy.sumofansweredcalls[:"2021-03-31T00:00:00.000000"]}],
    freq="D"
)



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
print(f"forecast_entry--{forecast_entry}")

evaluator = Evaluator()
agg_metrics, item_metrics = evaluator(iter(tss), iter(forecasts), num_series=len(test_data))

print(json.dumps(agg_metrics, indent=4))

