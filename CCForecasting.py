import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from gluonts.mx.model.deepar import DeepAREstimator
from gluonts.mx.trainer import Trainer
from gluonts.dataset.common import ListDataset
from gluonts.dataset.field_names import FieldName
from math import factorial
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split

import appLog

df = pd.read_excel('./Call-Center-Dataset.xlsx')

df["Date"] = pd.to_datetime(df["Date"])

df["Time"] = df["Time"] * 24
df['Time'] = df['Time'].apply(lambda x: (datetime.min + timedelta(hours=x)).strftime('%H:%M:%S'))

df["CallReceivedDateTime"] = pd.to_datetime(df['Date'].dt.strftime('%Y-%m-%d') + ' ' + df['Time'].astype(str))

df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')

df["Answered"] = df['Answered (Y/N)'].apply(lambda y: 1 if y == 'Y' else 0)

dfCopy = df[df['Answered'] == 1].groupby('Date').size().reset_index(name='sumofansweredcalls')

dfCopy.rename(columns={'Date': 'callsReceivedDate'}, inplace=True)

dfCopy = dfCopy.set_index('callsReceivedDate')

freq = "D"  # rate at  which dataset is sampled
start_train = pd.Timestamp("2021-01-01")  # start index
start_test = pd.Timestamp(
    "2021-03-27")  # start_index for test_set verify by df_all.columns[40000:].shape == df_test.shape
prediction_length = 7

df_train = dfCopy.iloc[:, 1:45].values
df_test = dfCopy.iloc[:, 46:].values

print(f"df_test.shape--{df_test.shape}")
print(f"df_train--{df_train.shape}")

ts_code = dfCopy['sumofansweredcalls'].values

print(ts_code)

model = DeepAREstimator(
    freq="D",
    num_layers=2,
    num_cells=32,
    cell_type='lstm',
    dropout_rate=0.25,
    prediction_length=prediction_length,
    trainer=Trainer(epochs=10)
)

train_ds = ListDataset([
    {
        FieldName.TARGET: target,
        FieldName.START: start_train,
        FieldName.FEAT_STATIC_CAT: fsc
    }
    for (target, fsc) in zip(df_train,
                             ts_code.reshape(-1, 1))
], freq=freq)

test_ds = ListDataset([
    {
        FieldName.TARGET: target,
        FieldName.START: start_test,
        FieldName.FEAT_STATIC_CAT: fsc
    }
    for (target, fsc) in zip(df_test,
                             ts_code.reshape(-1, 1))
], freq=freq)

predictor = model.train(training_data=train_ds)

#
# training_data, test_gen = split(dfCopy, offset=-36)
#
# # print(training_data)
# print(training_data)

# test_data = test_gen.generate_instances(prediction_length=12, windows=3)
#
# # Train the model and make predictions
# model = DeepAREstimator(
#     prediction_length=7, freq="D", trainer=Trainer(epochs=10)
# ).train(training_data)


# model  = DeepAREstimator(
#     freq="D",
#     prediction_length=7,
#     trainer=Trainer(epochs=10)
# )

# dataset = PandasDataset(dfCopy, target="#sumofansweredcalls")

# print(dataset)
#
# dfCopy['callsReceivedDate'] = pd.to_datetime(dfCopy['callsReceivedDate'])
#
# # np.pad(a, 10, mode='constant', constant_values=12)
#
# training_data = ListDataset(
#     [{"start": dfCopy.callsReceivedDate.min(), "target": dfCopy.sumofansweredcalls.values}],
#     freq="D"
# )
#
# estimator = DeepAREstimator(
#     freq="D",
#     prediction_length=7,
#     trainer=Trainer(epochs=10)
# )
#
# model = estimator.train(dfCopy.train)
#
# predictor = estimator.train(training_data=training_data)

# print(estimator)
