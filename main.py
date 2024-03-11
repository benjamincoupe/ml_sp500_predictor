import pandas as pd
from variables import *
from functions import *
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM


data = pd.read_csv('data.csv')
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Start with daily, can try monthly
data['Next'] = data[f'Adj Close_{predicted}'].shift(-1)
target = (data['Next'] > data[f'Adj Close_{predicted}']).astype(int) # If price return is 2 std above from mean
target = target.reset_index(drop=True)
predictor_columns = [f'Adj Close_{predictor}' for predictor in predictors]
data = data[predictor_columns]
predictors = [col for col in data.columns]

train_split = int(split_fraction * data.shape[0])
val_split = int((split_backtest - split_fraction) * data.shape[0])
backtest_split = int((1 - split_backtest) * data.shape[0])

train_data = data[:train_split]
val_data = data[train_split:train_split + val_split]
backtest_data = data[train_split+val_split:]

x_train = train_data.values
y_train = target.iloc[0:train_split]

x_val = val_data.values
y_val = target.iloc[train_split:train_split + val_split]

x_test = backtest_data.values
y_test = target.iloc[train_split + val_split:]

sequence_length = int(past / step)
dataset_train = keras.preprocessing.timeseries_dataset_from_array(
    x_train,
    y_train,
    sequence_length=sequence_length,
    sampling_rate=step,
    batch_size=batch_size,
)

dataset_val = keras.preprocessing.timeseries_dataset_from_array(
    x_val,
    y_val,
    sequence_length=sequence_length,
    sampling_rate=step,
    batch_size=batch_size,
)

# For this, we need to predict without giving y, then check for accuracy
dataset_test = keras.preprocessing.timeseries_dataset_from_array(
    x_test,
    y_test,
    sequence_length=sequence_length,
    sampling_rate=step,
    batch_size=batch_size,
)

for batch in dataset_train.take(1):
    inputs, targets = batch

inputs = keras.layers.Input(shape=(inputs.shape[1], inputs.shape[2]))
lstm_out = keras.layers.LSTM(32)(inputs)
outputs = keras.layers.Dense(1)(lstm_out)

model = keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")

path_checkpoint = "model_checkpoint.weights.h5"
es_callback = keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=5)

modelckpt_callback = keras.callbacks.ModelCheckpoint(
    monitor="val_loss",
    filepath=path_checkpoint,
    verbose=1,
    save_weights_only=True,
    save_best_only=True,
)

"""
history = model.fit(
    dataset_train,
    epochs=epochs,
    validation_data=dataset_val,
    callbacks=[es_callback, modelckpt_callback],
)
"""
model.load_weights("model_checkpoint.weights.h5")

predictions = model.predict(dataset_test)
predictions = pd.Series(predictions.flatten())
predictions = (predictions > 0.5).astype(int)

len_predictions = (predictions.shape)[0]
test_target = (y_test[-len_predictions:]).reset_index(drop=True)

test_eval = pd.concat([test_target, predictions], axis=1)
test_eval.columns = ['Actual', 'Predicted']
test_eval['Win'] = (test_eval['Predicted'] == 1) & (test_eval['Actual'] == 1).astype(int)
test_eval['type1'] = (test_eval['Predicted'] == 1) & (test_eval['Actual'] == 0).astype(int)
test_eval['type2'] = (test_eval['Predicted'] == 0) & (test_eval['Actual'] == 1).astype(int)

wins = test_eval['Win'].sum()
losses = test_eval['type1'].sum()
missed = test_eval['type2'].sum()

# Money weighted backtest
backtest_prices = pd.read_csv('backtest_data.csv')
backtest_prices['Date'] = pd.to_datetime(backtest_prices['Date'])
backtest_prices.set_index('Date', inplace=True)

backtest_prices = backtest_prices[-len_predictions:]
backtest_prices['period_return'] = backtest_prices['Close'] / backtest_prices['Open'] - 1

test_eval.index = backtest_prices.index
backtest_prices['long'] = test_eval['Predicted']
backtest_prices['portfolio_return'] = backtest_prices['period_return'] * backtest_prices['long']
backtest_prices.fillna(0)
backtest_prices['portfolio_value'] = (backtest_prices['portfolio_return'] + 1).cumprod() * 100
backtest_prices['benchmark'] = (backtest_prices['period_return'] + 1).cumprod() * 100

years = int(round(len(backtest_prices.index) / periods, 0))
annualized_portfolio = (backtest_prices.loc[backtest_prices.index[-1], 'portfolio_value'] /
                        backtest_prices.loc[backtest_prices.index[0], 'portfolio_value']) ** (1 / years) - 1
annualized_benchmark = (backtest_prices.loc[backtest_prices.index[-1], 'benchmark'] /
                        backtest_prices.loc[backtest_prices.index[0], 'benchmark']) ** (1 / years) - 1

std_portfolio = backtest_prices['portfolio_return'].std() * (periods**0.5)
std_benchmark = backtest_prices['period_return'].std() * (periods**0.5)

portfolio_sharpe = annualized_portfolio / std_portfolio
benchmark_sharpe = annualized_benchmark / std_benchmark

report = {
    'Annualized Portfolio': annualized_portfolio,
    'Annualized Benchmark': annualized_benchmark,
    'Portfolio STD': std_portfolio,
    'Benchmark STD': std_benchmark,
    'Portfolio Sharpe': portfolio_sharpe,
    'Benchmark Sharpe': benchmark_sharpe,
    'Wins': int(wins),
    'Type 1': int(losses),
    'Type 2': int(missed)
}

report = pd.Series(report)
print(report)

"""
Preprocess differently: use returns vs prices, include other indicators like momentum/trend/ratios
Resample every 3/5 day instead to reduce noise (Rolling window for data collection?)
Find a way to find the tail events? ( Days when market will shoot up)

Normalize inputs: (does looking at price return bypass this?)
normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(predictors)
"""