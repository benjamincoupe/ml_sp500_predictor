import pandas as pd
from variables import *
from functions import *
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.model_selection import train_test_split


data = pd.read_csv('data.csv')
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Start with daily, can try monthly
data['Next'] = data['Adj Close_^GSPC'].shift(-1)
target = (data['Next'] > data['Adj Close_^GSPC']).astype(int)
target = target.reset_index(drop=True)
data = data[['Adj Close_XLP', 'Adj Close_XLY']]
predictors = [col for col in data.columns]




train_split = int(split_fraction * data.shape[0])
"""
val_split = int((1 - split_fraction) * (1 - split_backtest) * data.shape[0])
backtest_split = int((1 - split_fraction) * data.shape[0])

train_data = data[:train_split]
val_data = data[train_split:train_split + val_split]
backtest_data = data[train_split + val_split:backtest_split]

"""






features = normalize(data.values, train_split)
features = pd.DataFrame(features)
#features.columns = predictors
# Can get rid of following line?
#features = pd.concat([features, target], axis=1)

train_data = features.loc[0:train_split - 1]
val_data = features.loc[train_split:]

x_train = train_data.values
y_train = target.iloc[0:train_split]

sequence_length = int(past / step)
dataset_train = keras.preprocessing.timeseries_dataset_from_array(
    x_train,
    y_train,
    sequence_length=sequence_length,
    sampling_rate=step,
    batch_size=batch_size,
)

x_val = val_data[predictors].values
y_val = target.iloc[train_split:]

dataset_val = keras.preprocessing.timeseries_dataset_from_array(
    x_val,
    y_val,
    sequence_length=sequence_length,
    sampling_rate=step,
    batch_size=batch_size,
)

for batch in dataset_train.take(1):
    inputs, targets = batch

# Training
inputs = keras.layers.Input(shape=(inputs.shape[1], inputs.shape[2]))
lstm_out = keras.layers.LSTM(32)(inputs)
outputs = keras.layers.Dense(1)(lstm_out)

model = keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")
model.summary()

path_checkpoint = "model_checkpoint.weights.h5"
es_callback = keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=5)

modelckpt_callback = keras.callbacks.ModelCheckpoint(
    monitor="val_loss",
    filepath=path_checkpoint,
    verbose=1,
    save_weights_only=True,
    save_best_only=True,
)

history = model.fit(
    dataset_train,
    epochs=epochs,
    validation_data=dataset_val,
    callbacks=[es_callback, modelckpt_callback],
)

model.load_weights("model_checkpoint.weights.h5")








# Extracting predictors
predictors = data[['Adj Close_XLP', 'Adj Close_XLY']]
predictors = tf.convert_to_tensor(predictors)
normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(predictors)

print(predictors)
predictor_columns = ['Adj Close_XLP', 'Adj Close_XLY']
num_rows = data.shape[0]

train_data = data.iloc[:(num_rows//2)]
# Here, the -1 is used to make sure the shape of train == shape of test
test_data = data.iloc[(num_rows//2):-1]

X_train = train_data[predictor_columns]
Y_train = train_data['Target']

X_test = test_data[predictor_columns]
Y_test = test_data['Target']
print(X_train.shape)

model = Sequential()

# First layer: LSTM (good for time series forecasting)
model.add(LSTM(50, input_shape=(num_rows//2, len(predictor_columns)), activation='relu'))

# Second layer:
model.add(Dense(32, activation='relu'))

# Output layer
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='mean_squared_error')

def predict(X_train, Y_train, X_test, Y_test, predictors, model):

    model.fit(X_train, Y_train, epochs=50, batch_size=32, validation_data=(X_test, Y_test))
    preds = model.predict(X_test)[:,1]
    preds[preds >= 0.6] = 1
    preds[preds < 0.6] = 0
    preds = pd.Series(preds, index=X_test.index, name='Predictions')
    combined = pd.concat([X_test['Target'], preds], axis=1)
    return combined

predict(X_train, Y_train, X_test, Y_test, predictor_columns, model)


"""
Preprocess differently: use returns vs prices, include other indicators like momentum/trend/ratios

"""