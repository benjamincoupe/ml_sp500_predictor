predicted = "^GSPC"
predictors = ["XLY", "XLP"]

tickers = [predicted] + predictors

train_start = "1960-1-1"
train_end = '2022-1-1'

split_fraction = 0.65
split_backtest = 0.95
step = 1
past = 100
learning_rate = 0.001
batch_size = 10
epochs = 10

# Periods in a year (for annualized return calculation)
periods = 252

"""
sequence_length: This parameter specifies the length of the input sequences in each batch. In a time series context, 
it represents the number of time steps to consider as input for predicting the next time step. For example, if 
sequence_length is set to 10, each input sequence in a batch will consist of 10 time steps.

sampling_rate: This parameter controls the step size between consecutive time steps. It defines the time interval 
between consecutive samples in each sequence. For instance, if your time series data has time steps at intervals of 
1 hour and you set sampling_rate to 2, the resulting sequences will consider every other hour.

batch_size: This parameter specifies the number of sequences (samples) to include in each batch of the dataset.
"""