tickers = ["^GSPC", "XLY", "XLP"]

train_start = "1980-1-1"
train_end = "2010-1-1"

test_start = train_end
test_end = "2020-1-1"

backtest_start = test_end
backtest_end = '2022-1-1'

split_fraction = 0.65
split_backtest = 0.95
step = 6
past = 500
future = 1
learning_rate = 0.001
batch_size = 250
epochs = 10
