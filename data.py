import yfinance as yf
import pandas as pd
from variables import *

def download_data(start, end):

    data = yf.download(tickers, start, end)
    data.columns = data.columns.map("_".join)

    for column in data.columns:
        data[column] = pd.to_numeric(data[column], errors='coerce')

    data.to_csv("data.csv")

def backtest_data(start, end):

    data = yf.download(predicted, start, end)
    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    data.to_csv("backtest_data.csv")


if __name__ == "__main__":
    download_data(train_start, train_end)
    backtest_data(train_start, train_end)