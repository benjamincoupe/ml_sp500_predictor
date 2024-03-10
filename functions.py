import pandas as pd

def normalize(data, train_split):
    data_mean = data[:train_split].mean(axis=0)
    data_std = data[:train_split].std(axis=0)
    return (data - data_mean) / data_std


def predict(x_train, y_train, x_val, y_val, model):
    model.fit(x_train, y_train)
    preds = model.predict_proba(x_val)[:,1]
    preds[preds >= 0.6] = 1
    preds[preds < 0.6] = 0
    preds = pd.Series(preds, index=x_val.index, name='Predictions')
    combined = pd.concat([y_val, preds], axis=1)
    return combined