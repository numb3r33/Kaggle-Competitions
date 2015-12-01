import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder


def ToWeight(y):
    w = np.zeros(y.shape, dtype=float)
    ind = y != 0
    w[ind] = 1./(y[ind]**2)
    return w


def rmspe(yhat, y):
    w = ToWeight(y)
    rmspe = np.sqrt(np.mean( w * (y - yhat)**2 ))
    return rmspe



def rmspe_xg(yhat, y):
    
    """
    This implementation of Root Mean Square Percentage error
    for XGBoost.
    """

    y = y.get_label()
    y = np.exp(y) - 1
    yhat = np.exp(yhat) - 1
    w = ToWeight(y)
    rmspe = np.sqrt(np.mean(w * (y - yhat)**2))
    return "rmspe", rmspe


def get_object_cols(train_df):
    return [col for col in train_df.columns if train_df[col].dtype == 'O']

def preprocessing(train_df, test_df):
    cols = get_object_cols(train_df)

    for col in cols:
        lbl = LabelEncoder()
        data = pd.concat([train_df[col], test_df[col]])

        lbl.fit(data)

        train_df[col] = lbl.transform(train_df[col])
        test_df[col] = lbl.transform(test_df[col])

    return train_df, test_df

def get_data(train_df, start_date, end_date):
    """
    Gets data between date range
    """
    mask = ((train_df.Date >= start_date) & (train_df.Date <= end_date))
    return train_df[mask]


def create_submission(ids, preds, filename):
    submission_df = pd.DataFrame({'Id': ids, 'Sales': preds})
    submission_df.to_csv('./submissions/' + filename, index=False)