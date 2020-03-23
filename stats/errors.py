from math import sqrt
import numpy as np
import sys


# Calculate the Mean Absolute Percentage Error of a prediction
def sMAPE(predicted, actual):
    act = actual.to_numpy()
    pred = predicted.to_numpy()
    mask = act != 0
    temp = np.fabs(act[mask] - pred[mask]) / (
            np.fabs(act[mask]) + np.fabs(pred[mask])
        )
    return (2 * temp).mean() * 100


# Calculate the Root Mean Squared Error of a prediction
def RMSE(predicted, actual):
    act = actual.to_numpy()
    pred = predicted.to_numpy()
    return sqrt(np.sum(np.square(act - pred)) / len(actual))


# Calculate the Mean Absolute Error of a prediction
def MAE(predicted, actual):
    act = actual.to_numpy()
    pred = predicted.to_numpy()
    return np.fabs(act - pred).mean()


# Calculate the Mean Average Scaled Error of a prediction
def MASE(predicted, actual, seasonality, test_hours):
    act = actual[-test_hours:].to_numpy()
    pred = predicted.to_numpy()
    prev_1 = actual[seasonality:-test_hours].to_numpy()
    prev_2 = actual[:-(seasonality + test_hours)].to_numpy()
    return np.fabs(act - pred).mean() / np.fabs(prev_1 - prev_2).mean()


# Calculate the Overall Weighted Average (OWA) of the prediction (see M4 paper)
def OWA(naive2_sMAPE, naive2_MASE, pred_sMAPE, pred_MASE):
    return ((pred_sMAPE / naive2_sMAPE) + (pred_MASE / naive2_MASE)) / 2