from math import sqrt
import numpy as np


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


# Calculate the Mean Average Scaled Error of a prediction
def MASE(predicted, actual, season):
    act = actual.to_numpy()
    pred = predicted.to_numpy()
    temp = ((act - pred) / np.fabs(act[season:] - act[:-season]).mean())
    return np.fabs(temp).mean()


# Calculate the Mean Absolute Error of a prediction
def MAE(predicted, actual):
    act = actual.to_numpy()
    pred = predicted.to_numpy()
    return np.fabs(act - pred).mean()