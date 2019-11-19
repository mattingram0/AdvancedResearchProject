from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def forecast(data, train_days, test_days):
    # Train on the first 6 days to predict the 7th day
    train = data.iloc[0:train_days * 24]

    # Create the SES Model
    model = SimpleExpSmoothing(np.asarray(train['adjusted']))
    model._index = pd.to_datetime(train.index)

    # Fit the model, and forecast
    fit = model.fit()
    pred = fit.forecast((test_days * 24) - 1)
    data['ses'] = list(fit.fittedvalues) + list(pred)

    # print("SES: Optimal Alpha:", str(fit.params['smoothing_level'])[:3])


def forecast_plots(data):
    fig = plt.figure(figsize=(12.8, 9.6), dpi=250)
    ax = fig.add_subplot(1, 1, 1)

    offset = 500

    data.index = pd.to_datetime(data.index, utc=True)
    train = data.iloc[100 + offset:200 + offset]
    train.index = pd.to_datetime(train.index)
    test = data.iloc[200 + offset:210 + offset]
    test.index = pd.to_datetime(test.index)

    model = SimpleExpSmoothing(np.asarray(train['total load actual']))
    model._index = pd.to_datetime(train.index)

    colours = ["orange", "green"]

    # Fit and plot two SES models, with alpha = 0.2, 0.4
    for i in range(1, 3):
        fit = model.fit(smoothing_level=str(0.2 * i))
        pred = fit.forecast(9)
        ax.plot(train.index, fit.fittedvalues,
                label=u'\u03B1'+" = "+str(0.1 * i), color=colours[i - 1])
        ax.plot(test.index, pred, color=colours[i - 1])

    # Fit and plot the optimal SES model - statsmodels finds the optimal alpha
    optimised_fit = model.fit(optimized=True)
    optimised_pred = optimised_fit.forecast(9)
    ax.plot(train.index, optimised_fit.fittedvalues,
            label=u'\u03B1'+" = "+str(optimised_fit.params[
                                        'smoothing_level'])[:3],
            color="darkkhaki")
    ax.plot(test.index, optimised_pred, color="darkkhaki")

    # Plot the training and test values
    ax.plot(train.index, train['total load actual'], label="Training Data",
            marker="o")
    ax.plot(test.index, test['total load actual'], color="grey", marker="o",
            label="Test Data")

    # Add legend and show plot
    ax.legend(loc="best")
    plt.show()

