from statsmodels.tsa.holtwinters import Holt
import pandas as pd
import numpy as np


def forecast(data, train_hours, test_hours):
    # Split data into training and test data
    train = data.iloc[25:train_hours]

    # Create the Holt Model
    model = Holt(np.asarray(train['seasonally differenced']))
    model._index = pd.to_datetime(train.index)

    # Fit the model, and forecast
    fit = model.fit()
    pred = fit.forecast(test_hours)
    data['holt'] = 0
    data['holt'][25:] = list(fit.fittedvalues) + list(pred)

    # print("Holt: Optimal Alpha:", str(fit.params['smoothing_level'])[:4])
    # print("Holt: Optimal Beta:", str(fit.params['smoothing_slope'])[:4])


def undifference(data, train_hours, test_hours):
    start_value = [data['total load actual'][train_hours] for i in range(
        test_hours)]
    cum_forecast = np.cumsum(data['holt'][train_hours:])
    data['holt undiff'] = 0
    data['holt undiff'][train_hours:] = start_value + cum_forecast
