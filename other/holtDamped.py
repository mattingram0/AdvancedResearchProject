from statsmodels.tsa.holtwinters import Holt
import pandas as pd
import numpy as np


def forecast(data, train_hours, test_hours):
    # Train on the first 6 days to predict the 7th day
    train = data.iloc[25:train_hours]

    # Create the SES Model
    model = Holt(np.asarray(train['seasonally differenced']), damped=True)
    model._index = pd.to_datetime(train.index)

    # Fit the model, and forecast
    fit = model.fit()
    pred = fit.forecast(test_hours)
    data['holtDamped'] = 0
    data['holtDamped'][25:] = list(fit.fittedvalues) + list(pred)

    # print("Holt: Optimal Alpha:", str(fit.params['smoothing_level'])[:4])
    # print("Holt: Optimal Beta:", str(fit.params['smoothing_slope'])[:4])


def undifference(data, train_hours, test_hours):
    start_value = [data['total load actual'][train_hours] for i in range(
        test_hours)]
    cum_forecast = np.cumsum(data['holtDamped'][train_hours:])
    data['holtDamped undiff'] = 0
    data['holtDamped undiff'][train_hours:] = start_value + cum_forecast
