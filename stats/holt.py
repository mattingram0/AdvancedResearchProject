from statsmodels.tsa.holtwinters import Holt
import pandas as pd
import numpy as np


def forecast(data, train_days, test_days):
    # Split data into training and test data
    train = data.iloc[0:train_days * 24]

    # Create the Holt Model
    model = Holt(np.asarray(train['adjusted']))
    model._index = pd.to_datetime(train.index)

    # Fit the model, and forecast
    fit = model.fit()
    pred = fit.forecast((test_days * 24) - 1)
    data['holt'] = list(fit.fittedvalues) + list(pred)

    # print("Holt: Optimal Alpha:", str(fit.params['smoothing_level'])[:4])
    # print("Holt: Optimal Beta:", str(fit.params['smoothing_slope'])[:4])
