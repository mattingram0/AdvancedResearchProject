import pandas as pd
import numpy as np
from statsmodels.tsa.api import ExponentialSmoothing


def forecast(data, train_hours, test_hours, in_place=True):
    train = data.iloc[:train_hours]

    # Create the Holt-Winters model
    model = ExponentialSmoothing(np.asarray(train['total load actual']),
                                 seasonal_periods=24, seasonal='add')
    model._index = pd.to_datetime(train.index)

    # Fit the model, and forecast
    fit = model.fit()
    pred = fit.forecast(test_hours)
    fcst = list(fit.fittedvalues) + list(pred)

    if in_place:
        data['holtWinters'] = fcst
    else:
        return fcst
