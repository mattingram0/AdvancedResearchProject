import statsmodels.api as sm
import pandas as pd


def forecast(data, train_hours, test_hours, in_place=True):
    model2 = sm.tsa.statespace.SARIMAX(
        data['total load actual'][:train_hours],
        order=(0, 1, 1),
        seasonal_order=(0, 1, 1, 24)
    )
    res = model2.fit(disp=0)

    fcst = pd.concat(
        [pd.Series([0] * train_hours), res.forecast(steps=test_hours)]
    )

    if in_place:
        data['sarima'] = fcst
    else:
        return fcst