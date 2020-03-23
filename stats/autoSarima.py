import pmdarima as pm
import pandas as pd


# TODO - ensure this works!
def forecast(data, train_hours, test_hours, in_place=True):
    model = pm.auto_arima(data[:train_hours],
                          start_p=0, start_q=0, max_p=2, max_q=2,
                          m=24, start_P=0, start_Q=0, max_Q=2, max_P=2,
                          max_d=2, max_D=2, trace=True, suppress_warnings=True,
                          stepwise=True)
    model.fit(data['total load actual'][:train_hours])

    fcst = pd.concat(
        [pd.Series([0] * train_hours),
         pd.Series(model.predict(n_periods=test_hours))]
    )

    if in_place:
        data['auto sarima'] = fcst
    else:
        return fcst

