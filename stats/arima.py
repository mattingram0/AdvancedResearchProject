import statsmodels.api as sm
import pmdarima as pm
from timeit import default_timer as timer
import numpy as np
import pandas as pd
import sys

# In ARIMA, may need to change 0 to 1 in predict (if decide differencing is needed,
# and also may need to do typ="levels" instead of "linear" as it is
# currently. By default, if we include differencing of order d in our model then we
# probably  shouldn't use a constant, as this constant is equivalent to
# adding a polynomial trend of order d.
# If we don't have any differencing (i.e our model is stationary to
# begin with), then we probably do want to include a constant which is
# the mean of the series. Failing to use a constant will mean that our
# model will go to zero in long term forecasts, which is not what we want.
# See https://robjhyndman.com/hyndsight/arimaconstants/ and
# https://otexts.com/fpp2/arima-r.html for more information, include how
# the automated process works. Include a constant is the AICc improves,
# basically.


# Calculate ARIMA prediction
def arima(data, forecast_length, order):
    fitted_model = sm.tsa.ARIMA(data, order=order).fit(disp=-1)
    prediction = fitted_model.predict(0, len(data) + forecast_length - 1)
    return prediction, np.concatenate(
        (fitted_model.arparams, fitted_model.maparams)
    ).tolist()


# Approximate diffuse initialisation to avoid the LU Decomposition bug. See:
# https://stackoverflow.com/questions/54136280/sarimax-python-np-linalg-
# linalg-linalgerror-lu-decomposition-error
# https://www.statsmodels.org/dev/generated/statsmodels.tsa.holtwinters.
# ExponentialSmoothing.html

# Calculate SARIMA prediction
def sarima(data, forecast_length, order, seasonal_order):
    try:
        fitted_model = sm.tsa.statespace.SARIMAX(
            data, order=order, seasonal_order=seasonal_order
        ).fit(method='powell', maxiter=25, disp=True, low_memory=True)
    except np.linalg.LinAlgError as err:
        print(err)
        print("SARIMA Forecast Failed. Exiting.")
        sys.exit(0)
    sys.stderr.flush()
    sys.stdout.flush()
    prediction = fitted_model.forecast(forecast_length)
    return prediction, fitted_model.params.to_dict()


# Calculate automatically identified SARIMA model prediction
def auto(data, forecast_length, seasonality):
    fitted_model = pm.auto_arima(
        data,
        start_p=0, start_q=0, max_p=2, max_q=2,
        start_P=0, start_Q=0, max_Q=2, max_P=2,
        m=seasonality, max_d=1, max_D=1, maxiter=25,
        trace=True, suppress_warnings=False, stepwise=True,
        information_criterion='aicc', seasonal=True, stationary=True,
        method="powell", low_memory=True
    )
    sys.stderr.flush()
    sys.stdout.flush()
    # fitted = fitted_model.predict_in_sample(start=0, end=(len(data) - 1))
    prediction = fitted_model.predict(forecast_length)
    params = fitted_model.params().tolist()
    hyper_params = fitted_model.get_params()
    params.append(hyper_params["order"])
    params.append(hyper_params["seasonal_order"])

    # return pd.Series(np.concatenate((fitted, prediction))), params
    return pd.Series(prediction), params
