import pandas as pd
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt, ExponentialSmoothing


def ses(data, forecast_length):
    fitted_model = SimpleExpSmoothing(data).fit()
    prediction = fitted_model.predict(0, len(data) + forecast_length - 1)
    params = fitted_model.params
    params['initial_seasons'] = params['initial_seasons'].tolist()
    return prediction, params


def holt(data, forecast_length):
    fitted_model = Holt(data).fit()
    prediction = fitted_model.predict(0, len(data) + forecast_length - 1)
    params = fitted_model.params
    params['initial_seasons'] = params['initial_seasons'].tolist()
    return prediction, params


def damped(data,  forecast_length):
    fitted_model = Holt(data, damped=True).fit()
    prediction = fitted_model.predict(0, len(data) + forecast_length - 1)
    params = fitted_model.params
    params['initial_seasons'] = params['initial_seasons'].tolist()
    return prediction, params


def holt_winters(data, forecast_length, seasonality):
    fitted_model = ExponentialSmoothing(
        data, seasonal_periods=seasonality, seasonal="mul"
    ).fit(use_basinhopping=True, remove_bias=True)
    prediction = fitted_model.predict(0, len(data) + forecast_length - 1)
    params = fitted_model.params
    params['initial_seasons'] = params['initial_seasons'].tolist()
    return prediction, params


def comb(data, forecast_length):
    ses_forecast = ses(data, forecast_length)[0]  # Discard model parameters
    holt_forecast = holt(data, forecast_length)[0]
    damped_forecast = damped(data, forecast_length)[0]
    return pd.DataFrame(
        [ses_forecast, holt_forecast, damped_forecast]
    ).mean(axis=0)