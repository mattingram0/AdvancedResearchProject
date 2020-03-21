import pandas as pd
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt, ExponentialSmoothing


def ses(data, train_index, forecast_length):
    fitted_model = SimpleExpSmoothing(data[:train_index + 1]).fit()
    prediction = fitted_model.predict(0, train_index + forecast_length)
    return prediction, fitted_model.params


def holt(data, train_index, forecast_length):
    fitted_model = Holt(data[:train_index + 1]).fit()
    prediction = fitted_model.predict(0, train_index + forecast_length)
    return prediction, fitted_model.params


def damped(data, train_index, forecast_length):
    fitted_model = Holt(data[:train_index + 1], damped=True).fit()
    prediction = fitted_model.predict(0, train_index + forecast_length)
    return prediction, fitted_model.params


def holt_winters(data, train_index, forecast_length, seasonality):
    fitted_model = ExponentialSmoothing(
        data, seasonal_periods=seasonality, seasonal="mul"
    ).fit()
    prediction = fitted_model.predict(0, train_index + forecast_length)
    return prediction, fitted_model.params


def comb(data, train_index, forecast_length):
    ses_forecast = ses(data, train_index, forecast_length)
    holt_forecast = holt(data, train_index, forecast_length)
    damped_forecast = damped(data, train_index, forecast_length)
    return pd.DataFrame(
        [ses_forecast, holt_forecast, damped_forecast]
    ).mean(axis=0)