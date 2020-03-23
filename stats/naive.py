import pandas as pd


def naive_1(data,  forecast_length):
    fitted_values = data.shift(1)
    fitted_values[0] = data[0]
    forecasted_values = pd.Series([data[-1]] * forecast_length)
    return fitted_values.append(forecasted_values).reset_index(drop=True)


def naive_2(data, forecast_length):
    fitted_values = data.shift(1)
    fitted_values[0] = data[0]
    forecasted_values = pd.Series([data[-1]] * forecast_length)
    return fitted_values.append(forecasted_values).reset_index(drop=True)


def naive_s(data, forecast_length, seasonality):
    fitted_values = data.shift(seasonality)
    fitted_values[:seasonality] = data[:seasonality]

    forecasted_values = []
    for i in range(forecast_length):
        index = len(data) - (seasonality - (i % seasonality))
        forecasted_values.append(data[index])

    return fitted_values.append(pd.Series(forecasted_values)).reset_index(
        drop=True)
