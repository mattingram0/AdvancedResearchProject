import pandas as pd


def naive_1(data, train_index, forecast_length):
    fitted_values = data[:train_index + 1].shift(1)
    fitted_values[0] = data[0]
    forecasted_values = pd.Series([data[train_index]] * forecast_length)
    return fitted_values.append(forecasted_values).reset_index(drop=True)


def naive_2(data, train_index, forecast_length):
    fitted_values = data[:train_index + 1].shift(1)
    fitted_values[0] = data[0]
    forecasted_values = pd.Series([data[train_index]] * forecast_length)
    return fitted_values.append(forecasted_values).reset_index(drop=True)


def naive_s(data, train_index, forecast_length, seasonality):
    fitted_values = data[:train_index + 1].shift(seasonality)
    fitted_values[:seasonality] = data[:seasonality]

    forecasted_values = []
    for i in range(forecast_length):
        index = train_index - (seasonality - (i % seasonality)) + 1
        forecasted_values.append(data[index])

    return fitted_values.append(pd.Series(forecasted_values)).reset_index(
        drop=True)
