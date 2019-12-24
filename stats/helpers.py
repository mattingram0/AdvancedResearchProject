import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.api import seasonal_decompose


# Seasonally adjust the data using statsmodels decomposition
def decomp_adjust(data, train_hours, test_hours, model):
    data.index = pd.to_datetime(data.index, utc=True)
    decomp = seasonal_decompose(
        data['total load actual'][0:train_hours], model=model,
        freq=24
    )
    seasonality = list(decomp.seasonal[:24]) * int((train_hours +
                                                    test_hours) / 24)

    data['seasonality'] = seasonality
    data['seasonally decomposed'] = \
        data['total load actual'] - seasonality if model == "additive" \
        else data['total load actual'] / seasonality


# Seasonally adjust the data using seasonal indices
def indices_adjust(data, train_hours, test_hours):
    data['moving average'] = data['total load actual'].rolling(24).mean()
    data['seasonal ratio'] = data['total load actual'] / data['moving average']
    seasonal_indices = []
    for i in range(24):
        subset = data['seasonal ratio'][:train_hours][i::24]
        seasonal_indices.append(subset.mean())

    data['seasonal indices'] = seasonal_indices * int(
        (train_hours + test_hours) / 24
    )
    data['seasonally adjusted'] = data['total load actual'] / data[
        'seasonal indices'
    ]


# Non-seasonally difference the data
def difference(data):
    data['differenced'] = data['total load actual'].diff(1)


# Seasonally difference the data
def seasonally_difference(data):
    data['seasonally differenced'] = data['total load actual'].diff(24)


# Difference both seasonally and locally
def double_difference(data):
    seasonally_differenced = data['total load actual'].diff(24)
    data['double differenced'] = seasonally_differenced.diff(1)


# Must have double differenced the data first before calling this function
def test_stationarity(data):
    result_original = adfuller(data["total load actual"], autolag='AIC')
    result_differenced = adfuller(data["seasonally differenced"][25:],
                                  autolag='AIC')
    print("Original Data")
    print(
        "Test Statistic = {:.3f}".format(result_original[0]))  # The error
    # is a bug
    print("P-value = {:.3f}".format(result_original[1]))
    print("Critical values: ")
    for k, v in result_original[4].items():
        print(
            "\t{}: {:.4f} (The data is {}stationary with {}% "
            "confidence)".format(
                k, v, "not " if v < result_original[0] else "",
                100 - int(k[
                          :-1])))

    print("\nSeasonally Differenced Data")
    print("Test Statistic = {:.3f}".format(result_differenced[0]))
    print("P-value = {:.3f}".format(result_differenced[1]))
    print("Critical values: ")

    for k, v in result_differenced[4].items():
        print(
            "\t{}: {:.4f} (The data is {}stationary with {}% "
            "confidence)".format(
                k, v, "not " if v < result_differenced[0] else "",
                100 - int(k[:-1])
                )
            )
