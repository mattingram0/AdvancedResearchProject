import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.api import seasonal_decompose


# Give training data (must be a multiple of a whole day). Returns
# deseasonalised data, along with the seasonal indices
def deseasonalise(data, seasonality, method):
    # Use symmetric moving average to find the trend
    ma_seas = data.rolling(seasonality, center=True).mean()
    trend = ma_seas.rolling(2).mean().shift(-1)

    # Remove trend
    if method == "additive":
        detrended = data - trend
    else:
        detrended = data / trend

    # Calculate seasonal indices
    seasonal_indices = []
    for i in range(seasonality):
        subset = detrended[i::seasonality]
        seasonal_indices.append(subset.mean())

    # Normalise indices
    if method == "additive":
        seasonal_indices = [
            i - np.mean(seasonal_indices) for i in seasonal_indices
        ]
    else:
        seasonal_indices = [
            seasonality * i / sum(seasonal_indices) for i in seasonal_indices
        ]

    # Repeat indices along whole length of data
    seasonal_indices_repeated = seasonal_indices * int(len(data) / seasonality)
    seasonal_indices_repeated.extend(seasonal_indices)
    seasonal_indices_repeated = seasonal_indices_repeated[:len(data)]

    # Remove seasonality
    if method == "additive":
        deseasonalised = data - seasonal_indices_repeated
    else:
        deseasonalised = data / seasonal_indices_repeated

    return deseasonalised, seasonal_indices


def reseasonalise(data, indices, method):
    for i in range(len(data)):
        if method == "additive":
            data.iloc[i] = data.iloc[i] + indices[i % len(indices)]
        else:
            data.iloc[i] = data.iloc[i] * indices[i % len(indices)]

    return data


def split_data(df):
    return {
        "Winter": [
            df.loc["2015-01-01 00:00:00+01:00":"2015-02-28 23:00:00+01:00"],
            df.loc["2015-12-01 00:00:00+01:00":"2016-02-29 23:00:00+01:00"],
            df.loc["2016-12-01 00:00:00+01:00":"2017-02-28 23:00:00+01:00"],
            df.loc["2017-12-01 00:00:00+01:00":"2018-02-28 23:00:00+01:00"]
        ],
        "Spring": [
            df.loc["2015-12-01 00:00:00+01:00":"2016-02-29 23:00:00+01:00"],
            df.loc["2016-03-01 00:00:00+01:00":"2016-05-31 23:00:00+01:00"],
            df.loc["2017-03-01 00:00:00+01:00":"2017-05-31 23:00:00+01:00"],
            df.loc["2018-03-01 00:00:00+01:00":"2018-05-31 23:00:00+01:00"]
        ],
        "Summer": [
            df.loc["2015-06-01 00:00:00+01:00":"2015-08-31 23:00:00+01:00"],
            df.loc["2015-12-01 00:00:00+01:00":"2016-02-29 23:00:00+01:00"],
            df.loc["2017-06-01 00:00:00+01:00":"2017-08-31 23:00:00+01:00"],
            df.loc["2018-06-01 00:00:00+01:00":"2018-08-31 23:00:00+01:00"]
        ],
        "Autumn": [
            df.loc["2015-09-01 00:00:00+01:00":"2015-11-30 23:00:00+01:00"],
            df.loc["2016-09-01 00:00:00+01:00":"2016-11-30 23:00:00+01:00"],
            df.loc["2017-09-01 00:00:00+01:00":"2017-11-30 23:00:00+01:00"],
            df.loc["2018-09-01 00:00:00+01:00":"2018-11-30 23:00:00+01:00"]
        ]
    }


# ***************** Non Used Functions ************************

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


def double_deseasonalise(data, seasonality, method):
    # Remove weekly seasonal effects first
    daily_average = data.resample('D').mean()
    ma_week = daily_average.rolling(7, center=True).mean()
    week_trend = ma_week.rolling(2).mean().shift(-1)
    seasonal_indices = []

    if method == "additive":
        detrended_week = daily_average - week_trend
    else:
        detrended_week = daily_average / week_trend

    for i in range(7):
        subset = detrended_week[i::7]
        seasonal_indices.append(subset.mean())

    # Normalise
    if method == "additive":
        week_seasonal_indices = [
            i - np.mean(seasonal_indices) for i in seasonal_indices
        ]
    else:
        week_seasonal_indices = [
            7 * i / sum(seasonal_indices) for i in seasonal_indices
        ]

    week_indices = [i for i in week_seasonal_indices for _ in range(24)]
    # Repeat weekly indices along whole length of data
    week_indices_repeated = week_indices * int(len(data) / 168)
    week_indices_repeated.extend(week_indices)
    week_indices_repeated = week_indices_repeated[:len(data)]

    if method == "additive":
        data_w = data - week_indices_repeated
    else:
        data_w = data / week_indices_repeated

    # Now remove the daily effects

    # Use symmetric moving average to find the trend
    ma_seas = data_w.rolling(seasonality, center=True).mean()
    trend = ma_seas.rolling(2).mean().shift(-1)

    if method == "additive":
        detrended = data_w - trend
    else:
        detrended = data_w / trend

    seasonal_indices = []
    for i in range(seasonality):
        subset = detrended[i::seasonality]
        seasonal_indices.append(subset.mean())

    if method == "additive":
        seasonal_indices = [
            i - np.mean(seasonal_indices) for i in seasonal_indices
        ]
    else:
        seasonal_indices = [
            seasonality * i / sum(seasonal_indices) for i in seasonal_indices
        ]

    # Repeat seasonal indices along whole length of data
    seasonal_indices_repeated = seasonal_indices * int(len(data) / seasonality)
    seasonal_indices_repeated.extend(seasonal_indices)
    seasonal_indices_repeated = seasonal_indices_repeated[:len(data)]

    if method == "additive":
        deseasonalised = data_w - seasonal_indices_repeated
    else:
        deseasonalised = data_w / seasonal_indices_repeated

    return deseasonalised, seasonal_indices, week_seasonal_indices, trend, \
           data_w, week_trend


# Seasonally adjust the data using seasonal indices
def indices_adjust(data, train_hours, test_hours, method):
    data['24 - MA'] = data['total load actual'].rolling(24, center=True).mean()
    data['2x24 - MA'] = data['24 - MA'].rolling(2).mean().shift(-1)

    if method == "additive":
        data['detrended'] = data['total load actual'] - data['2x24 - MA']
    else:
        data['detrended'] = data['total load actual'] / data['2x24 - MA']

    seasonal_indices = []
    for i in range(24):
        subset = data['detrended'][:train_hours][i::24]
        seasonal_indices.append(subset.mean())

    # Normalise the indices
    if method == "additive":
        seasonal_indices = [
            i - np.mean(seasonal_indices) for i in seasonal_indices
        ]
    else:
        seasonal_indices = [
            24 * i / sum(seasonal_indices) for i in seasonal_indices
        ]

    # Copy seasonal indices down the entire column
    data['seasonal indices'] = seasonal_indices * int(
        (train_hours + test_hours) / 24
    )

    if method == "additive":
        data['seasonally adjusted'] = data['total load actual'] - data[
            'seasonal indices'
        ]
    else:
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
