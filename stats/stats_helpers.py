import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import sys

from statsmodels.tsa.api import seasonal_decompose

from timeit import default_timer as timer
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


# De-seasonalise data using classical decomposition. Also returns indices
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


# Re-seasonalise data
def reseasonalise(data, indices, method):
    for i in range(len(data)):
        if method == "additive":
            data.iloc[i] = data.iloc[i] + indices[i % len(indices)]
        else:
            data.iloc[i] = data.iloc[i] * indices[i % len(indices)]

    return data


# Split the dataset into the 16 sub-datasets
def split_data(df):
    return {
        "Winter": [
            df.loc["2015-01-01 00:00:00+01:00":"2015-02-28 23:00:00+01:00"],
            df.loc["2015-12-01 00:00:00+01:00":"2016-02-29 23:00:00+01:00"],
            df.loc["2016-12-01 00:00:00+01:00":"2017-02-28 23:00:00+01:00"],
            df.loc["2017-12-01 00:00:00+01:00":"2018-02-28 23:00:00+01:00"]
        ],
        "Spring": [
            df.loc["2015-03-01 00:00:00+01:00":"2015-05-31 23:00:00+01:00"],
            df.loc["2016-03-01 00:00:00+01:00":"2016-05-31 23:00:00+01:00"],
            df.loc["2017-03-01 00:00:00+01:00":"2017-05-31 23:00:00+01:00"],
            df.loc["2018-03-01 00:00:00+01:00":"2018-05-31 23:00:00+01:00"]
        ],
        "Summer": [
            df.loc["2015-06-01 00:00:00+01:00":"2015-08-31 23:00:00+01:00"],
            df.loc["2016-06-01 00:00:00+01:00":"2016-08-31 23:00:00+01:00"],
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


# Identify the correct ARIMA order for each season
def identify_arima(df, plot):
    # Optionally pass in a season as command line argument. If we do so,
    # only that season
    seas_in = -1 if len(sys.argv) == 1 else str(sys.argv[1])
    all_data = split_data(df)

    for season in ["Winter", "Spring", "Summer", "Autumn"]:
        # If we passed in our own season, then skip all other seasons
        if seas_in != -1 and seas_in != season:
            continue

        years = all_data[season]
        years = [years[i]["total load actual"] for i in range(4)]
        resids = []

        for yn, y in enumerate(years):

            y, ind = deseasonalise(y, 168, "multiplicative")
            best_order = [1, 0, 0]
            best_const = True
            fitted_model = sm.tsa.ARIMA(y[:-48], order=best_order).fit(disp=-1)
            best = fitted_model.aic
            updated = True

            while updated:
                updated = False
                new_order_1 = best_order[:]
                new_order_1[0] += 1
                new_order_2 = best_order[:]
                new_order_2[2] += 1

                try:
                    aic = sm.tsa.ARIMA(y[:-48], order=new_order_1).fit(
                        disp=-1).aic
                    if aic < best:
                        best = aic
                        best_order = new_order_1[:]
                        best_const = True
                        updated = True
                except (ValueError, np.linalg.LinAlgError) as err:
                    pass

                try:
                    aic = sm.tsa.ARIMA(y[:-48], order=new_order_1).fit(
                        disp=-1, trend="nc").aic
                    if aic < best:
                        best = aic
                        best_order = new_order_1[:]
                        best_const = False
                        updated = True
                except (ValueError, np.linalg.LinAlgError) as err:
                    pass

                try:
                    aic = sm.tsa.ARIMA(y[:-48], order=new_order_2).fit(
                        disp=-1).aic
                    if aic < best:
                        best = aic
                        best_order = new_order_2[:]
                        best_const = True
                        updated = True
                except (ValueError, np.linalg.LinAlgError) as err:
                    pass

                try:
                    aic = sm.tsa.ARIMA(y[:-48], order=new_order_2).fit(
                        disp=-1, trend="nc").aic
                    if aic < best:
                        best = aic
                        best_order = new_order_2[:]
                        best_const = False
                        updated = True
                except (ValueError, np.linalg.LinAlgError) as err:
                    pass

            print("Year:", str(yn + 1), "- Season:", season)
            print("Best Order:", str(best_order))
            print("Best AIC:", str(best))
            print("Best uses Constant:", str(best_const))
            print()
            sys.stderr.flush()
            sys.stdout.flush()

            c = "c" if best_const else "nc"
            fitted_model = sm.tsa.ARIMA(y[:-48], order=best_order).fit(
                disp=-1, trend=c)
            resids.append(fitted_model.resid)

        if plot:
            # Plot Residuals
            fig, axes = plt.subplots(2, 2, figsize=(20, 15), dpi=250)
            plt.suptitle(season + " Residuals", y=0.99)
            for y, (ax, res) in enumerate(zip(axes.flatten(), resids)):
                ax.plot(res)
                ax.set_title("Year:" + str(y + 1))
            plt.show()

            # Plot ACFs of Residuals
            fig, axes = plt.subplots(2, 2, figsize=(20, 15), dpi=250)
            plt.suptitle(season + " AFCs", y=0.99)
            for y, (ax, res) in enumerate(zip(axes.flatten(), resids)):
                plot_acf(res, ax=ax, alpha=0.05, lags=168)
                ax.set_title("Year:" + str(y + 1))
            plt.show()

            # Plot PACFs of Residuals
            fig, axes = plt.subplots(2, 2, figsize=(20, 15), dpi=250)
            plt.suptitle(season + " PAFCs", y=0.99)
            for y, (ax, res) in enumerate(zip(axes.flatten(), resids)):
                plot_pacf(res, ax=ax, alpha=0.05, lags=168)
                ax.set_title("Year:" + str(y + 1))
            plt.show()


# Identify the correct SARIMA order for each season
def identify_sarima(df):
    all_data = split_data(df)

    for season in ["Winter", "Spring", "Summer", "Autumn"]:
        years = all_data[season]
        years = [years[i]["total load actual"] for i in range(1)]
        resids = []

        years = [years[0]]

        for yn, y in enumerate(years):
            start = timer()
            y, ind = deseasonalise(y, 168, "multiplicative")
            best_order = [2, 0, 1]
            best_seasonal_order = [2, 0, 1, 168]
            best_const_trend = 'n'
            start_1 = timer()
            print("Started")
            fitted_model = sm.tsa.SARIMAX(
                y[:-48], order=best_order,
                seasonal_order=best_seasonal_order,
                trend=best_const_trend
            ).fit()
            end_1 = timer()
            print("Time to Fit one Model:", end_1 - start_1)
            sys.exit(0)
            best = fitted_model.aic
            updated = True

            while updated:
                updated = False
                new_orders, new_seas_orders = gen_new_orders(
                    best_order, best_seasonal_order
                )

                for o in new_orders:
                    for so in new_seas_orders:
                        for t in ['n', 'c', 't', 'ct']:
                            try:
                                print("Trying:", o, so, t)
                                aic = sm.tsa.SARIMAX(
                                    y[:-48], order=o,
                                    seasonal_order=so, trend=t
                                ).fit(disp=-1).aic

                                if aic < best:
                                    best = aic
                                    best_order = o
                                    best_seasonal_order = so
                                    best_const_trend = t
                                    updated = True

                                    print("New Best Order:", o)
                                    print("New Best S-Order:", so)
                                    print("New Best Trend:", t)
                                    print("New Best AIC:", best)
                            except ValueError as err:
                                print("Order:", o, " - Seasonal Order:", so)
                                print(err)
            end = timer()

            print("Year:", str(yn + 1), "- Season:", season)
            print("Best Order:", str(best_order))
            print("Best Seasonal Order:", str(best_seasonal_order))
            print("Best AIC:", str(best))
            print("Best uses Constant:", str(best_const_trend))
            print("Time Elapsed:", end - start)
            print()

            fitted_model = sm.tsa.SARIMAX(
                y[:-48], order=best_order, seasonal_order=best_seasonal_order,
                trend=best_const_trend
            ).fit(disp=-1)
            resids.append(fitted_model.resid)


# Vary the order and seasonal order of the current best ARIMA model
def gen_new_orders(order, seasonal_order):
    orders = [order[:], order[:]]
    seasonal_orders = [seasonal_order[:], seasonal_order[:]]
    orders[0][0] += 1
    orders[1][2] += 1
    seasonal_orders[0][0] += 1
    seasonal_orders[1][2] += 1
    return orders, seasonal_orders


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


# Remove both weekly and hourly seasonality
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
