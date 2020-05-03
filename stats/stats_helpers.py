import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
import sys
import json
import os

from statsmodels.tsa.api import seasonal_decompose
from numpy.polynomial.polynomial import polyfit
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from functools import reduce
from timeit import default_timer as timer
from math import fabs
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from stats.errors import sMAPE


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

# Plot the test result forecasts for all the stats models for a given
# season, year, and test number.
def plot_forecasts(df, season, year):
    split = split_data(df)
    for test in range(8, 1, -1):
        fig, axes = plt.subplots(2, 1, figsize=(20, 15), dpi=250)

        test_path = "/Users/matt/Projects/AdvancedResearchProject/results" \
                    "/non_ensemble_results/res/"
        (_, _, filenames) = next(os.walk(test_path))

        train_end = -(test * 24)
        test_end = -(test * 24 - 48) if test > 2 else None
        actual = split[season][year]["total load actual"][train_end:test_end].tolist()

        axes[0].plot(actual, label="Actual")
        axes[1].plot(actual, label="Actual")

        for file in filenames:
            if "forecasts" not in file:
                continue

            seas, method, _ = file.split("_")

            if seas != season:
                continue

            # methods = ["Naive1", "Naive2", "NaiveS", "SES", "Holt", "Damped",
            #            "Holt-Winters", "Comb", "ARIMA", "SARIMA", "Auto", "Theta",
            #            "TSO", "ES-RNN-S", "ES-RNN-SW", "ES-RNN-D", "ES-RNN-DW",
            #            "ES-RNN-I", "ES-RNN-IW"]
            # good = [""]

            # top = ["NaiveS", "ES-RNN-I", "Comb", "Holt", "Naive2", "SES", "TSO"]
            top = ["NaiveS", "TSO"]
            bottom = ["Theta", "Damped", "ARIMA", "SARIMA", "Holt-Winters", "Naive1"]

            with open(test_path + file) as f:
                all_forecasts = json.load(f)
                forecast = all_forecasts[str(year)][str(1)][str(test - 1)]

            # Plot 6 tests on first axes
            if method in top:
                axes[0].plot(forecast, label=method, marker='o')
            elif method in bottom:
                axes[1].plot(forecast, label=method, marker='o')
            else:
                pass  # To handle the empty 'Auto' forecasts

        axes[0].legend(loc="best")
        axes[1].legend(loc="best")
        plt.show()


def check_errors(df):
    forecast_length = 48
    base = "/Users/matt/Projects/AdvancedResearchProject/results" \
           "/non_ensemble_results/res/"
    seasons = ["Spring_", "Summer_", "Autumn_", "Winter_"]
    methods = ["Naive2_forecasts.txt", "NaiveS_forecasts.txt", "TSO_forecasts.txt", "ES-RNN-I_forecasts.txt"]
    seas_dict = {1: "Spring", 2: "Summer", 3: "Autumn", 4: "Winter"}

    # Begin with a hard coded season:
    seas = 3  # 0: Spring, 1: Summer, 2: Autumn, 3: Winter
    seas_n = seas_dict[seas + 1]

    split = split_data(df)

    # Load forecasts
    with open(base + seasons[seas] + methods[0]) as f:
        naive2_forecasts = json.load(f)

    with open(base + seasons[seas] + methods[1]) as f:
        naives_forecasts = json.load(f)

    with open(base + seasons[seas] + methods[2]) as f:
        tso_forecasts = json.load(f)

    with open(base + seasons[seas] + methods[3]) as f:
        es_rnn_i_forecasts = json.load(f)

    # TSO Results
    with open(base + seasons[seas] + "TSO_results.txt") as f:
        tso_results = json.load(f)

    # Calculate sMAPES:
    tso_smapes = []
    es_rnn_smapes = []
    naive2_smapes = []
    naives_smapes = []
    tso_test_smapes = []
    for y in range(1, 5):
        for t in range(1, 8):
            train_end = -((t + 1) * 24)
            test_end = -((t + 1) * 24 - forecast_length) if (t + 1) > 2 else None
            naive2 = naive2_forecasts[str(y)][str(1)][str(t)]
            naives = naives_forecasts[str(y)][str(1)][str(t)]
            tso = tso_forecasts[str(y)][str(1)][str(t)]
            es_rnn = es_rnn_i_forecasts[str(y)][str(1)][str(t)]
            actual = split[seas_n][y - 1]["total load actual"][
                     train_end:test_end].tolist()
            tso_smapes.append(sMAPE(pd.Series(actual), pd.Series(tso)))
            es_rnn_smapes.append(sMAPE(pd.Series(actual), pd.Series(es_rnn)))
            naive2_smapes.append(sMAPE(pd.Series(actual), pd.Series(naive2)))
            naives_smapes.append(sMAPE(pd.Series(actual), pd.Series(naives)))
            tso_test_smapes.append(tso_results["sMAPE"][str(1)][str(y)][str(
                t)][47])

    print("Average ES-RNN-I sMAPE:", np.mean(es_rnn_smapes))
    print("Average TSO sMAPE:", np.mean(tso_smapes))
    print("Average TSO (Results):", np.mean(tso_test_smapes))
    print("Average Naive2 sMAPE:", np.mean(naive2_smapes))
    print("Average NaiveS sMAPE:", np.mean(naives_smapes))


def plot_48_results():
    methods = ["Naive1", "Naive2", "NaiveS", "SES", "Holt", "Damped",
               "Holt-Winters", "Comb", "ARIMA", "SARIMA", "Auto", "Theta",
               "TSO", "ES-RNN-S", "ES-RNN-SW", "ES-RNN-D", "ES-RNN-DW",
               "ES-RNN-I", "ES-RNN-IW"]
    circles = ["ES-RNN-S", "ES-RNN-SW", "ES-RNN-D", "ES-RNN-DW",
               "ES-RNN-I", "ES-RNN-IW"]
    # exclude = ["Naive1", "Auto", "SARIMA"]
    exclude = ["TSO", "Auto", "Naive1"]
    sns.reset_orig()
    clrs = sns.color_palette('husl', n_colors=len(methods))
    x_tick_locs = [0] + [i for i in range(-1, 48, 5)][1:] + [47]
    x_tick_labels = [1] + [i + 1 for i in range(-1, 48, 5)][1:] + [48]

    # Load OWA
    with open("/Users/matt/Projects/AdvancedResearchProject/results/"
              "non_ensemble_results/res/results_48_owa.txt") as file:
        owas48 = json.load(file)

    # Convert {Lead: {Method: Val, ...}, ...} to {Method: [Vals], ...}
    owas = {m: [0] * 48 for m in methods}
    for lead, vals in owas48.items():
        for m, owa in vals.items():
            owas[m][int(lead) - 1] = owa

    # Load sMAPE
    with open("/Users/matt/Projects/AdvancedResearchProject/results/"
              "non_ensemble_results/res/results_48_smape.txt") as file:
        smapes48 = json.load(file)

    # Convert {Lead: {Method: Val, ...}, ...} to {Method: [Vals], ...}
    smapes = {m: [0] * 48 for m in methods}
    for lead, vals in smapes48.items():
        for m, smape in vals.items():
            smapes[m][int(lead) - 1] = smape

    # Load MASE
    with open("/Users/matt/Projects/AdvancedResearchProject/results/"
              "non_ensemble_results/res/results_48_mase.txt") as file:
        mases48 = json.load(file)

    # Convert {Lead: {Method: Val, ...}, ...} to {Method: [Vals], ...}
    mases = {m: [0] * 48 for m in methods}
    for lead, vals in mases48.items():
        for m, mase in vals.items():
            mases[m][int(lead) - 1] = mase

    font = {'size': 24}
    plt.rc('font', **font)

    # Plot OWAS
    # fig, ax = plt.subplots(1, 1, figsize=(20, 15), dpi=250)
    # for i, (m, e) in enumerate(owas.items()):
    #     if m in exclude:
    #         continue
    #
    #     marker = 'o' if m in circles else 'x'
    #     ax.plot(e, label=m, marker=marker)
    #
    #
    # ax.legend(loc="best")
    # ax.axvline(x=24, linestyle=":")
    # ax.set_title("OWA against Lead Time")
    # ax.set_xticks(x_tick_locs)
    # ax.set_xticklabels(x_tick_labels)
    # plt.show()

    # Plot sMAPES - 48 hour
    legend_1_plots = []
    legend_2_plots = []
    fig, ax = plt.subplots(1, 1, figsize=(20, 15), dpi=250)
    for i, (m, e) in enumerate(smapes.items()):
        if m in exclude:
            continue

        marker = 'o' if m in circles else 'x'
        a, = ax.plot(e, label=m, marker=marker)
        ax.set_ylim(1, 5)

        if m in circles:
            legend_1_plots.append(a)
        else:
            legend_2_plots.append(a)

    # ax.legend(loc="lower right")
    ax.axvline(x=23, linestyle=":")
    ax.set_title("sMAPE against Lead Time: 48 Hour Forecast")
    ax.set_xticks(x_tick_locs)
    ax.set_xticklabels(x_tick_labels)
    ax.set_xlabel("Hour Ahead")
    ax.set_ylabel("sMAPE (%)")
    first_legend = ax.legend(handles=legend_1_plots, loc='upper left')
    ax.add_artist(first_legend)
    ax.legend(handles=legend_2_plots, loc='lower right')
    plt.show()

    # Plot sMAPES - 12 Hour
    max_lead = 12
    legend_1_plots = []
    legend_2_plots = []
    fig, ax = plt.subplots(1, 1, figsize=(20, 15), dpi=250)
    for i, (m, e) in enumerate(smapes.items()):
        if m in exclude:
            continue

        marker = 'o' if m in circles else 'x'
        a, = ax.plot(e[:max_lead], label=m, marker=marker)
        ax.set_ylim(1, 4)

        if m in circles:
            legend_1_plots.append(a)
        else:
            legend_2_plots.append(a)

    first_legend = ax.legend(handles=legend_1_plots, loc='upper left')
    ax.add_artist(first_legend)
    ax.legend(handles=legend_2_plots, loc='lower right')
    ax.set_xticks([i for i in range(max_lead)])
    ax.set_xticklabels([i + 1 for i in range(max_lead)])
    ax.set_title("sMAPE against Lead Time: " + str(max_lead) + " Hour "
                                                               "Forecast")
    ax.set_xlabel("Hour Ahead")
    ax.set_ylabel("sMAPE (%)")
    plt.show()

    print(pd.DataFrame(smapes))

    # Plot MASES
    # fig, ax = plt.subplots(1, 1, figsize=(20, 15), dpi=250)
    # for i, (m, e) in enumerate(mases.items()):
    #     if m in exclude:
    #         continue
    #
    #     marker = 'o' if m in circles else 'x'
    #     ax.plot(e, label=m, marker=marker)
    #
    # ax.legend(loc="best")
    # ax.axvline(x=24, linestyle=":")
    # ax.set_title("MASE against Lead Time")
    # ax.set_xticks(x_tick_locs)
    # ax.set_xticklabels(x_tick_labels)
    # plt.show()



# Finds the closest number higher than the desired batch size bs which
# divides the number of training examples
def calc_batch_size(n, bs):
    factors = list(set(
        reduce(
            list.__add__,
            ([i, n // i] for i in range(1, int(n ** 0.5) + 1) if n % i == 0)
        )
    ))
    return factors[np.argmin([fabs(v - bs) if v >= bs else sys.maxsize for v
                              in factors])]


# Plot actual and deseasonalised data, ACF, PACF for all season in all years
def analyse(df):
    all_data = split_data(df)

    for season in ["Winter", "Spring", "Summer", "Autumn"]:
        years = all_data[season]

        fig, axes = plt.subplots(2, 2, figsize=(20, 15), dpi=250)
        plt.suptitle(season + " - Data", y=0.99)
        for i, (year, ax) in enumerate(zip(years, axes.flatten())):
            deseason, ind, = deseasonalise(
                year["total load actual"], 168, "multiplicative"
            )

            ax.plot(year, label="Actual")
            ax.plot(deseason, label="Deseasonalised")
            ax.set_title("Year " + str(i + 1))
            ax.set_xticks([])
            ax.legend(loc="best")

            adf = adfuller(deseason, autolag='AIC')
            print("Original Data")
            print("Test Statistic (rounded) = {:.3f}".format(adf[0]))
            print("P-value (rounded) = {:.3f}".format(adf[1]))
            print("Critical values: ")
            for k, v in adf[4].items():
                print("\t{}: {:.4f} (The data is {}stationary with {}% "
                      "confidence)".format(
                    k, v, "not " if v < adf[0] else "",
                    100 - int(k[
                              :-1])))
            print()
        print()
        plt.show()

        # Plot Data ACFs
        fig, axes = plt.subplots(2, 2, figsize=(20, 15), dpi=250)
        plt.suptitle(season + " - ACFs (Actual)", y=0.99)
        for i, (year, ax) in enumerate(zip(years, axes.flatten())):
            plot_acf(year["total load actual"], ax=ax, alpha=0.05, lags=168)

        plt.show()

        # Plot Data PACFs
        fig, axes = plt.subplots(2, 2, figsize=(20, 15), dpi=250)
        plt.suptitle(season + " - PACFs (Actual)", y=0.99)
        for i, (year, ax) in enumerate(zip(years, axes.flatten())):
            plot_pacf(year["total load actual"], ax=ax, alpha=0.05, lags=168)

        plt.show()

        # Plot Deseasonalised ACFs
        fig, axes = plt.subplots(2, 2, figsize=(20, 15), dpi=250)
        plt.suptitle(season + " - ACFs (Deseasonalised)", y=0.99)
        for i, (year, ax) in enumerate(zip(years, axes.flatten())):
            deseason, _ = deseasonalise(
                year["total load actual"], 168, "multiplicative"
            )
            plot_acf(deseason, ax=ax, alpha=0.05, lags=168)

        plt.show()

        # Plot Deseasonalised PACFs
        fig, axes = plt.subplots(2, 2, figsize=(20, 15), dpi=250)
        plt.suptitle(season + " - PACFs (Deseasonalised)", y=0.99)
        for i, (year, ax) in enumerate(zip(years, axes.flatten())):
            deseason, _ = deseasonalise(
                year["total load actual"], 168, "multiplicative"
            )
            plot_pacf(deseason, ax=ax, alpha=0.05, lags=168)

        plt.show()


# Plots a sample of the weeks in the different seasons.
def plot_sample(df):
    rand_weeks = {'Winter': [[4, 1], [3, 1, 3], [4, 3, 1], [3, 1, 2]],
                  'Spring': [[1, 4, 3], [4, 2, 3], [3, 4, 1], [4, 4, 2]],
                  'Summer': [[1, 2, 3], [2, 4, 2], [3, 4, 3], [1, 2, 4]],
                  'Autumn': [[2, 4, 2], [2, 3, 3], [4, 3, 3], [2, 4, 4]]
                  }

    # Choose random week
    winter_weeks = [
        df.loc["2015-01-26 00:00:00+01:00": "2015-02-01 23:00:00+01:00"],
        df.loc["2015-02-02 00:00:00+01:00": "2015-02-08 23:00:00+01:00"],

        df.loc["2015-12-21 00:00:00+01:00": "2015-12-27 23:00:00+01:00"],
        df.loc["2016-01-04 00:00:00+01:00": "2016-01-10 23:00:00+01:00"],
        df.loc["2016-02-15 00:00:00+01:00": "2016-02-21 23:00:00+01:00"],

        df.loc["2016-12-26 00:00:00+01:00": "2017-01-01 23:00:00+01:00"],
        df.loc["2017-01-16 00:00:00+01:00": "2017-01-22 23:00:00+01:00"],
        df.loc["2017-02-06 00:00:00+01:00": "2017-02-12 23:00:00+01:00"],

        df.loc["2017-12-18 00:00:00+01:00": "2017-12-24 23:00:00+01:00"],
        df.loc["2018-01-01 00:00:00+01:00": "2018-01-07 23:00:00+01:00"],
        df.loc["2018-02-12 00:00:00+01:00": "2018-02-18 23:00:00+01:00"],
    ]

    spring_weeks = [
        df.loc["2015-03-02 00:00:00+01:00": "2015-03-08 23:00:00+01:00"],
        df.loc["2015-04-27 00:00:00+01:00": "2015-05-03 23:00:00+01:00"],
        df.loc["2015-05-18 00:00:00+01:00": "2015-05-24 23:00:00+01:00"],

        df.loc["2016-03-28 00:00:00+01:00": "2016-04-03 23:00:00+01:00"],
        df.loc["2016-04-11 00:00:00+01:00": "2016-04-17 23:00:00+01:00"],
        df.loc["2016-05-16 00:00:00+01:00": "2016-05-22 23:00:00+01:00"],

        df.loc["2017-03-20 00:00:00+01:00": "2017-03-26 23:00:00+01:00"],
        df.loc["2017-04-24 00:00:00+01:00": "2017-04-30 23:00:00+01:00"],
        df.loc["2017-05-01 00:00:00+01:00": "2017-05-07 23:00:00+01:00"],

        df.loc["2018-03-26 00:00:00+01:00": "2018-04-01 23:00:00+01:00"],
        df.loc["2018-04-23 00:00:00+01:00": "2018-04-29 23:00:00+01:00"],
        df.loc["2018-05-14 00:00:00+01:00": "2018-05-20 23:00:00+01:00"],
    ]

    summer_weeks = [
        df.loc["2015-06-01 00:00:00+01:00": "2015-06-07 23:00:00+01:00"],
        df.loc["2015-07-06 00:00:00+01:00": "2015-07-12 23:00:00+01:00"],
        df.loc["2015-08-17 00:00:00+01:00": "2015-08-23 23:00:00+01:00"],

        df.loc["2016-06-13 00:00:00+01:00": "2016-06-19 23:00:00+01:00"],
        df.loc["2016-07-25 00:00:00+01:00": "2016-07-31 23:00:00+01:00"],
        df.loc["2016-08-08 00:00:00+01:00": "2016-08-14 23:00:00+01:00"],

        df.loc["2017-06-19 00:00:00+01:00": "2017-06-25 23:00:00+01:00"],
        df.loc["2017-07-24 00:00:00+01:00": "2017-07-30 23:00:00+01:00"],
        df.loc["2017-08-21 00:00:00+01:00": "2017-08-27 23:00:00+01:00"],

        df.loc["2018-06-04 00:00:00+01:00": "2018-06-10 23:00:00+01:00"],
        df.loc["2018-07-09 00:00:00+01:00": "2018-07-15 23:00:00+01:00"],
        df.loc["2018-08-27 00:00:00+01:00": "2018-09-02 23:00:00+01:00"],
    ]

    autumn_weeks = [
        df.loc["2015-09-14 00:00:00+01:00": "2015-09-20 23:00:00+01:00"],
        df.loc["2015-10-26 00:00:00+01:00": "2015-11-01 23:00:00+01:00"],
        df.loc["2015-11-09 00:00:00+01:00": "2015-11-15 23:00:00+01:00"],

        df.loc["2016-09-12 00:00:00+01:00": "2016-09-18 23:00:00+01:00"],
        df.loc["2016-10-17 00:00:00+01:00": "2016-10-23 23:00:00+01:00"],
        df.loc["2016-11-21 00:00:00+01:00": "2016-11-27 23:00:00+01:00"],

        df.loc["2017-09-25 00:00:00+01:00": "2017-10-01 23:00:00+01:00"],
        df.loc["2017-10-16 00:00:00+01:00": "2017-10-22 23:00:00+01:00"],
        df.loc["2017-11-20 00:00:00+01:00": "2017-11-26 23:00:00+01:00"],

        df.loc["2018-09-10 00:00:00+01:00": "2018-09-16 23:00:00+01:00"],
        df.loc["2017-10-16 00:00:00+01:00": "2017-10-22 23:00:00+01:00"],
        df.loc["2018-11-26 00:00:00+01:00": "2018-12-02 23:00:00+01:00"],
    ]

    # Plot winter weeks
    fig, axes = plt.subplots(4, 3, figsize=(20, 15), dpi=250)
    axes.flatten()[0].set_ylabel("Year 1")
    for i, (ax, week) in enumerate(zip(axes.flatten()[1:], winter_weeks)):
        if i == 2:
            ax.set_title("December")
            ax.set_ylabel("Year 2")
        if i == 0:
            ax.set_title("January")
        if i == 1:
            ax.set_title("February")
        if i == 5:
            ax.set_ylabel("Year 3")
        if i == 8:
            ax.set_ylabel("Year 4")
        ax.plot(week["total load actual"], color="C0")
        ax.set_xticks([])
    plt.show()

    # Plot spring weeks
    fig, axes = plt.subplots(4, 3, figsize=(20, 15), dpi=250)
    for i, (ax, week) in enumerate(zip(axes.flatten(), spring_weeks)):
        if i == 0:
            ax.set_title("March")
            ax.set_ylabel("Year 1")
        if i == 1:
            ax.set_title("April")
        if i == 2:
            ax.set_title("May")
        if i == 3:
            ax.set_ylabel("Year 2")
        if i == 6:
            ax.set_ylabel("Year 3")
        if i == 9:
            ax.set_ylabel("Year 4")
        ax.plot(week["total load actual"], color="C1")
        ax.set_xticks([])
    plt.show()

    # Plot summer weeks
    fig, axes = plt.subplots(4, 3, figsize=(20, 15), dpi=250)
    for i, (ax, week) in enumerate(zip(axes.flatten(), summer_weeks)):
        if i == 0:
            ax.set_title("June")
            ax.set_ylabel("Year 1")
        if i == 1:
            ax.set_title("July")
        if i == 2:
            ax.set_title("August")
        if i == 3:
            ax.set_ylabel("Year 2")
        if i == 6:
            ax.set_ylabel("Year 3")
        if i == 9:
            ax.set_ylabel("Year 4")
        ax.plot(week["total load actual"], color="C2")
        ax.set_xticks([])
    plt.show()

    # Plot autumn weeks
    fig, axes = plt.subplots(4, 3, figsize=(20, 15), dpi=250)
    for i, (ax, week) in enumerate(zip(axes.flatten(), autumn_weeks)):
        if i == 0:
            ax.set_title("September")
            ax.set_ylabel("Year 1")
        if i == 1:
            ax.set_title("October")
        if i == 2:
            ax.set_title("November")
        if i == 3:
            ax.set_ylabel("Year 2")
        if i == 6:
            ax.set_ylabel("Year 3")
        if i == 9:
            ax.set_ylabel("Year 4")
        ax.plot(week["total load actual"], color="C3")
        ax.set_xticks([])
    plt.show()


# Plot an example week from each season, and plot the
def typical_plot(df):
    # Typical weeks
    font = {'size': 20}
    plt.rc('font', **font)
    seas_dict = {0: "Winter", 1: "Spring", 2: "Summer", 3: "Autumn"}
    years = [df.loc["2015-12-01 00:00:00+01:00":"2016-02-29 23:00:00+01:00"],
             df.loc["2015-03-01 00:00:00+01:00":"2015-05-31 23:00:00+01:00"],
             df.loc["2018-06-01 00:00:00+01:00":"2018-08-31 23:00:00+01:00"],
             df.loc["2017-09-01 00:00:00+01:00":"2017-11-30 23:00:00+01:00"]
             ]
    weeks = [df.loc["2017-01-16 00:00:00+01:00": "2017-01-22 23:00:00+01:00"],
             df.loc["2016-05-16 00:00:00+01:00": "2016-05-22 23:00:00+01:00"],
             df.loc["2017-07-24 00:00:00+01:00": "2017-07-30 23:00:00+01:00"],
             df.loc["2017-10-16 00:00:00+01:00": "2017-10-22 23:00:00+01:00"]]

    # Plot data
    fig, axes = plt.subplots(2, 2, figsize=(20, 15), dpi=250)
    for i, (week, ax) in enumerate(zip(weeks, axes.flatten())):
        ax.plot(week)
        ax.set_title(seas_dict[i])
        ax.set_xticks([])
    plt.show()

    # Plot ACF
    fig, axes = plt.subplots(2, 2, figsize=(20, 15), dpi=250)
    for i, (year, ax) in enumerate(zip(years, axes.flatten())):
        plot_acf(year, ax=ax, alpha=0.05, lags=168)
        ax.set_title(seas_dict[i])
    plt.show()

    # Plot PACF
    fig, axes = plt.subplots(2, 2, figsize=(20, 15), dpi=250)
    for i, (year, ax) in enumerate(zip(years, axes.flatten())):
        plot_pacf(year, ax=ax, alpha=0.05, lags=168)
        ax.set_title(seas_dict[i])
    plt.show()


# Plot an exaple test/training data split, for two train periods
def train_test_split(df):
    font = {'size': 20}
    plt.rc('font', **font)
    fig, axes = plt.subplots(3, 1, figsize=(20, 15), dpi=250)
    axes[0].plot(
        df.loc["2015-01-01 00:00:00+01:00":"2015-02-13 23:00:00+01:00"],
        label="Training"
    )
    axes[0].plot(
        df.loc["2015-02-14 00:00:00+01:00":"2015-02-20 23:00:00+01:00"],
        label="Validation", color="C2"
    )
    axes[0].plot(
        df.loc["2015-02-21 00:00:00+01:00":"2015-02-28 23:00:00+01:00"],
        label="Test", color="C1"
    )

    axes[1].plot(
        df.loc["2015-01-01 00:00:00+01:00":"2015-02-20 23:00:00+01:00"],
        label="Test 1 - Training"
    )
    axes[1].plot(
        df.loc["2015-02-21 00:00:00+01:00":"2015-02-22 23:00:00+01:00"],
        label="Test 1 - Test"
    )
    axes[1].plot(
        df.loc["2015-02-23 00:00:00+01:00":"2015-02-28 23:00:00+01:00"],
        label="Test 1 - Unused",
        color="#7f7f7f"
    )

    axes[2].plot(
        df.loc["2015-01-01 00:00:00+01:00":"2015-02-24 23:00:00+01:00"],
        label="Test 5 - Training"
    )
    axes[2].plot(
        df.loc["2015-02-25 00:00:00+01:00":"2015-02-26 23:00:00+01:00"],
        label="Test 5 - Test"
    )
    axes[2].plot(
        df.loc["2015-02-27 00:00:00+01:00":"2015-02-28 23:00:00+01:00"],
        label="Test 5 - Unused",
        color="#7f7f7f"
    )

    axes[0].set_title("Year 1 - Winter - Training/Validation/Test Split")
    axes[1].set_title("Year 1 - Winter - Test 1")
    axes[2].set_title("Year 1 - Winter - Test 5")
    axes[0].set_xticks([])
    axes[1].set_xticks([])
    axes[2].set_xticks([])
    axes[0].legend(loc="best")
    axes[1].legend(loc="best")
    axes[2].legend(loc="best")
    plt.show()


def plot_a_season(df, season):
    split = split_data(df)

    for y in split[season]:
        # Plot first quarter
        fig, ax = plt.subplots(1, 1, figsize=(20, 15), dpi=250)
        ax.plot(y["total load actual"][:int(len(y["total load actual"])/4.0)])
        plt.show()

        # Plot second quarter
        fig, ax = plt.subplots(1, 1, figsize=(20, 15), dpi=250)
        ax.plot(y["total load actual"][int(len(y["total load actual"]) /
                                           4.0):int(len(y["total load "
                                                         "actual"]) / 2.0)])
        plt.show()

        # Plot third quarter
        fig, ax = plt.subplots(1, 1, figsize=(20, 15), dpi=250)
        ax.plot(y["total load actual"][int(len(y["total load actual"]) /
                                           2.0):int(3.0 * len(y["total load "
                                                          "actual"]) / 4.0)])
        plt.show()

        # Plot fourth quarter
        fig, ax = plt.subplots(1, 1, figsize=(20, 15), dpi=250)
        ax.plot(y["total load actual"][int(3.0 * len(y["total load actual"])
                                           / 4):])
        plt.show()


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


# Plot the deseasonalised data for 24- and 168-seasonality
def deseason(df):
    year = df.loc["2015-01-01 00:00:00+01:00":"2015-02-28 23:00:00+01:00"]
    year_24, _ = deseasonalise(
        year["total load actual"], 24, "multiplicative"
    )
    year_168, _ = deseasonalise(
        year["total load actual"], 168, "multiplicative"
    )

    # Create figure
    font = {'size': 20}
    plt.rc('font', **font)
    fig = plt.figure(figsize=(20, 15), dpi=250)
    gs = fig.add_gridspec(2, 2)
    ax_1 = fig.add_subplot(gs[0, :])
    ax_2 = fig.add_subplot(gs[1, 0])
    ax_3 = fig.add_subplot(gs[1, 1])

    # Plot data
    ax_1.plot(year)
    ax_2.plot(year_24)
    ax_3.plot(year_168)

    # Add weekend highlighting
    ax_1.axvspan("2015-01-03 00:00:00+01:00",
                 "2015-01-04 23:00:00+01:00", alpha=0.1)
    ax_1.axvspan("2015-01-10 00:00:00+01:00",
                 "2015-01-11 23:00:00+01:00", alpha=0.1)
    ax_1.axvspan("2015-01-17 00:00:00+01:00",
                 "2015-01-18 23:00:00+01:00", alpha=0.1)
    ax_1.axvspan("2015-01-24 00:00:00+01:00",
                 "2015-01-25 23:00:00+01:00", alpha=0.1)
    ax_1.axvspan("2015-01-31 00:00:00+01:00",
                 "2015-02-01 23:00:00+01:00", alpha=0.1)
    ax_1.axvspan("2015-02-07 00:00:00+01:00",
                 "2015-02-08 23:00:00+01:00", alpha=0.1)
    ax_1.axvspan("2015-02-14 00:00:00+01:00",
                 "2015-02-15 23:00:00+01:00", alpha=0.1)
    ax_1.axvspan("2015-02-21 00:00:00+01:00",
                 "2015-02-22 23:00:00+01:00", alpha=0.1)
    ax_1.axvspan("2015-02-28 00:00:00+01:00",
                 "2015-02-28 23:00:00+01:00", alpha=0.1)

    ax_2.axvspan("2015-01-03 00:00:00+01:00",
                 "2015-01-04 23:00:00+01:00", alpha=0.1)
    ax_2.axvspan("2015-01-10 00:00:00+01:00",
                 "2015-01-11 23:00:00+01:00", alpha=0.1)
    ax_2.axvspan("2015-01-17 00:00:00+01:00",
                 "2015-01-18 23:00:00+01:00", alpha=0.1)
    ax_2.axvspan("2015-01-24 00:00:00+01:00",
                 "2015-01-25 23:00:00+01:00", alpha=0.1)
    ax_2.axvspan("2015-01-31 00:00:00+01:00",
                 "2015-02-01 23:00:00+01:00", alpha=0.1)
    ax_2.axvspan("2015-02-07 00:00:00+01:00",
                 "2015-02-08 23:00:00+01:00", alpha=0.1)
    ax_2.axvspan("2015-02-14 00:00:00+01:00",
                 "2015-02-15 23:00:00+01:00", alpha=0.1)
    ax_2.axvspan("2015-02-21 00:00:00+01:00",
                 "2015-02-22 23:00:00+01:00", alpha=0.1)
    ax_2.axvspan("2015-02-28 00:00:00+01:00",
                 "2015-02-28 23:00:00+01:00", alpha=0.1)

    ax_3.axvspan("2015-01-03 00:00:00+01:00",
                 "2015-01-04 23:00:00+01:00", alpha=0.1)
    ax_3.axvspan("2015-01-10 00:00:00+01:00",
                 "2015-01-11 23:00:00+01:00", alpha=0.1)
    ax_3.axvspan("2015-01-17 00:00:00+01:00",
                 "2015-01-18 23:00:00+01:00", alpha=0.1)
    ax_3.axvspan("2015-01-24 00:00:00+01:00",
                 "2015-01-25 23:00:00+01:00", alpha=0.1)
    ax_3.axvspan("2015-01-31 00:00:00+01:00",
                 "2015-02-01 23:00:00+01:00", alpha=0.1)
    ax_3.axvspan("2015-02-07 00:00:00+01:00",
                 "2015-02-08 23:00:00+01:00", alpha=0.1)
    ax_3.axvspan("2015-02-14 00:00:00+01:00",
                 "2015-02-15 23:00:00+01:00", alpha=0.1)
    ax_3.axvspan("2015-02-21 00:00:00+01:00",
                 "2015-02-22 23:00:00+01:00", alpha=0.1)
    ax_3.axvspan("2015-02-28 00:00:00+01:00",
                 "2015-02-28 23:00:00+01:00", alpha=0.1)

    # Add titles
    ax_1.set_xticks([])
    ax_1.set_title("Year 1 - Winter - Actual")
    ax_2.set_xticks([])
    ax_2.set_title("Hourly Seasonality Removed")
    ax_3.set_xticks([])
    ax_3.set_title("Weekly Seasonality Removed")
    plt.show()

    # Plot the ACF and PACF of the deseasonalised data
    fig, axes = plt.subplots(2, 1, figsize=(20, 15), dpi=250)
    plot_acf(year_168, axes[0], lags=168)
    plot_pacf(year_168, axes[1], lags=168)
    axes[0].set_title("Year 1 - Winter - Deseasonalised ACF")
    axes[1].set_title("Year 1 - Winter - Deseasonalised PACF")
    plt.show()


def holt_winters_test(df):
    data = split_data(df)["Summer"][2]["total load actual"]
    train_data = data[:-48]
    test_data = data[-48:]

    # [alpha, beta, gamma, l0, b0, phi, s0,.., s_(m - 1)]
    _, indices = deseasonalise(train_data, 168, "multiplicative")
    init_params = [0.25, 0.75, train_data[0]]
    init_params.extend(indices)

    fitted_model = ExponentialSmoothing(
        train_data, seasonal_periods=168, seasonal="mul"
    ).fit(use_basinhopping=True, start_params=init_params)
    init_prediction = fitted_model.predict(0, len(train_data) + 48 - 1)
    params = fitted_model.params
    print(params)

    fitted_model = ExponentialSmoothing(
        train_data, seasonal_periods=168, seasonal="mul"
    ).fit(use_basinhopping=True)
    prediction = fitted_model.predict(0, len(train_data) + 48 - 1)
    params = fitted_model.params
    print(params)

    fig, ax = plt.subplots(1, 1, figsize=(20, 15), dpi=250)
    ax.plot(test_data, label="Actual Data")
    ax.plot(prediction[-48:], label="Non initialised")
    ax.plot(init_prediction[-48:], label="Initialised")
    ax.legend(loc="best")
    plt.show()



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
