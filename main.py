import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm

import os.path, json, sys

from pandas.plotting import register_matplotlib_converters
from functools import reduce
from timeit import default_timer as timer
from math import fabs
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from stats import arima, exponential_smoothing, naive, theta, errors, helpers
from hybrid import hybrid


def load_data(filename, mult_ts):
    if mult_ts:
        df = pd.read_csv(
            filename, parse_dates=["time"], infer_datetime_format=True
        )

        # Drop the forecast columns
        df.drop(
            columns=["forecast solar day ahead",
                     "forecast wind offshore eday ahead",
                     "forecast wind onshore day ahead",
                     "total load forecast",
                     "price day ahead"],
            inplace=True
        )

        # TODO - REMOVE AND PROPERLY FIX THE ZERO VALUES
        df = df[["time", "total load actual", "price actual"]]

        # Drop the columns whose values are all either 0 or missing
        return df.loc[:, (pd.isna(df) == (df == 0)).any(axis=0)]

    else:
        return pd.read_csv(
            filename, parse_dates=["time"],
            usecols=["time", "total load actual"], infer_datetime_format=True
        )


def main():
    stats_test(int(sys.argv[1]), int(sys.argv[2]))
    # file_path = os.path.abspath(os.path.dirname(__file__))
    # data_path = os.path.join(file_path, "data/spain/energy_dataset.csv")
    # df = load_data(data_path, False)
    # df = df.set_index('time').asfreq('H')
    # df.interpolate(inplace=True)
    # identify_arima(df, False)
    # hybrid.es_rnn(df)

    # ---------------------- LOAD MULTIPLE TIME SERIES ----------------------

    # ---------------------- DATA PARAMETERS ------------------------
    # batch_size = calc_batch_size(
    #     train_hours - window_size - output_size + 1, 64
    # )

    # ---------------------- LOAD AND PREPROCESS ------------------------
    # print("Any missing values?", data.isnull().values.any())
    # print("Any zero values?", data.eq(0).any())
    # data.rename(
    #     columns={"generation other": "generation other on-renewable"},
    #     inplace=True
    # )


# Finds the closest number higher than the desired batch size bs which
# divides the number of training examples
def calc_batch_size(n, bs):
    factors = list(set(
        reduce(
            list.__add__,
            ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)
        )
    ))
    return factors[np.argmin([fabs(v - bs) if v >= bs else sys.maxsize for v
                              in factors])]


# Plot actual and deseasonalised data, ACF, PACF for all season in all years
def analyse(df):
    all_data = helpers.split_data(df)

    for season in ["Winter", "Spring", "Summer", "Autumn"]:
        years = all_data[season]

        fig, axes = plt.subplots(2, 2, figsize=(20, 15), dpi=250)
        plt.suptitle(season + " - Data", y=0.99)
        for i, (year, ax) in enumerate(zip(years, axes.flatten())):
            deseason, ind, = helpers.deseasonalise(
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
            deseason, _ = helpers.deseasonalise(
                year["total load actual"], 168, "multiplicative"
            )
            plot_acf(deseason, ax=ax, alpha=0.05, lags=168)

        plt.show()

        # Plot Deseasonalised PACFs
        fig, axes = plt.subplots(2, 2, figsize=(20, 15), dpi=250)
        plt.suptitle(season + " - PACFs (Deseasonalised)", y=0.99)
        for i, (year, ax) in enumerate(zip(years, axes.flatten())):
            deseason, _ = helpers.deseasonalise(
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
    fig, axes = plt.subplots(4, 3, figsize=(12.8, 9.6), dpi=250)
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
        ax.plot(week, color="C0")
        ax.set_xticks([])
    plt.show()

    # Plot spring weeks
    fig, axes = plt.subplots(4, 3, figsize=(12.8, 9.6), dpi=250)
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
        ax.plot(week, color="C1")
        ax.set_xticks([])
    plt.show()

    # Plot summer weeks
    fig, axes = plt.subplots(4, 3, figsize=(12.8, 9.6), dpi=250)
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
        ax.plot(week, color="C2")
        ax.set_xticks([])
    plt.show()

    # Plot autumn weeks
    fig, axes = plt.subplots(4, 3, figsize=(12.8, 9.6), dpi=250)
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
        ax.plot(week, color="C3")
        ax.set_xticks([])
    plt.show()


# Plot an example week from each season, and plot the
def typical_plot(df):
    # Typical weeks TODO - REDO AS YEARS ARE WRONG
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
    fig, axes = plt.subplots(2, 1, figsize=(20, 15), dpi=250)
    axes[0].plot(
        df.loc["2015-01-01 00:00:00+01:00":"2015-02-20 23:00:00+01:00"],
        label="Test 1 - Training"
    )
    axes[0].plot(
        df.loc["2015-02-21 00:00:00+01:00":"2015-02-22 23:00:00+01:00"],
        label="Test 1 - Test"
    )
    axes[0].plot(
        df.loc["2015-02-23 00:00:00+01:00":"2015-02-28 23:00:00+01:00"],
        label="Test 1 - Unused",
        color="#7f7f7f"
    )

    axes[1].plot(
        df.loc["2015-01-01 00:00:00+01:00":"2015-02-24 23:00:00+01:00"],
        label="Test 5 - Training"
    )
    axes[1].plot(
        df.loc["2015-02-25 00:00:00+01:00":"2015-02-26 23:00:00+01:00"],
        label="Test 5 - Test"
    )
    axes[1].plot(
        df.loc["2015-02-27 00:00:00+01:00":"2015-02-28 23:00:00+01:00"],
        label="Test 5 - Unused",
        color="#7f7f7f"
    )

    axes[0].set_title("Year 1 - Winter - Test 1")
    axes[1].set_title("Year 1 - Winter - Test 5")
    axes[0].set_xticks([])
    axes[1].set_xticks([])
    axes[0].legend(loc="best")
    axes[1].legend(loc="best")
    plt.show()


# Identify the correct ARIMA order for each season
def identify_arima(df, plot):
    # Optionally pass in a season as command line argument. If we do so,
    # only that season
    seas_in = -1 if len(sys.argv) == 1 else str(sys.argv[1])
    all_data = helpers.split_data(df)

    for season in ["Winter", "Spring", "Summer", "Autumn"]:
        # If we passed in our own season, then skip all other seasons
        if seas_in != -1 and seas_in != season:
            continue

        years = all_data[season]
        years = [years[i]["total load actual"] for i in range(4)]
        resids = []

        for yn, y in enumerate(years):

            y, ind = helpers.deseasonalise(y, 168, "multiplicative")
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
    all_data = helpers.split_data(df)

    for season in ["Winter", "Spring", "Summer", "Autumn"]:
        years = all_data[season]
        years = [years[i]["total load actual"] for i in range(1)]
        resids = []

        years = [years[0]]

        for yn, y in enumerate(years):
            start = timer()
            y, ind = helpers.deseasonalise(y, 168, "multiplicative")
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
    year_24, _ = helpers.deseasonalise(
        year["total load actual"], 24, "multiplicative"
    )
    year_168, _ = helpers.deseasonalise(
        year["total load actual"], 168, "multiplicative"
    )

    # Create figure
    fig = plt.figure(figsize=(12.8, 9.6), dpi=250)
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
    ax_2.set_title("Year 1 - Winter - Hourly Seasonality Removed")
    ax_3.set_xticks([])
    ax_3.set_title("Year 1 - Winter - Weekly Seasonality Removed")
    plt.show()

    # Plot the ACF and PACF of the deseasonalised data
    fig, axes = plt.subplots(2, 1, figsize=(20, 15), dpi=250)
    plot_acf(year_168, axes[0], lags=168)
    plot_pacf(year_168, axes[1], lags=168)
    axes[0].set_title("Year 1 - Winter - Deseasonalised ACF")
    axes[1].set_title("Year 1 - Winter - Deseasonalised PACF")
    plt.show()


# The method must accept two parameters: season no. (1 - 4) and method no.
def stats_test(season_no, model_no):
    # Load data
    file_path = os.path.abspath(os.path.dirname(__file__))
    data_path = os.path.join(file_path, "data/spain/energy_dataset.csv")
    df = pd.read_csv(data_path, parse_dates=["time"],
                     usecols=["time", "total load actual"],
                     infer_datetime_format=True)
    df = df.set_index('time').asfreq('H')
    df.interpolate(inplace=True)

    # Testing hyper-parameters
    seasonality = 168
    forecast_length = 48

    # Model No.: [Function, Name, Deseasonalise?, Additional Parameters,
    # Return Parameters, Number of Repetitions]
    test_dict = {
        1: [naive.naive_1, 'Naive1', False, None, False, 10],

        2: [naive.naive_2, 'Naive2', True, None, False, 10],

        3: [naive.naive_s, 'NaiveS', False, [24], False, 10],

        4: [exponential_smoothing.ses, 'SES', True, None, True, 10],

        5: [exponential_smoothing.holt, 'Holt', True, None, True, 10],

        6: [exponential_smoothing.damped, 'Damped', True, None, True, 10],

        7: [exponential_smoothing.holt_winters, 'Holt-Winters', False, [24],
            True, 10],

        8: [exponential_smoothing.comb, 'Comb', True, None, False, 10],

        9: [arima.arima, 'ARIMA', True, "-- SEE arima_orders --", True, 10],

        10: [arima.sarima, 'SARIMA', False,
             [(2, 0, 1), (2, 0, 1, seasonality)], True, 1],

        11: [arima.auto, 'Auto', False, [24], True, 1],

        12: [theta.theta, 'Theta', True, None, True, 10]
    }

    # Optimum ARIMA Parameters (automatically checked, using the
    # identify_arima function)
    arima_orders = {
        1: [[2, 0, 0], [2, 0, 0], [1, 0, 2], [2, 0, 2]],
        2: [[2, 0, 0], [2, 0, 0], [2, 0, 2], [2, 0, 2]],
        3: [[1, 0, 1], [2, 0, 2], [2, 0, 2], [2, 0, 2]],
        4: [[2, 0, 1], [2, 0, 2], [2, 0, 2], [2, 0, 2]],
    }

    seas_dict = {1: "Spring", 2: "Summer", 3: "Autumn", 4: "Winter"}

    # Get the parameters for the model
    model_func, model_name, deseasonalise, params, ret_params, num_reps = \
        test_dict[model_no]
    error_pairs = [("sMAPE", errors.sMAPE), ("RMSE", errors.RMSE),
                   ("MASE", errors.MASE), ("MAE", errors.MAE)]

    # Build empty data structures to hol results, naive results, forecasts and
    # fitted parameters
    results = {e: {r: {y: {t: [0] * forecast_length for t in range(1, 8)
                           } for y in range(1, 5)
                       } for r in range(1, num_reps + 1)
                   } for e in list(zip(*error_pairs))[0] + tuple(["OWA"])
               }

    n_results = {e: {r: {y: {t: [0] * forecast_length for t in range(1, 8)
                             } for y in range(1, 5)
                         } for r in range(1, num_reps + 1)
                     } for e in list(zip(*error_pairs))[0] + tuple(["OWA"])
                 }

    forecasts = {y: {r: {t: [] for t in range(1, 8)
                         } for r in range(1, num_reps + 1)
                     } for y in range(1, 5)
                 }

    final_params = {y: [] for y in range(1, 5)}

    all_data = helpers.split_data(df)
    years = all_data[seas_dict[season_no]]
    years = [years[i]["total load actual"] for i in range(4)]

    start = timer()
    for y_index, y in enumerate(years):  # Years

        # Specify correct ARIMA parameters
        if model_no == 9:
            params = arima_orders[season_no][y_index]

        for t in range(8, 1, -1):  # Train times
            # Get training data, deseasonalise if necessary
            train = y[:-(t * 24)]
            train_d, indices = helpers.deseasonalise(
                train, seasonality, "multiplicative"
            )
            if deseasonalise:
                train = train_d

            for r in range(1, num_reps + 1):  # Repetitions
                # Get test data. Change y[:-0] to y[:None].
                start = -(t * 24)
                end = -(t * 24 - forecast_length) if t > 2 else None
                test = y[start:end]

                # Fit model and forecast, with additional params if needed
                if params is not None:
                    forec_results = model_func(train, forecast_length, *params)
                else:
                    forec_results = model_func(train, forecast_length)

                # Split results into fit-forecast and parameters if needed
                if ret_params:
                    fit_forecast, fit_params = forec_results
                else:
                    fit_forecast = forec_results

                # Reseasonalise if necessary
                if deseasonalise:
                    fit_forecast = helpers.reseasonalise(
                        fit_forecast, indices, "multiplicative"
                    )

                # Generate naïve forecast
                naive_fit_forecast = helpers.reseasonalise(
                    naive.naive_2(train_d, forecast_length),
                    indices,
                    "multiplicative"
                )

                # Select only the forecast, not the fitted values
                forecast = fit_forecast[-forecast_length:]
                naive_forecast = naive_fit_forecast[-forecast_length:]

                # Loop through the error functions
                for e_name, e_func in error_pairs:
                    # Loop through the lead times
                    for l in range(1, forecast_length + 1):
                        if e_name == "MASE":
                            end = None if (t == 2 and l == 48) else -(t * 24
                                                                      - l)
                            error = e_func(forecast[:l], y[:end],
                                           seasonality, l)
                            n_error = e_func(naive_forecast[:l], y[:end],
                                             seasonality, l)
                        else:
                            error = e_func(forecast[:l], test[:l])
                            n_error = e_func(naive_forecast[:l], test[:l])

                        # Save error results for all lead times
                        results[e_name][r][y_index + 1][t - 1][l - 1] = error
                        n_results[e_name][r][y_index + 1][t - 1][l - 1] = \
                            n_error

                # Save 48 hour forecast
                forecasts[y_index + 1][r][t - 1] = forecast.to_list()

                # Save model params only for final repetition and train time
                if r == num_reps and t == 2 and ret_params:
                    final_params[y_index + 1] = fit_params
            end = timer()
        print(np.around(end - start, decimals=3), "seconds")

    # Calculate OWA for all forecasts
    for r in range(1, num_reps + 1):
        for y in range(1, 5):
            for t in range(1, 8):
                for l in range(0, forecast_length):
                    results["OWA"][r][y][t][l] = errors.OWA(
                        n_results["sMAPE"][r][y][t][l],
                        n_results["MASE"][r][y][t][l],
                        results["sMAPE"][r][y][t][l],
                        results["MASE"][r][y][t][l],
                    )

    # Average 48 hour forecast results
    all_res = []
    for r in range(1, num_reps + 1):
        for y in range(1, 5):
            for t in range(1, 8):
                all_res.append(results["OWA"][r][y][t][forecast_length - 1])

    mean = np.around(np.mean(all_res), decimals=3)
    std = np.around(np.std(all_res), decimals=3)

    # Save averaged 48 forecast results
    file_path = os.path.abspath(os.path.dirname(__file__))
    res_path = os.path.join(file_path, "results/results_1.txt")
    with open(res_path) as file:
        results_1 = json.load(file)

    results_1[seas_dict[season_no]][model_name] = [mean, std]

    with open(res_path, "w") as file:
        json.dump(results_1, file)

    # Average the lead time results
    all_res = {l: [] for l in range(1, forecast_length + 1)}
    for r in range(1, num_reps + 1):
        for y in range(1, 5):
            for t in range(1, 8):
                for l in range(1, forecast_length + 1):
                    all_res[l].append(results["OWA"][r][y][t][l - 1])

    for l in all_res.keys():
        all_res[l] = np.around(np.mean(all_res[l]), decimals=3)

    # Save the lead time results
    res_path = os.path.join(file_path, "results/results_48_seasons.txt")
    with open(res_path) as file:
        results_48 = json.load(file)

    for l in all_res.keys():
        results_48[str(l)][model_name][season_no - 1] = all_res[l]

    with open(res_path, "w") as file:
        json.dump(results_48, file)

    # Save the forecasts and results
    res_filename = seas_dict[season_no] + "_" + model_name + "_results.txt"
    forec_filename = seas_dict[season_no] + "_" + model_name + "_forecasts.txt"
    res_path = os.path.join(file_path, "results/" + res_filename)
    forec_path = os.path.join(file_path, "results/" + forec_filename)

    with open(res_path, "w") as file:
        json.dump(results, file)
    with open(forec_path, "w") as file:
        json.dump(forecasts, file)

    # Save the parameters (if model returns parameters)
    if ret_params:
        param_path = os.path.join(file_path, "results/params.txt")
        with open(param_path) as file:
            saved_params = json.load(file)

        for y in range(1, 5):
            saved_params[model_name][str(season_no)][str(y)] = final_params[
                y]

        with open(param_path, "w") as file:
            json.dump(saved_params, file)


# Delete all files in results/, then run this function to regenerate
def reset_results_files():
    file_path = os.path.abspath(os.path.dirname(__file__))

    # Parameters
    param_path = os.path.join(file_path, "results/params.txt")
    params = {m: {s: {y: [] for y in range(1, 5)} for s in range(1, 5)} for
              m in ["SES", "Holt", "Damped", "Holt-Winters", "ARIMA", "SARIMA",
                    "Auto", "Theta"]}
    with open(param_path, "w") as f:
        json.dump(params, f)

    # Results 48 (all seasons)
    res48s_path = os.path.join(file_path, "results/results_48_seasons.txt")
    methods = ["Naive1", "Naive2", "NaiveS", "SES", "Holt", "Damped",
               "Holt-Winters", "Comb", "ARIMA", "SARIMA", "Auto", "Theta"]
    res48s = {l: {m: [0, 0, 0, 0] for m in methods} for l in range(1, 49)}
    with open(res48s_path, "w") as f:
        json.dump(res48s, f)

    # Results 48
    res48_path = os.path.join(file_path, "results/results_48.txt")
    res48 = {l: {m: 0 for m in methods} for l in range(1, 49)}
    with open(res48_path, "w") as f:
        json.dump(res48, f)

    # Results 1
    res1_path = os.path.join(file_path, "results/results_1.txt")
    seasons = ["Spring", "Summer", "Winter", "Autumn", "Average"]
    res1 = {s: {m: [0, 0] for m in methods} for s in seasons}
    with open(res1_path, "w") as f:
        json.dump(res1, f)


def test(data, seasonality, test_hours, methods, names, multiple):
    forecast_methods = methods
    forecast_names = names
    # forecast_names = ['naive1', 'naiveS', 'naive2', 'ses', 'holt', 'damped',
    #                   'theta', 'comb', 'sarima']
    error_measures = [errors.sMAPE, errors.RMSE, errors.MASE, errors.MAE]
    error_names = ["sMAPE", "RMSE", "MASE", "MAE"]

    all_train_days = [7, 10]

    if multiple:
        results = {t: {f: {e: [0] * test_hours for e in error_names}
                       for f in forecast_names} for t in all_train_days}
        # Loop through the number of training days used
        for t in all_train_days:
            error_vals = {f: {e: [[] for i in range(test_hours)] for e in
                              error_names} for f in forecast_names}
            train_hours = t * seasonality

            # Loop through the entire time series
            for o in range(len(data) - train_hours - test_hours + 1):
                data_subset = data[o: o + train_hours + test_hours]
                helpers.indices_adjust(
                    data_subset, train_hours, test_hours, "multiplicative"
                )

                # Loop through each forecasting function
                for f_name, f in zip(forecast_names, forecast_methods):
                    forecast = f(
                        data_subset, train_hours, test_hours, False
                    )[train_hours: train_hours + test_hours]

                    # Loop through the error functions
                    for e_name, e in zip(error_names, error_measures):

                        # Loop through the forecast horizon
                        for i in range(1, test_hours + 1):
                            if e_name == "MASE":
                                end = o + train_hours + i
                                error = e(
                                    forecast[:i],
                                    data['total load actual'][o: end],
                                    seasonality,
                                    i
                                )
                            else:
                                start = o + train_hours
                                end = o + train_hours + i
                                error = e(
                                    forecast[:i],
                                    data['total load actual'][start: end]
                                )
                            error_vals[f_name][e_name][i - 1].append(error)

            # Calculate the average error for the given training length,
            # forecast method and error function
            for f, v in error_vals.items():
                for e, w in v.items():
                    for i in range(1, test_hours + 1):
                        results[t][f][e][i - 1] = np.mean(w[i - 1])

        # Calculate the OWA for each training length/forecast method/
        for t in all_train_days:
            for f in forecast_names:
                results[t][f]["OWA"] = [0] * test_hours
                for i in range(1, test_hours + 1):
                    results[t][f]["OWA"][i - 1] = errors.OWA(
                        results[t]["naive2"]["sMAPE"][i - 1],
                        results[t]["naive2"]["MASE"][i - 1],
                        results[t][f]["sMAPE"][i - 1],
                        results[t][f]["MASE"][i - 1]
                    )
    else:

        results = {t: {f: {e: 0 for e in error_names} for f in forecast_names}
                   for t in all_train_days}

        # Loop through the number of training days used
        for t in all_train_days:
            error_vals = {f: {e: [] for e in error_names} for f in
                          forecast_names}
            train_hours = t * seasonality

            # Loop through the entire time series
            for o in range(len(data) - train_hours - test_hours + 1):
                data_subset = data[o: o + train_hours + test_hours]
                helpers.indices_adjust(
                    data_subset, train_hours, test_hours, "multiplicative"
                )

                # Loop through each forecasting function
                for f_name, f in zip(forecast_names, forecast_methods):
                    forecast = f(
                        data_subset, train_hours, test_hours, False
                    )[train_hours: train_hours + test_hours]

                    for e_name, e in zip(error_names, error_measures):
                        if e_name == "MASE":
                            end = o + train_hours + test_hours
                            error = e(
                                forecast,
                                data['total load actual'][o: end],
                                seasonality,
                                test_hours
                            )
                        else:
                            start = o + train_hours
                            end = o + train_hours + test_hours
                            error = e(
                                forecast,
                                data['total load actual'][start: end]
                            )
                        error_vals[f_name][e_name].append(error)

            # Calculate the average error for the given training length,
            # forecast method and error function
            for f, v in error_vals.items():
                for e, w in v.items():
                    results[t][f][e] = np.mean(w)

        # Calculate the OWA for each training length/forecast method/
        for t in all_train_days:
            for f in forecast_names:
                results[t][f]["OWA"] = errors.OWA(
                    results[t]["naive2"]["sMAPE"],
                    results[t]["naive2"]["MASE"],
                    results[t][f]["sMAPE"],
                    results[t][f]["MASE"]
                )

    return results


def write_results(results, test_no, multiple):
    file_path = os.path.abspath(os.path.dirname(__file__))
    all_results = pd.DataFrame()

    for t, v in results.items():
        res = pd.DataFrame(v)

        if multiple:
            res_path = os.path.join(
                file_path,
                "run/results/" + test_no + "_" + str(t) + "m.csv"
            )
            res = res.apply(pd.Series.explode)
        else:
            res_path = os.path.join(
                file_path,
                "run/results/" + test_no + "_" + str(t) + ".csv"
            )

        print(res)

        res.to_csv(res_path)
        res.reset_index(inplace=True)
        res.rename(columns={"index": "Error"}, inplace=True)
        res["Train Time"] = t
        all_results = pd.concat([all_results, res])

    all_results.set_index("Train Time", inplace=True)

    print(all_results)

    all_res_path = os.path.join(
        file_path,
        "run/results/all_results" + test_no + ".csv"
    )
    all_results.to_csv(all_res_path)


#
# if __name__ == "main":
#     # matplotlib.style.use('seaborn-deep')
#     register_matplotlib_converters()
#     # pd.options.mode.chained_assignment = None TURN OFF CHAINED ASSIGNMENTS
#     # WARNING - possibly use in the future if I run into that message again
#     main()
# Entry point when using PyCharm - REMOVE
# matplotlib.style.use('seaborn-deep')


register_matplotlib_converters()
pd.options.mode.chained_assignment = None
pd.set_option('display.max_columns', 500)  # Shows all columns
pd.set_option('display.expand_frame_repr', False)  # Prevents line break
plt.style.use('seaborn-deep')
main()
