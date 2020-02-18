import pandas as pd
from pandas.plotting import register_matplotlib_converters
from functools import reduce
import matplotlib.pyplot as plt
import numpy as np
from math import fabs
import os.path
import sys

import stats.plots
from stats import ses, helpers, naive1, naive2, naiveS, holt, holtDamped, \
    holtWinters, autoSarima, sarima, naive2_adjusted, ses_adjusted, \
    holt_adjusted, holtDamped_adjusted, comb, comb_adjusted, theta, errors, \
    plots
from ml import lstm_adjusted, basic_lstm, lstm_48, lstm_48_multiple, \
    drnn_48, drnn_48_multiple


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

        # Drop the columns whose values are all either 0 or missing
        return df.loc[:, (pd.isna(df) == (df == 0)).any(axis=0)]

    else:
        return pd.read_csv(
            filename, parse_dates=["time"],
            usecols=["time", "total load actual"], infer_datetime_format=True
        )


def main():
    # ---------------------- LOAD MULTIPLE TIME SERIES ----------------------
    mult_ts = True

    # ---------------------- DATA PARAMETERS ------------------------
    offset_days = 12
    train_days = 28
    valid_days = 7
    test_days = 2

    offset_hours = offset_days * 24
    train_hours = train_days * 24
    valid_hours = valid_days * 24
    test_hours = test_days * 24
    window_size = 168
    output_size = 48
    batch_size = calc_batch_size(
        train_hours - window_size - output_size + 1, 64
    )
    seasonality = 24

    # ---------------------- LOAD AND PREPROCESS ------------------------
    file_path = os.path.abspath(os.path.dirname(__file__))
    data_path = os.path.join(file_path, "data/spain/energy_dataset.csv")
    data = load_data(data_path, mult_ts)
    data.interpolate(inplace=True)
    data = data.set_index('time').asfreq('H')
    data.index = pd.to_datetime(data.index, utc=True)

    # Train hours = 672 hours (28 days)
    # Window size = 268 hours (7 days) -> This is the input size to our model
    # => 672 - 268 = 404 training examples
    #
    # We have to feed in at 268 data points at a time to get it to
    # generate a single forecast. So we now start at train_hours -
    # window_size + 1, and go to
    # Validate
    # on the next 14 days:

    data = data[
        offset_hours:offset_hours + train_hours + valid_hours + test_hours
    ]
    forecast = drnn_48_multiple.forecast(
        data, train_hours, valid_hours, test_hours, window_size,
        output_size, batch_size, True
    )

    # -------- Adjusted LSTM Testing ---------
    # actual_vals = data[168:216]
    # data = data[:168]
    # forecast = lstm_adjusted.forecast(
    #     data, train_hours, test_hours, in_place=True
    # )
    #
    # print("Actual Values:", actual_vals)
    #
    # # Plot the actual data and the forecast
    # fig = plt.figure(figsize=(12.8, 9.6), dpi=250)
    # ax = fig.add_subplot(1, 1, 1)
    # ax.set_title("Basic LSTM Forecasts")
    #
    # ax.plot(data.index[0:168],
    #         data['total load actual'][0:168],
    #         label="Training Data")
    # ax.plot(actual_vals.index,
    #         actual_vals['total load actual'],
    #         label="Test Data")
    # ax.plot(actual_vals.index,
    #         forecast,
    #         label="Forecast")
    #
    # ax.legend(loc="best")
    # plt.show()



    # ----------------------- RUN THE TEST ------------------------
    test_dict = {
        1: [
            [naive2_adjusted.forecast, naive1.forecast],
            ['naive2', 'naive1']
        ],

        2: [
            [naive2_adjusted.forecast, naiveS.forecast],
            ['naive2', 'naiveS']
        ],

        3: [
            [naive2_adjusted.forecast],
            ['naive2']
        ],

        4: [
            [naive2_adjusted.forecast, ses_adjusted.forecast],
            ['naive2', 'ses']
        ],

        5: [
            [naive2_adjusted.forecast, holt_adjusted.forecast],
            ['naive2', 'holt']
        ],

        6: [
            [naive2_adjusted.forecast, holtDamped_adjusted.forecast],
            ['naive2', 'damped']
        ],

        7: [
            [naive2_adjusted.forecast, theta.forecast],
            ['naive2', 'theta']
        ],

        8: [
            [naive2_adjusted.forecast, ses_adjusted.forecast,
             holt_adjusted.forecast, holtDamped_adjusted.forecast,
             comb_adjusted.forecast],
            ['naive2', 'ses', 'holt', 'damped', 'comb']
        ],

        9: [
            [naive2_adjusted.forecast, sarima.forecast],
            ['naive2', 'sarima']
        ],

        10: [
            [naive2_adjusted.forecast, holtWinters.forecast],
            ['naive2', 'holtWinters']
        ],

        11: [
            [naive2_adjusted.forecast, autoSarima.forecast],
            ['naive2', 'auto sarima']
        ]
    }

    # write_results(
    #     test(
    #         data[0:336], seasonality, test_hours, *test_dict[int(sys.argv[1])],
    #         True
    #     ),
    #     sys.argv[1], True
    # )

    # ---------------------- PLOT THE RESULTS ------------------------
    # stats.plots.results_plots()

    # ----------------------- RUN THE DEMO ------------------------
    # data = data[offset_hours:offset_hours + train_hours + test_hours]
    # demo(data, offset_hours, train_hours, test_hours)

    # fig = plt.figure(figsize=(12.8, 9.6), dpi=250)
    # ax = fig.add_subplot(1, 1, 1)
    #
    # data = data[offset_hours:offset_hours + train_hours + test_hours]
    # daily_average = data['total load actual'].resample('D').mean()
    #
    # # Plot the hourly data for the first year
    # ax.set_title("Daily Average Data")
    # ax.plot(daily_average, label="Daily Average")
    #
    # for i in range(train_days + test_days):
    #     if (i + 1) % 7 == 0 or (i + 2) % 7 == 0:
    #         ax.scatter(daily_average.index[i], daily_average[i], marker="o",
    #                    color="orange")
    #     else:
    #         ax.scatter(daily_average.index[i], daily_average[i], marker="o",
    #                    color="blue")
    #
    #     # if (i - 3) % 28 == 0:
    #     #    ax.scatter(daily_average.index[i], daily_average[i], marker="o",
    #     #                color="green")
    #
    # ax.legend(loc="best")
    # plt.setp(ax.get_xticklabels(), rotation=45)
    # plt.show()


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


def demo(data, offset_hours, train_hours, test_hours):
    # Plot the data at different frequencies to get a feel for it
    stats.plots.resample_plots(data)

    # For the rest of the demo, consider only the first month of data
    data = data[offset_hours:offset_hours + train_hours + test_hours]

    # Plot the ACF of the first two weeks to further assess seasonality
    # helpers.acf_plots(data)

    # Plot Time Series Decomposition to get an idea of the seasonality
    stats.plots.decomp_adjusted_plots(data, "additive")
    stats.plots.decomp_adjusted_plots(data, "multiplicative")

    # Calculate the seasonal indices, and then divide each data point to get
    # a seasonally-adjusted value, which can then be used to forecast,
    # and then convert back
    stats.plots.indices_adjusted_plots(data, train_hours, test_hours)

    # Power Spectrum Plot - currently broken
    # helpers.power_spectrum_plot(data)

    # Plot the ACF and PACF of differenced and seasonally differenced data
    stats.plots.differenced_plots(data, train_hours, test_hours)

    # Statistical test for stationarity
    helpers.test_stationarity(data)

    # Naive 1, Naive S and the ARIMA models use the original data
    naive1.forecast(data, train_hours)
    naiveS.forecast(data, train_hours, test_hours)
    autoSarima.forecast(data, train_hours, test_hours)
    # data['auto sarima'] = data['total load actual'][train_hours]
    sarima.forecast(data, train_hours, test_hours)

    # The other statistical forecasts use the seasonally differenced data
    naive2.forecast(data, train_hours)
    ses.forecast(data, train_hours, test_hours)
    holt.forecast(data, train_hours, test_hours)
    holtDamped.forecast(data, train_hours, test_hours)
    holtWinters.forecast(data, train_hours, test_hours)

    # Forecast using the data divided by the seasonal indices
    naive2_adjusted.forecast(data, train_hours)
    ses_adjusted.forecast(data, train_hours, test_hours)
    holt_adjusted.forecast(data, train_hours, test_hours)
    holtDamped_adjusted.forecast(data, train_hours, test_hours)

    # Convert differenced forecasts back into actual forecasts
    naive2.undifference(data, train_hours, test_hours)
    ses.undifference(data, train_hours, test_hours)
    holt.undifference(data, train_hours, test_hours)
    holtDamped.undifference(data, train_hours, test_hours)

    # Compute the two forecasting methods that are simply averages of the
    # some of the others
    comb.forecast(data)
    comb_adjusted.forecast(data)

    # The final, theta forecasts
    theta.forecast(data, train_hours, test_hours)

    # Figure for the non-seasonally adjusted forecasts
    fig = plt.figure(figsize=(12.8, 9.6), dpi=250)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title("Forecasts of Non-Seasonally Differenced Data")

    # Plot the error bars first so the lines go over the top
    # for i in range(test_hours):
    #     ax.plot([data.index[train_hours + i:],
    #              data.index[train_hours + i:]], [data[
    #                                                      'holtWinters'][
    #                                                  train_hours + i:],
    #                                                data['total load actual'][
    #                                                  train_hours + i:]],
    #             color="lightgrey", linestyle="--")

    # Plot the non-seasonally adjusted actual data
    ax.plot(data.index[0:train_hours],
            data['total load actual'][0:train_hours],
            label="Training Data")
    ax.plot(data.index[train_hours:],
            data['total load actual'][train_hours:],
            label="Test Data")

    # Plot the Naive1, NaiveS, and Holt-Winters Forecasts
    ax.plot(data.index[train_hours:], data['naive1'][train_hours:],
            label="Naive1")
    ax.plot(data.index[train_hours:], data['naiveS'][train_hours:],
            label="NaiveS")
    ax.plot(data.index[train_hours:], data['holtWinters'][train_hours:],
            label="Holt Winters")

    # Add the legend and show the plot
    ax.legend(loc="best")
    plt.show()

    # Figure for the the seasonally-adjusted forecasts
    fig = plt.figure(figsize=(12.8, 9.6), dpi=250)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title("Forecasts of Seasonal Differences")

    # Plot the seasonally adjusted actual data
    ax.plot(data.index[0:train_hours],
            data['seasonally differenced'][0:train_hours],
            label="Train Data")
    ax.plot(data.index[train_hours:],
            data['seasonally differenced'][train_hours:],
            label="Test Data")

    # Plot the Naive2, SES, Holt and Damped Holt Difference Forecasts
    ax.plot(data.index[train_hours:], data['naive2'][train_hours:],
            label="Naive2")
    ax.plot(data.index[train_hours:], data['ses'][train_hours:],
            label="SES")
    ax.plot(data.index[train_hours:], data['holt'][train_hours:],
            label="Holt")
    ax.plot(data.index[train_hours:], data['holtDamped'][train_hours:],
            label="Holt Damped")

    # Add the legend and show the plot
    ax.legend(loc="best")
    plt.show()

    # Figure for the Undifferenced difference forecasts
    fig = plt.figure(figsize=(12.8, 9.6), dpi=250)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title("Actual Forecasts using Seasonal Difference Forecast")

    # Plot the non-seasonally adjusted actual data
    ax.plot(data.index[0:train_hours],
            data['total load actual'][0:train_hours],
            label="Training Data")
    ax.plot(data.index[train_hours:],
            data['total load actual'][train_hours:],
            label="Test Data")

    ax.plot(data.index[train_hours:], data['naive2 undiff'][train_hours:],
            label="Naive2")
    ax.plot(data.index[train_hours:], data['ses undiff'][train_hours:],
            label="SES")
    ax.plot(data.index[train_hours:], data['holt undiff'][train_hours:],
            label="Holt")
    ax.plot(data.index[train_hours:], data['holtDamped undiff'][train_hours:],
            label="Holt Damped")

    # Add the legend and show the plot
    ax.legend(loc="best")
    plt.show()

    # Figure for the ARIMA and SARIMA Plots
    fig = plt.figure(figsize=(12.8, 9.6), dpi=250)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title("ARIMA Forecasts")

    # Plot the non-seasonally adjusted actual data
    ax.plot(data.index[0:train_hours],
            data['total load actual'][0:train_hours],
            label="Training Data")
    ax.plot(data.index[train_hours:],
            data['total load actual'][train_hours:],
            label="Test Data")

    ax.plot(data.index[train_hours:], data['auto sarima'][train_hours:],
            label="Auto SARIMA")
    ax.plot(data.index[train_hours:], data['sarima'][train_hours:],
            label="SARIMA")
    ax.legend(loc="best")
    plt.show()

    # Figure for the adjusted forecasts
    fig = plt.figure(figsize=(12.8, 9.6), dpi=250)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title("Forecasts using Seasonally Adjusted Data")

    ax.plot(data.index[0:train_hours],
            data['total load actual'][0:train_hours],
            label="Training Data")
    ax.plot(data.index[train_hours:],
            data['total load actual'][train_hours:],
            label="Test Data")

    ax.plot(data.index[train_hours:], data['naive2 adjusted'][train_hours:],
            label="Naive2")
    ax.plot(data.index[train_hours:], data['ses adjusted'][train_hours:],
            label="SES")
    ax.plot(data.index[train_hours:], data['holt adjusted'][train_hours:],
            label="Holt")
    ax.plot(data.index[train_hours:],
            data['holtDamped adjusted'][train_hours:],
            label="Holt Damped")

    # Add the legend and show the plot
    ax.legend(loc="best")
    plt.show()

    # Figure for the combined models
    fig = plt.figure(figsize=(12.8, 9.6), dpi=250)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title("Forecasts using Combined Methods")

    ax.plot(data.index[0:train_hours],
            data['total load actual'][0:train_hours],
            label="Training Data")
    ax.plot(data.index[train_hours:],
            data['total load actual'][train_hours:],
            label="Test Data")

    ax.plot(data.index[train_hours:], data['comb undiff'][train_hours:],
            label="Combined")
    ax.plot(data.index[train_hours:], data['comb adjusted'][train_hours:],
            label="Combined Adjusted")
    # Add the legend and show the plot
    ax.legend(loc="best")
    plt.show()

    # Figure for the the theta forecasts
    fig = plt.figure(figsize=(12.8, 9.6), dpi=250)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title("Theta Forecasts")

    # Plot the seasonally adjusted actual data
    ax.plot(data.index[:train_hours],
            data['total load actual'][0:train_hours],
            label="Training Data")
    ax.plot(data.index[train_hours:],
            data['total load actual'][train_hours:],
            label="Test Data")

    # Plot the theta forecasts
    ax.plot(data.index[:train_hours],
            data['theta0'][:train_hours],
            label="Theta 0")
    ax.plot(data.index[:train_hours],
            data['theta2'][:train_hours],
            label="Theta 2")
    ax.plot(data.index[train_hours:],
            data['theta'][train_hours:],
            label="Theta")

    # Add the legend and show the plot
    ax.legend(loc="best")
    plt.show()

    # Show the error statistics
    error_dict = {
        "Method": ["Naive 1", "Naive S", "Naive 2", "SES", "Holt", "Damped",
                   "Holt Winters", "Auto SARIMA", "ARIMA", "SES*", "Comb",
                   "Theta"],
        "MAPE": [errors.sMAPE(data["naive1"][train_hours:],
                              data["total load actual"][train_hours:]),
                 errors.sMAPE(data["naiveS"][train_hours:],
                              data["total load actual"][train_hours:]),
                 errors.sMAPE(data["naive2 undiff"][train_hours:],
                              data["total load actual"][train_hours:]),
                 errors.sMAPE(data["ses undiff"][train_hours:],
                              data["total load actual"][train_hours:]),
                 errors.sMAPE(data["holt undiff"][train_hours:],
                              data["total load actual"][train_hours:]),
                 errors.sMAPE(data["holtDamped undiff"][train_hours:],
                              data["total load actual"][train_hours:]),
                 errors.sMAPE(data["holtWinters"][train_hours:],
                              data["total load actual"][train_hours:]),
                 errors.sMAPE(data['auto sarima'][train_hours:],
                              data["total load actual"][train_hours:]),
                 errors.sMAPE(data['sarima'][train_hours:],
                              data["total load actual"][train_hours:]),
                 errors.sMAPE(data['ses adjusted'][train_hours:],
                              data["total load actual"][train_hours:]),
                 errors.sMAPE(data['comb adjusted'][train_hours:],
                              data["total load actual"][train_hours:]),
                 errors.sMAPE(data['theta'][train_hours:],
                              data["total load actual"][train_hours:])
                 ],

        "RMSE": [errors.RMSE(data["naive1"][train_hours:],
                             data["total load actual"][train_hours:]),
                 errors.RMSE(data["naiveS"][train_hours:],
                             data["total load actual"][train_hours:]),
                 errors.RMSE(data["naive2 undiff"][train_hours:],
                             data["total load actual"][train_hours:]),
                 errors.RMSE(data["ses undiff"][train_hours:],
                             data["total load actual"][train_hours:]),
                 errors.RMSE(data["holt undiff"][train_hours:],
                             data["total load actual"][train_hours:]),
                 errors.RMSE(data["holtDamped undiff"][train_hours:],
                             data["total load actual"][train_hours:]),
                 errors.RMSE(data["holtWinters"][train_hours:],
                             data["total load actual"][train_hours:]),
                 errors.RMSE(data['auto sarima'][train_hours:],
                             data["total load actual"][train_hours:]),
                 errors.RMSE(data['sarima'][train_hours:],
                             data["total load actual"][train_hours:]),
                 errors.RMSE(data['ses adjusted'][train_hours:],
                             data["total load actual"][train_hours:]),
                 errors.RMSE(data['comb adjusted'][train_hours:],
                             data["total load actual"][train_hours:]),
                 errors.RMSE(data['theta'][train_hours:],
                             data["total load actual"][train_hours:])
                 ],
        "MASE": [errors.MASE(data["naive1"][train_hours:],
                             data["total load actual"][train_hours:], 1),
                 errors.MASE(data["naiveS"][train_hours:],
                             data["total load actual"][train_hours:], 1),
                 errors.MASE(data["naive2 undiff"][train_hours:],
                             data["total load actual"][train_hours:], 1),
                 errors.MASE(data["ses undiff"][train_hours:],
                             data["total load actual"][train_hours:], 1),
                 errors.MASE(data["holt undiff"][train_hours:],
                             data["total load actual"][train_hours:], 1),
                 errors.MASE(data["holtDamped undiff"][train_hours:],
                             data["total load actual"][train_hours:], 1),
                 errors.MASE(data["holtWinters"][train_hours:],
                             data["total load actual"][train_hours:], 1),
                 errors.MASE(data['auto sarima'][train_hours:],
                             data["total load actual"][train_hours:], 1),
                 errors.MASE(data['sarima'][train_hours:],
                             data["total load actual"][train_hours:], 1),
                 errors.MASE(data['ses adjusted'][train_hours:],
                             data["total load actual"][train_hours:], 1),
                 errors.MASE(data['comb adjusted'][train_hours:],
                             data["total load actual"][train_hours:], 1),
                 errors.MASE(data['theta'][train_hours:],
                             data["total load actual"][train_hours:], 1)
                 ],
        "MAE": [errors.MAE(data["naive1"][train_hours:],
                           data["total load actual"][train_hours:]),
                errors.MAE(data["naiveS"][train_hours:],
                           data["total load actual"][train_hours:]),
                errors.MAE(data["naive2 undiff"][train_hours:],
                           data["total load actual"][train_hours:]),
                errors.MAE(data["ses undiff"][train_hours:],
                           data["total load actual"][train_hours:]),
                errors.MAE(data["holt undiff"][train_hours:],
                           data["total load actual"][train_hours:]),
                errors.MAE(data["holtDamped undiff"][train_hours:],
                           data["total load actual"][train_hours:]),
                errors.MAE(data["holtWinters"][train_hours:],
                           data["total load actual"][train_hours:]),
                errors.MAE(data['auto sarima'][train_hours:],
                           data["total load actual"][train_hours:]),
                errors.MAE(data['sarima'][train_hours:],
                           data["total load actual"][train_hours:]),
                errors.MAE(data['ses adjusted'][train_hours:],
                           data["total load actual"][train_hours:]),
                errors.MAE(data['comb adjusted'][train_hours:],
                           data["total load actual"][train_hours:]),
                errors.MAE(data['theta'][train_hours:],
                           data["total load actual"][train_hours:])
                ]
    }

    error_data = pd.DataFrame(error_dict)
    print("Forecast Accuracy:")
    print(error_data)

    plt.show()

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
