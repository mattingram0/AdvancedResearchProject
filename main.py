import pandas as pd
from pandas.plotting import register_matplotlib_converters
import matplotlib.pyplot as plt

import stats.errors
import stats.plots
from stats import ses, helpers, naive1, naive2, naiveS, holt, holtDamped, \
    holtWinters, autoSarima, sarima, naive2_adjusted, ses_adjusted, \
    holt_adjusted, holtDamped_adjusted, comb, comb_adjusted, theta


def load_data(filename):
    return pd.read_csv(filename, parse_dates=["time"],
                       usecols=["time", "total load actual"],
                       infer_datetime_format=True)


def main():
    # Training/Test Data Parameters
    offset_days = 5
    train_days = 6
    test_days = 1

    offset_hours = offset_days * 24
    train_hours = train_days * 24
    test_hours = test_days * 24

    # Load and Pre-process the Data
    data = load_data("/Users/matt/Projects/AdvancedResearchProject/data/spain/"
                     "energy_dataset.csv")
    data.interpolate(inplace=True)  # Interpolate missing values
    data = data.set_index('time').asfreq('H')
    data.index = pd.to_datetime(data.index, utc=True)

    data = data[offset_hours:offset_hours + train_hours + test_hours]
    # Compare differenced vs seasonal indices vs seasonal adjustment
    helpers.indices_adjust(data, train_hours, test_hours)
    helpers.seasonally_difference(data)

    # Run the demo
    # demo(data, offset_hours, train_hours, test_hours)
    #
    #
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
    #     #     ax.scatter(daily_average.index[i], daily_average[i], marker="o",
    #     #                color="green")
    #
    # ax.legend(loc="best")
    # plt.setp(ax.get_xticklabels(), rotation=45)
    # plt.show()


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

    # Power Spectrum Plot - currently broken
    # helpers.power_spectrum_plot(data)

    # Plot the ACF and PACF of differenced and seasonally differenced data
    stats.plots.differenced_plots(data, train_hours, test_hours)

    # Calculate the seasonal indices, and then divide each data point to get
    # a seasonally-adjusted value, which can then be used to forecast,
    # and then convert back
    stats.plots.indices_adjusted_plots(data, train_hours, test_hours)

    # Statistical test for stationarity
    helpers.test_stationarity(data)

    # Naive 1, Naive S and the ARIMA models use the original data
    naive1.forecast(data, train_hours)
    naiveS.forecast(data, train_hours, test_hours)
    autoSarima.forecast(data, train_hours, test_hours)
    # data['auto sarima'] = data['total load actual'][train_hours]  # Dummy
    # data
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
    #                                                  data['total load actual'][
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
        "MAPE": [stats.errors.sMAPE(data["naive1"][train_hours:],
                               data["total load actual"][train_hours:]),
                 stats.errors.sMAPE(data["naiveS"][train_hours:],
                               data["total load actual"][train_hours:]),
                 stats.errors.sMAPE(data["naive2 undiff"][train_hours:],
                               data["total load actual"][train_hours:]),
                 stats.errors.sMAPE(data["ses undiff"][train_hours:],
                               data["total load actual"][train_hours:]),
                 stats.errors.sMAPE(data["holt undiff"][train_hours:],
                               data["total load actual"][train_hours:]),
                 stats.errors.sMAPE(data["holtDamped undiff"][train_hours:],
                               data["total load actual"][train_hours:]),
                 stats.errors.sMAPE(data["holtWinters"][train_hours:],
                               data["total load actual"][train_hours:]),
                 stats.errors.sMAPE(data['auto sarima'][train_hours:],
                               data["total load actual"][train_hours:]),
                 stats.errors.sMAPE(data['sarima'][train_hours:],
                               data["total load actual"][train_hours:]),
                 stats.errors.sMAPE(data['ses adjusted'][train_hours:],
                               data["total load actual"][train_hours:]),
                 stats.errors.sMAPE(data['comb adjusted'][train_hours:],
                               data["total load actual"][train_hours:]),
                 stats.errors.sMAPE(data['theta'][train_hours:],
                               data["total load actual"][train_hours:])
                 ],

        "RMSE": [stats.errors.RMSE(data["naive1"][train_hours:],
                              data["total load actual"][train_hours:]),
                 stats.errors.RMSE(data["naiveS"][train_hours:],
                              data["total load actual"][train_hours:]),
                 stats.errors.RMSE(data["naive2 undiff"][train_hours:],
                              data["total load actual"][train_hours:]),
                 stats.errors.RMSE(data["ses undiff"][train_hours:],
                              data["total load actual"][train_hours:]),
                 stats.errors.RMSE(data["holt undiff"][train_hours:],
                              data["total load actual"][train_hours:]),
                 stats.errors.RMSE(data["holtDamped undiff"][train_hours:],
                              data["total load actual"][train_hours:]),
                 stats.errors.RMSE(data["holtWinters"][train_hours:],
                              data["total load actual"][train_hours:]),
                 stats.errors.RMSE(data['auto sarima'][train_hours:],
                              data["total load actual"][train_hours:]),
                 stats.errors.RMSE(data['sarima'][train_hours:],
                              data["total load actual"][train_hours:]),
                 stats.errors.RMSE(data['ses adjusted'][train_hours:],
                              data["total load actual"][train_hours:]),
                 stats.errors.RMSE(data['comb adjusted'][train_hours:],
                              data["total load actual"][train_hours:]),
                 stats.errors.RMSE(data['theta'][train_hours:],
                              data["total load actual"][train_hours:])
                 ],
        "MASE": [stats.errors.MASE(data["naive1"][train_hours:],
                              data["total load actual"][train_hours:], 1),
                 stats.errors.MASE(data["naiveS"][train_hours:],
                              data["total load actual"][train_hours:], 1),
                 stats.errors.MASE(data["naive2 undiff"][train_hours:],
                              data["total load actual"][train_hours:], 1),
                 stats.errors.MASE(data["ses undiff"][train_hours:],
                              data["total load actual"][train_hours:], 1),
                 stats.errors.MASE(data["holt undiff"][train_hours:],
                              data["total load actual"][train_hours:], 1),
                 stats.errors.MASE(data["holtDamped undiff"][train_hours:],
                              data["total load actual"][train_hours:], 1),
                 stats.errors.MASE(data["holtWinters"][train_hours:],
                              data["total load actual"][train_hours:], 1),
                 stats.errors.MASE(data['auto sarima'][train_hours:],
                              data["total load actual"][train_hours:], 1),
                 stats.errors.MASE(data['sarima'][train_hours:],
                              data["total load actual"][train_hours:], 1),
                 stats.errors.MASE(data['ses adjusted'][train_hours:],
                              data["total load actual"][train_hours:], 1),
                 stats.errors.MASE(data['comb adjusted'][train_hours:],
                              data["total load actual"][train_hours:], 1),
                 stats.errors.MASE(data['theta'][train_hours:],
                              data["total load actual"][train_hours:], 1)
                 ],
        "MAE": [stats.errors.MAE(data["naive1"][train_hours:],
                            data["total load actual"][train_hours:]),
                stats.errors.MAE(data["naiveS"][train_hours:],
                            data["total load actual"][train_hours:]),
                stats.errors.MAE(data["naive2 undiff"][train_hours:],
                            data["total load actual"][train_hours:]),
                stats.errors.MAE(data["ses undiff"][train_hours:],
                            data["total load actual"][train_hours:]),
                stats.errors.MAE(data["holt undiff"][train_hours:],
                            data["total load actual"][train_hours:]),
                stats.errors.MAE(data["holtDamped undiff"][train_hours:],
                            data["total load actual"][train_hours:]),
                stats.errors.MAE(data["holtWinters"][train_hours:],
                            data["total load actual"][train_hours:]),
                stats.errors.MAE(data['auto sarima'][train_hours:],
                            data["total load actual"][train_hours:]),
                stats.errors.MAE(data['sarima'][train_hours:],
                            data["total load actual"][train_hours:]),
                stats.errors.MAE(data['ses adjusted'][train_hours:],
                            data["total load actual"][train_hours:]),
                stats.errors.MAE(data['comb adjusted'][train_hours:],
                            data["total load actual"][train_hours:]),
                stats.errors.MAE(data['theta'][train_hours:],
                            data["total load actual"][train_hours:])
                ]
    }

    error_data = pd.DataFrame(error_dict)
    print("Forecast Accuracy:")
    print(error_data)

    plt.show()


if __name__ == "main":
    # matplotlib.style.use('seaborn-deep')
    register_matplotlib_converters()
    # pd.options.mode.chained_assignment = None TURN OFF CHAINED ASSIGNMENTS
    # WARNING - possibly use in the future if I run into that message again
    main()

# Entry point when using PyCharm - REMOVE
# matplotlib.style.use('seaborn-deep')
register_matplotlib_converters()
pd.options.mode.chained_assignment = None
# plt.style.use('Solarize_Light2')
plt.style.use('seaborn-deep')
main()
