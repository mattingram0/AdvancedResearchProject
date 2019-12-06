import pandas as pd

from pandas.plotting import register_matplotlib_converters
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
import matplotlib.pyplot as plt
import numpy as np
from stats import ses, helpers, naive1, naive2, naiveS, holt, holtDamped, \
    holtWinters, autoSarima, sarima


def load_data(filename):
    return pd.read_csv(filename, parse_dates=["time"],
                       usecols=["time", "total load actual"],
                       infer_datetime_format=True)


def main():

    # Training/Test Data Parameters
    offset_days = 7
    train_days = 14
    test_days = 2

    offset_hours = offset_days * 24
    train_hours = train_days * 24
    test_hours = test_days * 24

    # Load and Pre-process the Data
    data = load_data("/Users/matt/Projects/AdvancedResearchProject/data/spain/"
                     "energy_dataset.csv")
    data.interpolate(inplace=True)  # Interpolate missing values
    data = data.set_index('time').asfreq('H')
    data.index = pd.to_datetime(data.index, utc=True)

    # Run the demo
    demo(data, offset_days, train_hours, test_hours)


def demo(data, offset_hours, train_hours, test_hours):
    # Plot the data at different frequencies to get a feel for it
    helpers.resample_plots(data)

    # For the rest of the demo, consider only the first month of data
    data = data[offset_hours:offset_hours + train_hours + test_hours]

    # Plot the ACF of the first two weeks to further assess seasonality
    helpers.acf_plots(data)

    # Plot Time Series Decomposition to get an idea of the seasonality
    helpers.decompose_plots(data, "additive")
    helpers.decompose_plots(data, "multiplicative")

    # Power Spectrum Plot - currently broken
    # helpers.power_spectrum_plot(data)

    # Plot the ACF and PACF of differenced and seasonally differenced data
    helpers.differenced_plots(data, train_hours, test_hours)

    # Statistical test for stationarity
    helpers.test_stationarity(data)

    # Naive 1, Naive S and the ARIMA models use the original data
    naive1.forecast(data, train_hours)
    naiveS.forecast(data, train_hours, test_hours)
    autoSarima.forecast(data, train_hours, test_hours)
    sarima.forecast(data, train_hours, test_hours)

    # The other statistical forecasts use the seasonally adjusted data
    naive2.forecast(data, train_hours)
    ses.forecast(data, train_hours, test_hours)
    holt.forecast(data, train_hours, test_hours)
    holtDamped.forecast(data, train_hours, test_hours)
    holtWinters.forecast(data, train_hours, test_hours)

    # Convert differenced forecasts back into actual forecasts
    naive2.undifference(data, train_hours, test_hours)
    ses.undifference(data, train_hours, test_hours)
    holt.undifference(data, train_hours, test_hours)
    holtDamped.undifference(data, train_hours, test_hours)

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
    ax.plot(data.index[train_hours:],
            data['holtWinters'][train_hours:],
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

    # Show the error statistics
    error_dict = {
        "Method": ["Naive 1", "Naive S", "Naive 2", "SES", "Holt", "Damped",
                   "Holt Winters", "Auto SARIMA", "ARIMA"],
        "MAPE": [helpers.sMAPE(data["naive1"][train_hours:],
                               data["total load actual"][train_hours:]),
                 helpers.sMAPE(data["naiveS"][train_hours:],
                               data["total load actual"][train_hours:]),
                 helpers.sMAPE(data["naive2 undiff"][train_hours:],
                               data["total load actual"][train_hours:]),
                 helpers.sMAPE(data["ses undiff"][train_hours:],
                               data["total load actual"][train_hours:]),
                 helpers.sMAPE(data["holt undiff"][train_hours:],
                               data["total load actual"][train_hours:]),
                 helpers.sMAPE(data["holtDamped undiff"][train_hours:],
                               data["total load actual"][train_hours:]),
                 helpers.sMAPE(data["holtWinters"][train_hours:],
                               data["total load actual"][train_hours:]),
                 helpers.sMAPE(data['auto sarima'][train_hours:],
                               data["total load actual"][train_hours:]),
                 helpers.sMAPE(data['sarima'][train_hours:],
                               data["total load actual"][train_hours:])],
        "RMSE": [helpers.RMSE(data["naive1"][train_hours:],
                              data["total load actual"][train_hours:]),
                 helpers.RMSE(data["naiveS"][train_hours:],
                              data["total load actual"][train_hours:]),
                 helpers.RMSE(data["naive2 undiff"][train_hours:],
                              data["total load actual"][train_hours:]),
                 helpers.RMSE(data["ses undiff"][train_hours:],
                              data["total load actual"][train_hours:]),
                 helpers.RMSE(data["holt undiff"][train_hours:],
                              data["total load actual"][train_hours:]),
                 helpers.RMSE(data["holtDamped undiff"][train_hours:],
                              data["total load actual"][train_hours:]),
                 helpers.RMSE(data["holtWinters"][train_hours:],
                              data["total load actual"][train_hours:]),
                 helpers.RMSE(data['auto sarima'][train_hours:],
                              data["total load actual"][train_hours:]),
                 helpers.RMSE(data['sarima'][train_hours:],
                              data["total load actual"][train_hours:])],
        "MASE": [helpers.MASE(data["naive1"][train_hours:],
                              data["total load actual"][train_hours:], 1),
                 helpers.MASE(data["naiveS"][train_hours:],
                              data["total load actual"][train_hours:], 1),
                 helpers.MASE(data["naive2 undiff"][train_hours:],
                              data["total load actual"][train_hours:], 1),
                 helpers.MASE(data["ses undiff"][train_hours:],
                              data["total load actual"][train_hours:], 1),
                 helpers.MASE(data["holt undiff"][train_hours:],
                              data["total load actual"][train_hours:], 1),
                 helpers.MASE(data["holtDamped undiff"][train_hours:],
                              data["total load actual"][train_hours:], 1),
                 helpers.MASE(data["holtWinters"][train_hours:],
                              data["total load actual"][train_hours:], 1),
                 helpers.MASE(data['auto sarima'][train_hours:],
                              data["total load actual"][train_hours:], 1),
                 helpers.MASE(data['sarima'][train_hours:],
                              data["total load actual"][train_hours:], 1)
                 ],
        "MAE": [helpers.MAE(data["naive1"][train_hours:],
                            data["total load actual"][train_hours:]),
                helpers.MAE(data["naiveS"][train_hours:],
                            data["total load actual"][train_hours:]),
                helpers.MAE(data["naive2 undiff"][train_hours:],
                            data["total load actual"][train_hours:]),
                helpers.MAE(data["ses undiff"][train_hours:],
                            data["total load actual"][train_hours:]),
                helpers.MAE(data["holt undiff"][train_hours:],
                            data["total load actual"][train_hours:]),
                helpers.MAE(data["holtDamped undiff"][train_hours:],
                            data["total load actual"][train_hours:]),
                helpers.MAE(data["holtWinters"][train_hours:],
                            data["total load actual"][train_hours:]),
                helpers.MAE(data['auto sarima'][train_hours:],
                            data["total load actual"][train_hours:]),
                helpers.MAE(data['sarima'][train_hours:],
                            data["total load actual"][train_hours:])]
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
plt.style.use('Solarize_Light2')
main()
