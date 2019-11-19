import pandas as pd
from pandas.plotting import register_matplotlib_converters
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt
import matplotlib.pyplot as plt
from stats import ses, helpers, naive1, naive2, naiveS, holt, holtDamped, \
    holtWinters


def load_data(filename):
    return pd.read_csv(filename, parse_dates=["time"],
                       usecols=["time", "total load actual"],
                       infer_datetime_format=True)


def main():
    # Load and preprocess
    data = load_data("/Users/matt/Projects/AdvancedResearchProject/data/spain/"
                     "energy_dataset.csv")
    data.set_index('time', inplace=True)
    data.interpolate(inplace=True)  # Interpolate missing values
    data.index = pd.to_datetime(data.index, utc=True)

    # Plots the data at different frequencies to get a feel for the data
    # helpers.resample_plots(data)

    # Plot Time Series Decomposition to get an idea of the seasonality
    # helpers.decompose_plots(data, "additive")
    # helpers.decompose_plots(data, "multiplicative")

    # Plot the ACF of the first two weeks to further assess seasonality
    # helpers.acf_plots(data)

    # Statistical test for stationarity
    # helpers.test_stationarity(data)

    # Specify the number of days to use for training and testing
    train_days = 7
    test_days = 2
    data = data[:(train_days + test_days) * 24]

    # Naive 1, Naive S use the original data
    naive1.forecast(data, train_days)
    naiveS.forecast(data)

    # All other statistical forecasts use the seasonally adjusted data
    helpers.seasonally_adjust(data, train_days, test_days, "multiplicative")
    naive2.forecast(data, train_days)
    ses.forecast(data, train_days, test_days)
    holt.forecast(data, train_days, test_days)
    holtDamped.forecast(data, train_days, test_days)
    holtWinters.forecast(data, train_days, test_days)

    # Plot the non-seasonally adjusted forecasts
    fig = plt.figure(figsize=(12.8, 9.6), dpi=250)
    ax = fig.add_subplot(1, 1, 1)

    # Plot the error bars first so the lines go over the top
    for i in range((train_days + test_days) * 24):
        ax.plot([data.index[i], data.index[i]], [data['holtWinters'][i],
                                                 data['total load actual'][i]],
                color="grey", linestyle="--")

    ax.plot(data.index[0:train_days * 24],
            data['total load actual'][0:train_days * 24],
            label="Training Data")
    ax.plot(data.index[train_days * 24:],
            data['total load actual'][train_days * 24:],
            label="Test Data")
    ax.plot(data.index, data['naive1'], label="Naive1",)
    # ax.plot(data.index[train_days * 24:], data['naiveS'][train_days * 24:],
    #         label="NaiveS")
    ax.plot(data.index, data['holtWinters'], label="Holt Winters")
    ax.legend(loc="best")
    plt.show()

    # Plot the seasonally-adjusted forecasts
    fig = plt.figure(figsize=(12.8, 9.6), dpi=250)
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(data.index[0:train_days * 24],
            data['adjusted'][0:train_days * 24],
            label="Training Data")
    ax.plot(data.index[train_days * 24:],
            data['adjusted'][train_days * 24:],
            label="Test Data")
    # ax.plot(data.index[train_days * 24:],
    # data['naive2'][train_days * 24:],
    # label="Naive2")
    # ax.plot(data.index, data['ses'], label="SES")
    ax.plot(data.index, data['holt'], label="Holt")
    ax.plot(data.index, data['holtDamped'], label="Holt Damped")


    ax.legend(loc="best")
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

