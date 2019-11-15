import pandas as pd
from pandas.plotting import register_matplotlib_converters
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt
import matplotlib.pyplot as plt
from stats import ses, helpers, naive1, naive2, naiveS


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

    # Plots the data at different frequencies to get a feel for the data
    # helpers.resample_plots(data)

    # Plot Time Series Decomposition to get an idea of the seasonality
    # helpers.decompose_plots(data, "additive")
    # helpers.decompose_plots(data, "multiplicative")

    # Plot the ACF of the first two weeks to further assess seasonality
    # helpers.acf_plots(data)

    # Statistical test for stationarity
    # helpers.test_stationarity(data)

    data = data[:168]  # Use only the first week of data

    # Naive 1, Naive S use the original data
    naive1.forecast(data)
    naiveS.forecast(data)

    # All other statistical forecasts use the seasonally adjusted data
    helpers.seasonally_adjust(data, "additive")
    naive2.forecast(data)
    ses.forecast(data)

    # Plot the non-seasonally adjusted forecasts
    fig = plt.figure(figsize=(12.8, 9.6), dpi=250)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(data.index[0:144], data['total load actual'][0:144],
            label="Training Data")
    ax.plot(data.index[144:], data['total load actual'][144:], label="Test "
                                                                     "Data")
    ax.plot(data.index, data['naive1'], label="Naive1",)
    ax.plot(data.index[144:], data['naiveS'][144:], label="NaiveS")
    ax.legend(loc="best")
    plt.show()

    # Plot the seasonally-adjusted forecasts
    fig = plt.figure(figsize=(12.8, 9.6), dpi=250)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(data.index[0:144], data['adjusted'][0:144], label="Training Data")
    ax.plot(data.index[144:], data['adjusted'][144:], label="Test Data")
    ax.plot(data.index[144:], data['naive2'][144:], label="Naive2")
    ax.plot(data.index, data['ses'], label="SES")
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

