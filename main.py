import pandas as pd
from pandas.plotting import register_matplotlib_converters

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt
import matplotlib.pyplot as plt
from stats import ses


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

    # Perform an Augmented Dickey-Fuller Test for stationarity
    # test_stationarity(data)

    # Plots the data at different frequencies
    # helpers.resample_plots(data)

    # Plot the ACF of the first two weeks to assess stationarity
    # helpers.acf_plots(data)

    # Simple Exponential Smoothing Forecast
    # ses.forecast(data)

    #


def test_stationarity(data):
    result = adfuller(data["total load actual"].iloc[0:1344],
                            autolag='AIC')
    print("Test Statistic = {:.3f}".format(result[0]))  # The error is a bug
    print("P-value = {:.3f}".format(result[1]))
    print("Critical values :")
    for k, v in result[4].items():
        print(
            "\t{}: {:.4f} (The data is {}stationary with {}% "
            "confidence)".format(
                k, v, "not " if v < result[0] else "", 100 - int(k[:-1])))


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

