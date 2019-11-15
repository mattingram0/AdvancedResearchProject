import pandas as pd
import matplotlib.pyplot as plt
from stats import naive1, naive2, naiveS
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.api import seasonal_decompose


def sma(data, window):
    data['sma' + str(window)] = data['total load actual'].rolling(
        window=window).mean()


def resample_plots(data):
    # Create a new figure of a certain size, with a higher resolution (DPI)
    # Add one subplot (axes) to the figure. We can now customise our plots
    # using these axes
    fig = plt.figure(figsize=(12.8, 9.6), dpi=250)
    ax = fig.add_subplot(2, 1, 1)

    # Plot the hourly data for the first year
    ax.set_title("Hourly Data")
    ax.plot(data.iloc[0:8760].index, data.iloc[0:8760]["total load actual"])

    # Convert the index to a DatatimeIndex - required for resampling
    data.index = pd.to_datetime(data.index, utc=True)

    # Resample the data - finding a daily and weekly average. Resampling is
    # a form of data aggregation
    daily_average = data.resample('D').mean()
    weekly_average = data.resample('W').mean()

    # Add a second subplot and plot the daily and weekly averages
    ax2 = fig.add_subplot(2, 1, 2)
    ax2.set_title("Daily and Weekly Averages")
    ax2.plot(daily_average.iloc[0:365].index,
             daily_average.iloc[0:365]["total load actual"], linestyle=":",
             label="Daily Average")
    ax2.plot(weekly_average.iloc[0:52].index,
             weekly_average.iloc[0:52]["total load actual"], linestyle="-",
             label="Weekly Average")

    # Convert the frequency of the data. Asfrew is a form of data selection
    # - certain values are selected at given frequencies
    # daily_start_frequency = data.asfreq("D")
    daily_midday_frequency = data.asfreq("D12H")  # Check this is correct
    weekly_frequency = data.asfreq("W")

    # ax2.plot(daily_midday_frequency.iloc[0:365].index,
    #          daily_midday_frequency.iloc[0:365]["total load actual"],
    #          linestyle="-", label="Midday")
    ax2.plot(weekly_frequency.iloc[0:52].index,
             weekly_frequency.iloc[0:52]["total load actual"],
             linestyle="--", label="Weekly Frequency")

    # Add the legend in the best place to the second axes
    ax2.legend(loc="best")

    # Show the plot
    plt.show()


def acf_plots(data):
    data = data.head(336)  # Using diff variable caused the weird errors

    # Create a figure with 6 (2 x 3) subplots, increase size and resolution,
    # add titles and rotate x-axis labels
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12.8, 9.6), dpi=250)
    fig.canvas.set_window_title("Visualisation and Autocorrelation")
    axes[0, 0].set_title("Week 1")
    axes[1, 0].set_title("Week 2")

    # Rotate x-axis labels by 45 degrees
    for ax in axes.flat:
        plt.setp(ax.get_xticklabels(), rotation=45)


    # Plot week 1 data and autocorrelation
    axes[0, 0].plot(data.iloc[0:168].index,
                    data.iloc[0:168]["total load actual"],
                    label="Total Load Actual")
    autocorrelation_plot(data.iloc[0:168]["total load actual"], ax=axes[0, 1])
    plot_acf(x=data.iloc[0:168]["total load actual"], ax=axes[0, 2], alpha=0.1,
             lags=50)

    # Plot week 2 data and autocorrelation
    axes[1, 0].plot(data.iloc[168:].index,
                    data.iloc[168:]["total load actual"],
                    label="Total Load Actual")
    autocorrelation_plot(data.iloc[168:]["total load actual"], ax=axes[1, 1])
    plot_acf(x=data.iloc[168:]["total load actual"], ax=axes[1, 2], alpha=0.1,
             lags=50)

    # Show legends on each of the subplots
    axes[0, 0].legend(loc="best")
    axes[1, 0].legend(loc="best")

    # Show the plot
    plt.show()


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


def decompose_plots(data, model):
    data.index = pd.to_datetime(data.index, utc=True)

    # Decompose
    decomp = seasonal_decompose(data['total load actual'][0:168],
                                model=model, freq=24)

    # Plot the decomposition
    fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(12.8, 9.6), dpi=250)
    axes[0].plot(decomp.observed)
    axes[1].plot(decomp.trend)
    axes[2].plot(decomp.seasonal)
    axes[3].plot(decomp.resid)
    axes[4].plot(decomp.observed - decomp.seasonal if model == "additive"
                 else decomp.observed / decomp.seasonal)

    # Add titles
    axes[0].set_title("Observed Data")
    axes[1].set_title("Trend Component")
    axes[2].set_title("Seasonal Component")
    axes[3].set_title("Residual Component")
    axes[4].set_title("Seasonally Adjusted Data")

    # Show the figure
    plt.show()


def seasonally_adjust(data, model):
    data.index = pd.to_datetime(data.index, utc=True)
    decomp = seasonal_decompose(data['total load actual'][0:168],
                                model=model, freq=24)
    data['adjusted'] = decomp.observed - decomp.seasonal if model == \
                                                            "additive" else \
        decomp.observed / decomp.seasonal
