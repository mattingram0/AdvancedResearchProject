import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt
from scipy import signal
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


def differenced_plots(data, train_hours, test_hours):
    # Adjust and Difference Data
    adjust(data, train_hours, test_hours, "additive")
    difference(data)
    seasonally_difference(data)
    double_difference(data)

    # Create the subplots
    fig = plt.figure(figsize=(20, 14), dpi=250)
    gs = fig.add_gridspec(3, 5)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[2, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    ax5 = fig.add_subplot(gs[2, 1])
    ax6 = fig.add_subplot(gs[1, 2])
    ax7 = fig.add_subplot(gs[2, 2])
    ax8 = fig.add_subplot(gs[1, 3])
    ax9 = fig.add_subplot(gs[2, 3])
    ax10 = fig.add_subplot(gs[1, 4])
    ax11 = fig.add_subplot(gs[2, 4])

    # Plot the time series data
    ax1.plot(data.index, data['total load actual'], label="Actual")
    ax1.plot(data.index, data['adjusted'], label="Adjusted")
    ax1.plot(data.index, data['differenced'], label="Differenced")
    ax1.plot(data.index, data['seasonally differenced'], label="Seasonally "
                                                               "Differenced")
    ax1.plot(data.index, data['double differenced'], label="Seasonally and "
                                                           "Non-Seasonally "
                                                           "Differenced")

    # Plot the ACFs and PACFs
    plot_acf(data[24:]["total load actual"], ax=ax2, alpha=0.05, lags=50)
    plot_pacf(data[24:]["total load actual"], ax=ax3, alpha=0.05, lags=50)
    plot_acf(data[12:-12]["adjusted"], ax=ax4, alpha=0.05, lags=50)
    plot_pacf(data[12:-12]["adjusted"], ax=ax5, alpha=0.05, lags=50)
    plot_acf(data[1:]["differenced"], ax=ax6, alpha=0.05, lags=50)
    plot_pacf(data[1:]["differenced"], ax=ax7, alpha=0.05, lags=50)
    plot_acf(data[24:]["seasonally differenced"], ax=ax8, alpha=0.05, lags=50)
    plot_pacf(data[24:]["seasonally differenced"], ax=ax9, alpha=0.05, lags=50)
    plot_acf(data[25:]["double differenced"], ax=ax10, alpha=0.05, lags=50)
    plot_pacf(data[25:]["double differenced"], ax=ax11, alpha=0.05, lags=50)

    ax2.set_title("ACF for Actual Data")
    ax3.set_title("PACF for Actual Data")
    ax4.set_title("ACF for Adjusted Data")
    ax5.set_title("PACF for Adjusted Data")
    ax6.set_title("ACF for Differenced Data")
    ax7.set_title("PACF for Differenced Data")
    ax8.set_title("ACF for Seasonally Differenced Data")
    ax9.set_title("PACF for Seasonally Differenced Data")
    ax10.set_title("ACF for Double Differenced Data")
    ax11.set_title("PACF for Double Differenced Data")

    # Calculate the standard deviations of the differenced data
    print("Standard Deviation (Actual Data):",
          np.std(data[24:]["total load actual"]))
    print("Standard Deviation (Adjusted Data):",
          np.std(data[24:]["adjusted"]))
    print("Standard Deviation (Non-Seasonally Differenced Data):",
          np.std(data[24:]["differenced"]))
    print("Standard Deviation (Seasonally Differenced Data):",
          np.std(data[24:]["seasonally differenced"]))
    print("Standard Deviation (Seasonally & Non-Seasonally Differenced Data):",
          np.std(data[25:]["double differenced"]))

    ax1.legend(loc="best")
    plt.show()


# TODO - finish this, doesn't work
def power_spectrum_plot(data):
    f, pxx_spec = signal.periodogram(data['total load actual'], fs=len(
        data['total load actual']), scaling='density')
    plt.figure()
    plt.semilogy(f[:50], np.sqrt(pxx_spec)[:50])
    plt.title('Power Spectrum (Welch\'s Method)')
    plt.show()


# Must have double differenced the data first before calling this function
def test_stationarity(data):
    result_original = adfuller(data["total load actual"], autolag='AIC')
    result_differenced = adfuller(data["seasonally differenced"][25:],
                                  autolag='AIC')
    print("Original Data")
    print("Test Statistic = {:.3f}".format(result_original[0]))  # The error
    # is a bug
    print("P-value = {:.3f}".format(result_original[1]))
    print("Critical values: ")
    for k, v in result_original[4].items():
        print(
            "\t{}: {:.4f} (The data is {}stationary with {}% "
            "confidence)".format(
                k, v, "not " if v < result_original[0] else "", 100 - int(k[
                                                                          :-1])))

    print("\nSeasonally Differenced Data")
    print("Test Statistic = {:.3f}".format(result_differenced[0]))  # The error
    # is a bug
    print("P-value = {:.3f}".format(result_differenced[1]))
    print("Critical values: ")
    for k, v in result_differenced[4].items():
        print(
            "\t{}: {:.4f} (The data is {}stationary with {}% "
            "confidence)".format(
                k, v, "not " if v < result_differenced[0] else "",
                100 - int(k[
                          :-1])))

def decompose(data, model):
    return seasonal_decompose(data['total load actual'],model=model, freq=24)


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
    axes[4].plot(decomp.observed - decomp.seasonal - decomp.trend if model ==
                                                                     "additive"
                 else (decomp.observed / decomp.seasonal) - decomp.trend)

    # Add titles
    axes[0].set_title("Observed Data")
    axes[1].set_title("Trend Component")
    axes[2].set_title("Seasonal Component")
    axes[3].set_title("Residual Component")
    axes[4].set_title("Seasonally and Trend Adjusted Data")

    # Show the figure
    plt.show()


def adjust(data, train_hours, test_hours, model):
    data.index = pd.to_datetime(data.index, utc=True)
    decomp = seasonal_decompose(
        data['total load actual'][0:train_hours + test_hours], model=model,
        freq=24)
    data[
        'adjusted'] = decomp.observed - decomp.seasonal - decomp.trend if model == \
                                                                          "additive" else \
        (decomp.observed / decomp.seasonal) - decomp.trend


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


# Calculate a seasonal index for each hour, and then divide by the seasonal
# index to get an adjusted value
def adjust_and_index(data, train_hours, test_hours):
    data['moving average'] = data['total load actual'].rolling(24).mean()
    data['seasonal ratio'] = data['total load actual'] / data['moving average']
    seasonal_indices = []
    for i in range(24):
        subset = data['seasonal ratio'][i::24]
        seasonal_indices.append(subset.mean())

    data['seasonal indices'] = seasonal_indices * int(
        (train_hours + test_hours) / 24
    )
    data['seasonally adjusted'] = data['total load actual'] / data[
        'seasonal indices'
    ]
    # Figure for the the seasonally-adjusted forecasts
    fig = plt.figure(figsize=(12.8, 9.6), dpi=250)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title("Seasonally Adjusted Data")

    # Plot the seasonally adjusted actual data
    ax.plot(data.index[0:train_hours],
            data['total load actual'][0:train_hours],
            label="Train Data")
    ax.plot(data.index[train_hours:],
            data['total load actual'][train_hours:],
            label="Test Data")

    # Plot the Naive2, SES, Holt and Damped Holt Difference Forecasts
    ax.plot(data.index, data['moving average'], label="Moving Average")
    ax.plot(data.index, data['seasonally adjusted'],
            label="Seasonally Adjusted")

    # Add the legend and show the plot
    ax.legend(loc="best")
    plt.show()


# Calculate the Mean Absolute Percentage Error of a prediction
def sMAPE(predicted, actual):
    act = actual.to_numpy()
    pred = predicted.to_numpy()
    mask = act != 0
    return (2 * np.fabs(act[mask] - pred[mask]) / (np.fabs(act[mask]) +
                                                   np.fabs(pred[mask]))).mean() \
           * 100


# Calculate the Root Mean Squared Error of a prediction
def RMSE(predicted, actual):
    act = actual.to_numpy()
    pred = predicted.to_numpy()
    return sqrt(np.sum(np.square(act - pred)) / len(actual))


# Calculate the Mean Average Scaled Error of a prediction
def MASE(predicted, actual, season):
    act = actual.to_numpy()
    pred = predicted.to_numpy()
    return np.fabs(
        ((act - pred) / np.fabs(act[season:] - act[:-season]).mean())).mean()


# Calculate the Mean Absolute Error of a prediction
def MAE(predicted, actual):
    act = actual.to_numpy()
    pred = predicted.to_numpy()
    return np.fabs(act - pred).mean()