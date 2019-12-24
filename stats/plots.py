import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pandas.plotting import autocorrelation_plot
from scipy import signal
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose

from stats.helpers import decomp_adjust, difference, seasonally_difference, \
    double_difference, indices_adjust


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
    decomp_adjust(data, train_hours, test_hours, "additive")
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
    ax1.plot(data.index, data['seasonally decomposed'], label="Adjusted")
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


def decomp_adjusted_plots(data, model):
    data.index = pd.to_datetime(data.index, utc=True)

    # Decompose
    decomp = seasonal_decompose(data['total load actual'], model, freq=24)

    # Plot the decomposition
    fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(12.8, 9.6), dpi=250)
    axes[0].plot(decomp.observed)
    axes[1].plot(decomp.trend)
    axes[2].plot(decomp.seasonal)
    axes[3].plot(decomp.resid)
    axes[4].plot(
        decomp.observed - decomp.seasonal - decomp.trend if model == "additive"
        else (decomp.observed / decomp.seasonal) - decomp.trend
    )

    # Add titles
    axes[0].set_title("Observed Data")
    axes[1].set_title("Trend Component")
    axes[2].set_title("Seasonal Component")
    axes[3].set_title("Residual Component")
    axes[4].set_title("Seasonally and Trend Adjusted Data")


    # Show the figure
    plt.show()


# Plots of the seasonally-adjusted data
def indices_adjusted_plots(data, train_hours, test_hours):
    # Adjust and index the data
    indices_adjust(data, train_hours, test_hours)

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