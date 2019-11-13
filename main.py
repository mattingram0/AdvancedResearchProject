import pandas as pd
from pandas.plotting import register_matplotlib_converters, \
    autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
import matplotlib.pyplot as plt
import matplotlib.style
from stats import helpers
from stats import naive1, naiveS


def load_data(filename):
    return pd.read_csv(filename, parse_dates=["time"], usecols=["time",
                                                                "total load "
                                                                "actual"])


def main():
    # Load and preprocess
    data = load_data("/Users/matt/Projects/AdvancedResearchProject/data/spain/"
                     "energy_dataset.csv")
    data.set_index('time', inplace=True)
    data = data.head(336)  # Using diff variable caused the weird errors
    data.interpolate(inplace=True)  # Interpolate missing values

    # Naive Forecast, and Simple Moving Average
    week1_naive1 = naive1.forecast(data.iloc[0:168])
    week2_naive1 = naive1.forecast(data.iloc[168:])
    week1_naiveS = naiveS.forecast(data.iloc[0:168])
    week2_naiveS = naiveS.forecast(data.iloc[168:])
    week1_sma = helpers.sma(data.iloc[0:168], 24)
    week2_sma = helpers.sma(data.iloc[168:], 24)

    # Create a figure with 6 (2 x 3) subplots, increase size and resolution,
    # add titles and rotate x-axis labels
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12.8, 9.6), dpi=250)
    fig.canvas.set_window_title("Visualisation and Autocorrelation")
    axes[0, 0].set_title("Week 1")
    axes[0, 1].set_title("Week 2")
    plt.xticks(rotation=45)

    # Plot week 1 data, forecast, and autocorrelation
    data.iloc[0:168].plot(ax=axes[0, 0])
    # week1_naive1.plot(ax=axes[0, 0])
    week1_naiveS.plot(ax=axes[0, 0])
    # week1_sma.plot(ax=axes[0, 0])
    autocorrelation_plot(data.iloc[0:168], ax=axes[1, 0])

    # Plot week 2 data, forecast and autocorrelation
    data.iloc[168:].plot(ax=axes[0, 1])
    # week2_naive1.plot(ax=axes[0, 1])
    week2_naiveS.plot(ax=axes[0, 1])
    # week2_sma.plot(ax=axes[0, 1])
    autocorrelation_plot(data.iloc[168:], ax=axes[1, 1])

    # Plot the ACF and PACF
    plot_acf(data.iloc[0:168], ax=axes[0, 2], alpha=0.1, lags=50)
    plot_acf(data.iloc[168:], ax=axes[1, 2], alpha=0.1, lags=50)

    # Show the plot
    plt.show()


if __name__ == "main":
    matplotlib.style.use('seaborn-deep')
    register_matplotlib_converters()
    # pd.options.mode.chained_assignment = None TURN OFF CHAINED ASSIGNMENTS
    # WARNING - possibly use in the future if I run into that message again
    main()

# Entry point when using PyCharm - REMOVE
matplotlib.style.use('seaborn-deep')
register_matplotlib_converters()
main()

