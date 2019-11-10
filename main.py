import pandas as pd
from pandas.plotting import register_matplotlib_converters
import matplotlib.pyplot as plt
import matplotlib.style
from stats import helpers
from stats import naive1


def load_data(filename):
    return pd.read_csv(filename, parse_dates=["time"], usecols=["time",
                                                                "total load "
                                                                "actual"])


def main():

    data = load_data("/Users/matt/Projects/AdvancedResearchProject/data/spain/"
                     "energy_dataset.csv")
    data.set_index('time', inplace=True)
    head = data.head(96)

    forecast = naive1.forecast(head)
    sma = helpers.sma(head, 24)

    print("Head \n", head, "Forecast \n", forecast, "SMA \n", sma)

    plt.xticks(rotation=45)
    head.plot()
    forecast.plot()
    sma.plot()
    plt.show()


if __name__ == "main":
    matplotlib.style.use('seaborn-deep')
    register_matplotlib_converters()
    main()

# Entry point when using PyCharm - REMOVE
matplotlib.style.use('seaborn-deep')
register_matplotlib_converters()
main()

