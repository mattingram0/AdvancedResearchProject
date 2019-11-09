import pandas as pd
from pandas.plotting import register_matplotlib_converters
import matplotlib.pyplot as plt
from stats import naive1


def load_data(filename):
    return pd.read_csv(filename, parse_dates=["time"])


def main():
    register_matplotlib_converters()
    data = load_data("/Users/matt/Projects/AdvancedResearchProject/data/spain/"
                     "energy_dataset.csv")

    head = data.head(24)[["time", "total load actual"]]
    head.set_index('time', inplace=True)

    forecast = naive1.forecast(head)

    plt.plot(head, color="blue")
    plt.plot(forecast, color="orange")
    plt.xticks(rotation=45)
    plt.show()


if __name__ == "main":
    main()

main()

