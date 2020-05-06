from matplotlib import pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller

from stats.stats_helpers import split_data, deseasonalise


# Plot actual and deseasonalised data, ACF, PACF for all season in all years
def analyse(df):
    all_data = split_data(df)

    for season in ["Winter", "Spring", "Summer", "Autumn"]:
        years = all_data[season]

        fig, axes = plt.subplots(2, 2, figsize=(20, 15), dpi=250)
        plt.suptitle(season + " - Data", y=0.99)
        for i, (year, ax) in enumerate(zip(years, axes.flatten())):
            deseason, ind, = deseasonalise(
                year["total load actual"], 168, "multiplicative"
            )

            ax.plot(year, label="Actual")
            ax.plot(deseason, label="Deseasonalised")
            ax.set_title("Year " + str(i + 1))
            ax.set_xticks([])
            ax.legend(loc="best")

            adf = adfuller(deseason, autolag='AIC')
            print("Original Data")
            print("Test Statistic (rounded) = {:.3f}".format(adf[0]))
            print("P-value (rounded) = {:.3f}".format(adf[1]))
            print("Critical values: ")
            for k, v in adf[4].items():
                print("\t{}: {:.4f} (The data is {}stationary with {}% "
                      "confidence)".format(
                    k, v, "not " if v < adf[0] else "",
                    100 - int(k[
                              :-1])))
            print()
        print()
        plt.show()

        # Plot Data ACFs
        fig, axes = plt.subplots(2, 2, figsize=(20, 15), dpi=250)
        plt.suptitle(season + " - ACFs (Actual)", y=0.99)
        for i, (year, ax) in enumerate(zip(years, axes.flatten())):
            plot_acf(year["total load actual"], ax=ax, alpha=0.05, lags=168)

        plt.show()

        # Plot Data PACFs
        fig, axes = plt.subplots(2, 2, figsize=(20, 15), dpi=250)
        plt.suptitle(season + " - PACFs (Actual)", y=0.99)
        for i, (year, ax) in enumerate(zip(years, axes.flatten())):
            plot_pacf(year["total load actual"], ax=ax, alpha=0.05, lags=168)

        plt.show()

        # Plot Deseasonalised ACFs
        fig, axes = plt.subplots(2, 2, figsize=(20, 15), dpi=250)
        plt.suptitle(season + " - ACFs (Deseasonalised)", y=0.99)
        for i, (year, ax) in enumerate(zip(years, axes.flatten())):
            deseason, _ = deseasonalise(
                year["total load actual"], 168, "multiplicative"
            )
            plot_acf(deseason, ax=ax, alpha=0.05, lags=168)

        plt.show()

        # Plot Deseasonalised PACFs
        fig, axes = plt.subplots(2, 2, figsize=(20, 15), dpi=250)
        plt.suptitle(season + " - PACFs (Deseasonalised)", y=0.99)
        for i, (year, ax) in enumerate(zip(years, axes.flatten())):
            deseason, _ = deseasonalise(
                year["total load actual"], 168, "multiplicative"
            )
            plot_pacf(deseason, ax=ax, alpha=0.05, lags=168)

        plt.show()


# Plots a sample of the weeks in the different seasons
def plot_sample(df):
    rand_weeks = {'Winter': [[4, 1], [3, 1, 3], [4, 3, 1], [3, 1, 2]],
                  'Spring': [[1, 4, 3], [4, 2, 3], [3, 4, 1], [4, 4, 2]],
                  'Summer': [[1, 2, 3], [2, 4, 2], [3, 4, 3], [1, 2, 4]],
                  'Autumn': [[2, 4, 2], [2, 3, 3], [4, 3, 3], [2, 4, 4]]
                  }

    # Choose random week
    winter_weeks = [
        df.loc["2015-01-26 00:00:00+01:00": "2015-02-01 23:00:00+01:00"],
        df.loc["2015-02-02 00:00:00+01:00": "2015-02-08 23:00:00+01:00"],

        df.loc["2015-12-21 00:00:00+01:00": "2015-12-27 23:00:00+01:00"],
        df.loc["2016-01-04 00:00:00+01:00": "2016-01-10 23:00:00+01:00"],
        df.loc["2016-02-15 00:00:00+01:00": "2016-02-21 23:00:00+01:00"],

        df.loc["2016-12-26 00:00:00+01:00": "2017-01-01 23:00:00+01:00"],
        df.loc["2017-01-16 00:00:00+01:00": "2017-01-22 23:00:00+01:00"],
        df.loc["2017-02-06 00:00:00+01:00": "2017-02-12 23:00:00+01:00"],

        df.loc["2017-12-18 00:00:00+01:00": "2017-12-24 23:00:00+01:00"],
        df.loc["2018-01-01 00:00:00+01:00": "2018-01-07 23:00:00+01:00"],
        df.loc["2018-02-12 00:00:00+01:00": "2018-02-18 23:00:00+01:00"],
    ]

    spring_weeks = [
        df.loc["2015-03-02 00:00:00+01:00": "2015-03-08 23:00:00+01:00"],
        df.loc["2015-04-27 00:00:00+01:00": "2015-05-03 23:00:00+01:00"],
        df.loc["2015-05-18 00:00:00+01:00": "2015-05-24 23:00:00+01:00"],

        df.loc["2016-03-28 00:00:00+01:00": "2016-04-03 23:00:00+01:00"],
        df.loc["2016-04-11 00:00:00+01:00": "2016-04-17 23:00:00+01:00"],
        df.loc["2016-05-16 00:00:00+01:00": "2016-05-22 23:00:00+01:00"],

        df.loc["2017-03-20 00:00:00+01:00": "2017-03-26 23:00:00+01:00"],
        df.loc["2017-04-24 00:00:00+01:00": "2017-04-30 23:00:00+01:00"],
        df.loc["2017-05-01 00:00:00+01:00": "2017-05-07 23:00:00+01:00"],

        df.loc["2018-03-26 00:00:00+01:00": "2018-04-01 23:00:00+01:00"],
        df.loc["2018-04-23 00:00:00+01:00": "2018-04-29 23:00:00+01:00"],
        df.loc["2018-05-14 00:00:00+01:00": "2018-05-20 23:00:00+01:00"],
    ]

    summer_weeks = [
        df.loc["2015-06-01 00:00:00+01:00": "2015-06-07 23:00:00+01:00"],
        df.loc["2015-07-06 00:00:00+01:00": "2015-07-12 23:00:00+01:00"],
        df.loc["2015-08-17 00:00:00+01:00": "2015-08-23 23:00:00+01:00"],

        df.loc["2016-06-13 00:00:00+01:00": "2016-06-19 23:00:00+01:00"],
        df.loc["2016-07-25 00:00:00+01:00": "2016-07-31 23:00:00+01:00"],
        df.loc["2016-08-08 00:00:00+01:00": "2016-08-14 23:00:00+01:00"],

        df.loc["2017-06-19 00:00:00+01:00": "2017-06-25 23:00:00+01:00"],
        df.loc["2017-07-24 00:00:00+01:00": "2017-07-30 23:00:00+01:00"],
        df.loc["2017-08-21 00:00:00+01:00": "2017-08-27 23:00:00+01:00"],

        df.loc["2018-06-04 00:00:00+01:00": "2018-06-10 23:00:00+01:00"],
        df.loc["2018-07-09 00:00:00+01:00": "2018-07-15 23:00:00+01:00"],
        df.loc["2018-08-27 00:00:00+01:00": "2018-09-02 23:00:00+01:00"],
    ]

    autumn_weeks = [
        df.loc["2015-09-14 00:00:00+01:00": "2015-09-20 23:00:00+01:00"],
        df.loc["2015-10-26 00:00:00+01:00": "2015-11-01 23:00:00+01:00"],
        df.loc["2015-11-09 00:00:00+01:00": "2015-11-15 23:00:00+01:00"],

        df.loc["2016-09-12 00:00:00+01:00": "2016-09-18 23:00:00+01:00"],
        df.loc["2016-10-17 00:00:00+01:00": "2016-10-23 23:00:00+01:00"],
        df.loc["2016-11-21 00:00:00+01:00": "2016-11-27 23:00:00+01:00"],

        df.loc["2017-09-25 00:00:00+01:00": "2017-10-01 23:00:00+01:00"],
        df.loc["2017-10-16 00:00:00+01:00": "2017-10-22 23:00:00+01:00"],
        df.loc["2017-11-20 00:00:00+01:00": "2017-11-26 23:00:00+01:00"],

        df.loc["2018-09-10 00:00:00+01:00": "2018-09-16 23:00:00+01:00"],
        df.loc["2017-10-16 00:00:00+01:00": "2017-10-22 23:00:00+01:00"],
        df.loc["2018-11-26 00:00:00+01:00": "2018-12-02 23:00:00+01:00"],
    ]

    # Plot winter weeks
    fig, axes = plt.subplots(4, 3, figsize=(20, 15), dpi=250)
    axes.flatten()[0].set_ylabel("Year 1")
    for i, (ax, week) in enumerate(zip(axes.flatten()[1:], winter_weeks)):
        if i == 2:
            ax.set_title("December")
            ax.set_ylabel("Year 2")
        if i == 0:
            ax.set_title("January")
        if i == 1:
            ax.set_title("February")
        if i == 5:
            ax.set_ylabel("Year 3")
        if i == 8:
            ax.set_ylabel("Year 4")
        ax.plot(week["total load actual"], color="C0")
        ax.set_xticks([])
    plt.show()

    # Plot spring weeks
    fig, axes = plt.subplots(4, 3, figsize=(20, 15), dpi=250)
    for i, (ax, week) in enumerate(zip(axes.flatten(), spring_weeks)):
        if i == 0:
            ax.set_title("March")
            ax.set_ylabel("Year 1")
        if i == 1:
            ax.set_title("April")
        if i == 2:
            ax.set_title("May")
        if i == 3:
            ax.set_ylabel("Year 2")
        if i == 6:
            ax.set_ylabel("Year 3")
        if i == 9:
            ax.set_ylabel("Year 4")
        ax.plot(week["total load actual"], color="C1")
        ax.set_xticks([])
    plt.show()

    # Plot summer weeks
    fig, axes = plt.subplots(4, 3, figsize=(20, 15), dpi=250)
    for i, (ax, week) in enumerate(zip(axes.flatten(), summer_weeks)):
        if i == 0:
            ax.set_title("June")
            ax.set_ylabel("Year 1")
        if i == 1:
            ax.set_title("July")
        if i == 2:
            ax.set_title("August")
        if i == 3:
            ax.set_ylabel("Year 2")
        if i == 6:
            ax.set_ylabel("Year 3")
        if i == 9:
            ax.set_ylabel("Year 4")
        ax.plot(week["total load actual"], color="C2")
        ax.set_xticks([])
    plt.show()

    # Plot autumn weeks
    fig, axes = plt.subplots(4, 3, figsize=(20, 15), dpi=250)
    for i, (ax, week) in enumerate(zip(axes.flatten(), autumn_weeks)):
        if i == 0:
            ax.set_title("September")
            ax.set_ylabel("Year 1")
        if i == 1:
            ax.set_title("October")
        if i == 2:
            ax.set_title("November")
        if i == 3:
            ax.set_ylabel("Year 2")
        if i == 6:
            ax.set_ylabel("Year 3")
        if i == 9:
            ax.set_ylabel("Year 4")
        ax.plot(week["total load actual"], color="C3")
        ax.set_xticks([])
    plt.show()


# Plot an example week from each season
def typical_plot(df):
    # Typical weeks
    font = {'size': 20}
    plt.rc('font', **font)
    seas_dict = {0: "Winter", 1: "Spring", 2: "Summer", 3: "Autumn"}
    years = [df.loc["2015-12-01 00:00:00+01:00":"2016-02-29 23:00:00+01:00"],
             df.loc["2015-03-01 00:00:00+01:00":"2015-05-31 23:00:00+01:00"],
             df.loc["2018-06-01 00:00:00+01:00":"2018-08-31 23:00:00+01:00"],
             df.loc["2017-09-01 00:00:00+01:00":"2017-11-30 23:00:00+01:00"]
             ]
    weeks = [df.loc["2017-01-16 00:00:00+01:00": "2017-01-22 23:00:00+01:00"],
             df.loc["2016-05-16 00:00:00+01:00": "2016-05-22 23:00:00+01:00"],
             df.loc["2017-07-24 00:00:00+01:00": "2017-07-30 23:00:00+01:00"],
             df.loc["2017-10-16 00:00:00+01:00": "2017-10-22 23:00:00+01:00"]]

    # Plot data
    fig, axes = plt.subplots(2, 2, figsize=(20, 15), dpi=250)
    for i, (week, ax) in enumerate(zip(weeks, axes.flatten())):
        ax.plot(week)
        ax.set_title(seas_dict[i])
        ax.set_xticks([])
    plt.show()

    # Plot ACF
    fig, axes = plt.subplots(2, 2, figsize=(20, 15), dpi=250)
    for i, (year, ax) in enumerate(zip(years, axes.flatten())):
        plot_acf(year, ax=ax, alpha=0.05, lags=168)
        ax.set_title(seas_dict[i])
    plt.show()

    # Plot PACF
    fig, axes = plt.subplots(2, 2, figsize=(20, 15), dpi=250)
    for i, (year, ax) in enumerate(zip(years, axes.flatten())):
        plot_pacf(year, ax=ax, alpha=0.05, lags=168)
        ax.set_title(seas_dict[i])
    plt.show()


def train_test_split(df):
    font = {'size': 20}
    plt.rc('font', **font)
    fig, axes = plt.subplots(3, 1, figsize=(20, 15), dpi=250)
    axes[0].plot(
        df.loc["2015-01-01 00:00:00+01:00":"2015-02-13 23:00:00+01:00"],
        label="Training"
    )
    axes[0].plot(
        df.loc["2015-02-14 00:00:00+01:00":"2015-02-20 23:00:00+01:00"],
        label="Validation", color="C2"
    )
    axes[0].plot(
        df.loc["2015-02-21 00:00:00+01:00":"2015-02-28 23:00:00+01:00"],
        label="Test", color="C1"
    )

    axes[1].plot(
        df.loc["2015-01-01 00:00:00+01:00":"2015-02-20 23:00:00+01:00"],
        label="Test 1 - Training"
    )
    axes[1].plot(
        df.loc["2015-02-21 00:00:00+01:00":"2015-02-22 23:00:00+01:00"],
        label="Test 1 - Test"
    )
    axes[1].plot(
        df.loc["2015-02-23 00:00:00+01:00":"2015-02-28 23:00:00+01:00"],
        label="Test 1 - Unused",
        color="#7f7f7f"
    )

    axes[2].plot(
        df.loc["2015-01-01 00:00:00+01:00":"2015-02-24 23:00:00+01:00"],
        label="Test 5 - Training"
    )
    axes[2].plot(
        df.loc["2015-02-25 00:00:00+01:00":"2015-02-26 23:00:00+01:00"],
        label="Test 5 - Test"
    )
    axes[2].plot(
        df.loc["2015-02-27 00:00:00+01:00":"2015-02-28 23:00:00+01:00"],
        label="Test 5 - Unused",
        color="#7f7f7f"
    )

    axes[0].set_title("Year 1 - Winter - Training/Validation/Test Split")
    axes[1].set_title("Year 1 - Winter - Test 1")
    axes[2].set_title("Year 1 - Winter - Test 5")
    axes[0].set_xticks([])
    axes[1].set_xticks([])
    axes[2].set_xticks([])
    axes[0].legend(loc="best")
    axes[1].legend(loc="best")
    axes[2].legend(loc="best")
    plt.show()


# Plot an exaple test/training data split, for two train periods
def plot_a_season(df, season):
    split = split_data(df)

    for y in split[season]:
        # Plot first quarter
        fig, ax = plt.subplots(1, 1, figsize=(20, 15), dpi=250)
        ax.plot(y["total load actual"][:int(len(y["total load actual"])/4.0)])
        plt.show()

        # Plot second quarter
        fig, ax = plt.subplots(1, 1, figsize=(20, 15), dpi=250)
        ax.plot(y["total load actual"][int(len(y["total load actual"]) /
                                           4.0):int(len(y["total load "
                                                         "actual"]) / 2.0)])
        plt.show()

        # Plot third quarter
        fig, ax = plt.subplots(1, 1, figsize=(20, 15), dpi=250)
        ax.plot(y["total load actual"][int(len(y["total load actual"]) /
                                           2.0):int(3.0 * len(y["total load "
                                                          "actual"]) / 4.0)])
        plt.show()

        # Plot fourth quarter
        fig, ax = plt.subplots(1, 1, figsize=(20, 15), dpi=250)
        ax.plot(y["total load actual"][int(3.0 * len(y["total load actual"])
                                           / 4):])
        plt.show()


def holt_winters_test(df):
    data = split_data(df)["Summer"][2]["total load actual"]
    train_data = data[:-48]
    test_data = data[-48:]

    # [alpha, beta, gamma, l0, b0, phi, s0,.., s_(m - 1)]
    _, indices = deseasonalise(train_data, 168, "multiplicative")
    init_params = [0.25, 0.75, train_data[0]]
    init_params.extend(indices)

    fitted_model = ExponentialSmoothing(
        train_data, seasonal_periods=168, seasonal="mul"
    ).fit(use_basinhopping=True, start_params=init_params)
    init_prediction = fitted_model.predict(0, len(train_data) + 48 - 1)
    params = fitted_model.params
    print(params)

    fitted_model = ExponentialSmoothing(
        train_data, seasonal_periods=168, seasonal="mul"
    ).fit(use_basinhopping=True)
    prediction = fitted_model.predict(0, len(train_data) + 48 - 1)
    params = fitted_model.params
    print(params)

    fig, ax = plt.subplots(1, 1, figsize=(20, 15), dpi=250)
    ax.plot(test_data, label="Actual Data")
    ax.plot(prediction[-48:], label="Non initialised")
    ax.plot(init_prediction[-48:], label="Initialised")
    ax.legend(loc="best")
    plt.show()


# Plot the deseasonalised data for 24- and 168-seasonality
def deseason(df):
    year = df.loc["2015-01-01 00:00:00+01:00":"2015-02-28 23:00:00+01:00"]
    year_24, _ = deseasonalise(
        year["total load actual"], 24, "multiplicative"
    )
    year_168, _ = deseasonalise(
        year["total load actual"], 168, "multiplicative"
    )

    # Create figure
    font = {'size': 20}
    plt.rc('font', **font)
    fig = plt.figure(figsize=(20, 15), dpi=250)
    gs = fig.add_gridspec(2, 2)
    ax_1 = fig.add_subplot(gs[0, :])
    ax_2 = fig.add_subplot(gs[1, 0])
    ax_3 = fig.add_subplot(gs[1, 1])

    # Plot data
    ax_1.plot(year)
    ax_2.plot(year_24)
    ax_3.plot(year_168)

    # Add weekend highlighting
    ax_1.axvspan("2015-01-03 00:00:00+01:00",
                 "2015-01-04 23:00:00+01:00", alpha=0.1)
    ax_1.axvspan("2015-01-10 00:00:00+01:00",
                 "2015-01-11 23:00:00+01:00", alpha=0.1)
    ax_1.axvspan("2015-01-17 00:00:00+01:00",
                 "2015-01-18 23:00:00+01:00", alpha=0.1)
    ax_1.axvspan("2015-01-24 00:00:00+01:00",
                 "2015-01-25 23:00:00+01:00", alpha=0.1)
    ax_1.axvspan("2015-01-31 00:00:00+01:00",
                 "2015-02-01 23:00:00+01:00", alpha=0.1)
    ax_1.axvspan("2015-02-07 00:00:00+01:00",
                 "2015-02-08 23:00:00+01:00", alpha=0.1)
    ax_1.axvspan("2015-02-14 00:00:00+01:00",
                 "2015-02-15 23:00:00+01:00", alpha=0.1)
    ax_1.axvspan("2015-02-21 00:00:00+01:00",
                 "2015-02-22 23:00:00+01:00", alpha=0.1)
    ax_1.axvspan("2015-02-28 00:00:00+01:00",
                 "2015-02-28 23:00:00+01:00", alpha=0.1)

    ax_2.axvspan("2015-01-03 00:00:00+01:00",
                 "2015-01-04 23:00:00+01:00", alpha=0.1)
    ax_2.axvspan("2015-01-10 00:00:00+01:00",
                 "2015-01-11 23:00:00+01:00", alpha=0.1)
    ax_2.axvspan("2015-01-17 00:00:00+01:00",
                 "2015-01-18 23:00:00+01:00", alpha=0.1)
    ax_2.axvspan("2015-01-24 00:00:00+01:00",
                 "2015-01-25 23:00:00+01:00", alpha=0.1)
    ax_2.axvspan("2015-01-31 00:00:00+01:00",
                 "2015-02-01 23:00:00+01:00", alpha=0.1)
    ax_2.axvspan("2015-02-07 00:00:00+01:00",
                 "2015-02-08 23:00:00+01:00", alpha=0.1)
    ax_2.axvspan("2015-02-14 00:00:00+01:00",
                 "2015-02-15 23:00:00+01:00", alpha=0.1)
    ax_2.axvspan("2015-02-21 00:00:00+01:00",
                 "2015-02-22 23:00:00+01:00", alpha=0.1)
    ax_2.axvspan("2015-02-28 00:00:00+01:00",
                 "2015-02-28 23:00:00+01:00", alpha=0.1)

    ax_3.axvspan("2015-01-03 00:00:00+01:00",
                 "2015-01-04 23:00:00+01:00", alpha=0.1)
    ax_3.axvspan("2015-01-10 00:00:00+01:00",
                 "2015-01-11 23:00:00+01:00", alpha=0.1)
    ax_3.axvspan("2015-01-17 00:00:00+01:00",
                 "2015-01-18 23:00:00+01:00", alpha=0.1)
    ax_3.axvspan("2015-01-24 00:00:00+01:00",
                 "2015-01-25 23:00:00+01:00", alpha=0.1)
    ax_3.axvspan("2015-01-31 00:00:00+01:00",
                 "2015-02-01 23:00:00+01:00", alpha=0.1)
    ax_3.axvspan("2015-02-07 00:00:00+01:00",
                 "2015-02-08 23:00:00+01:00", alpha=0.1)
    ax_3.axvspan("2015-02-14 00:00:00+01:00",
                 "2015-02-15 23:00:00+01:00", alpha=0.1)
    ax_3.axvspan("2015-02-21 00:00:00+01:00",
                 "2015-02-22 23:00:00+01:00", alpha=0.1)
    ax_3.axvspan("2015-02-28 00:00:00+01:00",
                 "2015-02-28 23:00:00+01:00", alpha=0.1)

    # Add titles
    ax_1.set_xticks([])
    ax_1.set_title("Year 1 - Winter - Actual")
    ax_2.set_xticks([])
    ax_2.set_title("Hourly Seasonality Removed")
    ax_3.set_xticks([])
    ax_3.set_title("Weekly Seasonality Removed")
    plt.show()

    # Plot the ACF and PACF of the deseasonalised data
    fig, axes = plt.subplots(2, 1, figsize=(20, 15), dpi=250)
    plot_acf(year_168, axes[0], lags=168)
    plot_pacf(year_168, axes[1], lags=168)
    axes[0].set_title("Year 1 - Winter - Deseasonalised ACF")
    axes[1].set_title("Year 1 - Winter - Deseasonalised PACF")
    plt.show()


# Test stationarity - must double de-deseasonalise first
def test_stationarity(data):
    result_original = adfuller(data["total load actual"], autolag='AIC')
    result_differenced = adfuller(data["seasonally differenced"][25:],
                                  autolag='AIC')
    print("Original Data")
    print(
        "Test Statistic = {:.3f}".format(result_original[0]))  # The error
    # is a bug
    print("P-value = {:.3f}".format(result_original[1]))
    print("Critical values: ")
    for k, v in result_original[4].items():
        print(
            "\t{}: {:.4f} (The data is {}stationary with {}% "
            "confidence)".format(
                k, v, "not " if v < result_original[0] else "",
                100 - int(k[
                          :-1])))

    print("\nSeasonally Differenced Data")
    print("Test Statistic = {:.3f}".format(result_differenced[0]))
    print("P-value = {:.3f}".format(result_differenced[1]))
    print("Critical values: ")

    for k, v in result_differenced[4].items():
        print(
            "\t{}: {:.4f} (The data is {}stationary with {}% "
            "confidence)".format(
                k, v, "not " if v < result_differenced[0] else "",
                100 - int(k[:-1])
                )
            )