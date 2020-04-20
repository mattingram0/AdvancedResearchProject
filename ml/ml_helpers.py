import numpy as np
from numpy.polynomial.polynomial import polyfit
import matplotlib.pyplot as plt
import pandas as pd
import torch
import json
from math import sin, exp
from os import walk


def weather_analysis():
    weather_file = "/Users/matt/Projects/AdvancedResearchProject/data/spain/" \
                   "weather_features.csv"
    demand_file = "/Users/matt/Projects/AdvancedResearchProject/data/spain/" \
             "energy_dataset.csv"
    weather_list = ["dt_iso", "city_name", "temp", "humidity", "wind_speed"]
    demand_list = ["time", "total load actual"]
    weather = pd.read_csv(weather_file, parse_dates=["dt_iso"],
                     infer_datetime_format=True,
                     usecols=weather_list)
    demand = pd.read_csv(demand_file, parse_dates=["time"],
                          infer_datetime_format=True,
                          usecols=demand_list)

    # Remove duplicates in the weather data
    weather = weather.drop_duplicates(["dt_iso", "city_name"])

    # Replace 0s to NaNs and interpolate the missing values in demand data
    demand.replace(0, np.NaN, inplace=True)
    demand.interpolate(inplace=True)

    # Reference temperature (C) for latent enthalpy calculation
    ref_temp = 25.6

    # Calculate the HD and CD for all
    weather["HD"] = weather["temp"].apply(
        lambda x: 288.65 - x if x < 288.65 else 0
    )
    weather["CD"] = weather["temp"].apply(
        lambda x: x - 296.5 if x > 296.5 else 0
    )
    weather["LE"] = weather.apply(
        lambda row: latent_enthalpy(row, ref_temp), axis=1
    )

    # Average weather across all cities
    avg_weather = weather.groupby(["dt_iso"]).mean()

    # Individual city weather data
    val = weather[weather["city_name"] == "Valencia"]
    mad = weather[weather["city_name"] == "Madrid"]
    bar = weather[weather["city_name"] == " Barcelona"]
    sev = weather[weather["city_name"] == "Seville"]
    bil = weather[weather["city_name"] == "Bilbao"]
    cities = [val, mad] #, bar, sev, bil]
    city_color_map = {"Valencia": "C1", "Madrid": "C2", "Barcelona": "C3",
                      "Seville": "C4", "Bilbao": "C5"}

    # Plot of average temperature v total load actual
    fig, ax = plt.subplots(1, 1, figsize=(20, 15), dpi=250)
    ax.scatter(avg_weather["temp"], demand["total load actual"])
    ax.set_title("Average: Demand against Temperature")
    b, m, m2 = polyfit(avg_weather["temp"], demand["total load actual"], 2)
    x = np.arange(
        np.floor(avg_weather["temp"].min()),
        np.ceil(avg_weather["temp"].max()) + 1
    )
    ax.plot(x, b + m * x + m2 * x**2, color="#fc8403")
    plt.show()

    fig, ax = plt.subplots(1, 1, figsize=(20, 15), dpi=250)
    ax.scatter(avg_weather["HD"], demand["total load actual"])
    ax.set_title("Average: Demand against Heating Degree")
    b, m = polyfit(avg_weather["HD"], demand["total load actual"], 1)
    x = np.arange(np.ceil(avg_weather["HD"].max()) + 1)
    ax.plot(x, b + m * x, color="#fc8403")
    plt.show()

    fig, ax = plt.subplots(1, 1, figsize=(20, 15), dpi=250)
    ax.scatter(avg_weather["CD"], demand["total load actual"])
    ax.set_title("Average: Demand against Cooling Degree")
    b, m = polyfit(avg_weather["CD"], demand["total load actual"], 1)
    x = np.arange(np.ceil(avg_weather["CD"].max()) + 1)
    ax.plot(x, b + m * x, color="#fc8403")
    plt.show()

    fig, ax = plt.subplots(1, 1, figsize=(20, 15), dpi=250)
    ax.scatter(avg_weather["LE"], demand["total load actual"])
    ax.set_title("Average: Demand against Latent Enthalpy")
    b, m = polyfit(avg_weather["LE"], demand["total load actual"], 1)
    x = np.arange(np.ceil(avg_weather["LE"].max()) + 1)
    ax.plot(x, b + m * x, color="#fc8403")
    plt.show()

    # TODO CHECK WORKS Notes: replace values over 2500 with mean
    # Not worth including
    # avg_weather.loc[avg_weather["pressure"] > 2500, "pressure"] = \
    #     avg_weather["pressure"].mean()
    # avg_weather.loc[avg_weather["pressure"] < 950, "pressure"] = \
    #     avg_weather["pressure"].mean()
    # fig, ax = plt.subplots(1, 1, figsize=(20, 15), dpi=250)
    # ax.scatter(avg_weather["pressure"], demand["total load actual"])
    # ax.set_title("Average: Demand against Pressure")
    # b, m = polyfit(avg_weather["pressure"], demand["total load actual"], 1)
    # x = np.arange(np.floor(avg_weather["pressure"].min()), np.ceil(
    #     avg_weather["pressure"].max()) + 1)
    # ax.plot(x, b + m * x, color="#fc8403")
    # plt.show()

    fig, ax = plt.subplots(1, 1, figsize=(20, 15), dpi=250)
    ax.scatter(avg_weather["humidity"], demand["total load actual"])
    ax.set_title("Average: Demand against Humidity")
    b, m = polyfit(avg_weather["humidity"], demand["total load actual"], 1)
    x = np.arange(
        np.floor(avg_weather["humidity"].min()),
        np.ceil(avg_weather["humidity"].max()) + 1
    )
    ax.plot(x, b + m * x, color="#fc8403")
    plt.show()

    # Set one value over 20 to the mean. Seems to be worth including
    avg_weather.loc[avg_weather["wind_speed"] > 20, "wind_speed"] = \
        avg_weather["wind_speed"].mean()
    fig, ax = plt.subplots(1, 1, figsize=(20, 15), dpi=250)
    ax.scatter(avg_weather["wind_speed"], demand["total load actual"])
    ax.set_title("Average: Demand against Wind Speed")
    b, m = polyfit(avg_weather["wind_speed"], demand["total load actual"], 1)
    x = np.arange(np.ceil(avg_weather["wind_speed"].max()) + 1)
    ax.plot(x, b + m * x, color="#fc8403")
    plt.show()

    # Of little use
    # fig, ax = plt.subplots(1, 1, figsize=(20, 15), dpi=250)
    # ax.scatter(avg_weather["rain_1h"], demand["total load actual"])
    # ax.set_title("Average: Demand against Rain 1h")
    # b, m = polyfit(avg_weather["rain_1h"], demand["total load actual"], 1)
    # x = np.arange(np.ceil(avg_weather["rain_1h"].max()) + 1)
    # ax.plot(x, b + m * x, color="#fc8403")
    # plt.show()

    # Remove the one value of 0.4 and replot to see the trend. Probably of
    # little use
    # avg_weather.loc[avg_weather["rain_3h"] > 0.35, "rain_3h"] = \
    #     avg_weather["rain_3h"].mean()
    # fig, ax = plt.subplots(1, 1, figsize=(20, 15), dpi=250)
    # ax.scatter(avg_weather["rain_3h"], demand["total load actual"])
    # ax.set_title("Average: Demand against Rain 3h")
    # b, m = polyfit(avg_weather["rain_3h"], demand["total load actual"], 1)
    # # x = np.arange(np.ceil(avg_weather["rain_3h"].max()) + 1)
    # x = np.linspace(0, 0.05, 20)
    # ax.plot(x, b + m * x, color="#fc8403")
    # plt.show()

    # So little snow that this is of little value
    # fig, ax = plt.subplots(1, 1, figsize=(20, 15), dpi=250)
    # ax.scatter(avg_weather["snow_3h"], demand["total load actual"])
    # ax.set_title("Average: Demand against Snow 3h")
    # b, m = polyfit(avg_weather["snow_3h"], demand["total load actual"], 1)
    # x = np.arange(np.ceil(avg_weather["snow_3h"].max()) + 1)
    # ax.plot(x, b + m * x, color="#fc8403")
    # plt.show()

    # Little point in using, no correlation really
    # fig, ax = plt.subplots(1, 1, figsize=(20, 15), dpi=250)
    # ax.scatter(avg_weather["clouds_all"], demand["total load actual"])
    # ax.set_title("Average: Demand against Cloud Cover")
    # b, m = polyfit(avg_weather["clouds_all"], demand["total load actual"], 1)
    # x = np.arange(np.ceil(avg_weather["clouds_all"].max()) + 1)
    # ax.plot(x, b + m * x, color="#fc8403")
    # plt.show()

    # Plots of temperature v total load actual for each city
    # for c in cities:
    #     c = c.set_index('dt_iso').asfreq('H')
    #     name = c.iloc[0]["city_name"].strip()
    #     color = city_color_map[name]
    #
    #     fig, ax = plt.subplots(1, 1, figsize=(20, 15), dpi=250)
    #     ax.scatter(c["temp"], demand["total load actual"], color=color)
    #     ax.set_title(name + ": Demand against Temperature")
    #     plt.show()
    #
    #     fig, ax = plt.subplots(1, 1, figsize=(20, 15), dpi=250)
    #     ax.scatter(c["HD"], demand["total load actual"], color=color)
    #     ax.set_title(name + ": Demand against Heating Degree")
    #     plt.show()
    #
    #     fig, ax = plt.subplots(1, 1, figsize=(20, 15), dpi=250)
    #     ax.scatter(c["CD"], demand["total load actual"], color=color)
    #     ax.set_title(name + ": Demand against Cooling Degree")
    #     plt.show()
    #
    #     fig, ax = plt.subplots(1, 1, figsize=(20, 15), dpi=250)
    #     ax.scatter(c["LE"], demand["total load actual"], color=color)
    #     ax.set_title(name + ": Demand against Latent Enthalpy")
    #     b, m = polyfit(c["LE"], demand["total load actual"], 1)
    #     x = np.arange(np.ceil(c["LE"].max()) + 1)
    #     ax.plot(x, b + m * x, color="#fc8403")
    #     plt.show()


def plot_lr():
    init = True

    init_fp = "/Users/matt/Projects/AdvancedResearchProject/test" \
              "/lr_test_init_year_1_summer_0001.txt"
    ninit_fp = "/Users/matt/Projects/AdvancedResearchProject/test" \
               "/lr_test_ninit_year_1_summer_0001.txt"

    # file_name = "/Users/matt/Projects/AdvancedResearchProject/test" \
    #     "/final_test_year_1_summer_01.txt"
    # local_labels = [0.01 for _ in range(10)] + [0.005 for _ in range(10)] + [
    #     0.001 for _ in range(10)] + [0.0005 for _ in range(5)]
    # global_labels = [0.005 for _ in range(10)] + [0.001 for _ in range(10)] + [
    #     0.0005 for _ in range(10)] + [0.0001 for _ in range(5)]
    # x = [i for i in range(0, 35, 2)]

    with open(init_fp if init else ninit_fp) as f:
        results = json.load(f)

    lr = [0.0001 * i for i in range(1, 10) for _ in range(3)] + \
         [0.001 * i for i in range(1, 10) for _ in range(3)] + \
         [0.01 * i for i in range(1, 10) for _ in range(3)] + \
         [0.1 * i for i in range(1, 10) for _ in range(3)]
    labels = [np.round(lr[i], decimals=4) for i in range(0, len(lr), 5)]
    x = [i for i in range(0, len(lr), 5)]

    # ---------- PLOT ALL EPOCHS (LRs) ----------
    font = {'size': 20}
    plt.rc('font', **font)
    fig = plt.figure(figsize=(20, 15), dpi=250)
    plt.suptitle("Initialised" if init else "Uninitialised")
    gs = fig.add_gridspec(3, 2)
    ax_1 = fig.add_subplot(gs[0, :])
    ax_2 = fig.add_subplot(gs[1, 0])
    ax_3 = fig.add_subplot(gs[1, 1])
    ax_4 = fig.add_subplot(gs[2, 0])
    ax_5 = fig.add_subplot(gs[2, 1])
    axes = [ax_2, ax_3, ax_4, ax_5]

    losses = results["7"]["losses"]
    rnn = losses["total load actual"]["RNN"]

    ax_1.plot(rnn, label="RNN Losses")
    ax_1.legend(loc="best")
    ax_1.set_xticks(x)
    ax_1.set_xticklabels(labels, rotation=45)

    for i, k in enumerate(losses.keys()):
        lvp = losses[k]["LVP"]
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(labels, rotation=45)
        axes[i].plot(lvp, label=k + " LVP")
        axes[i].legend(loc="best")

    plt.show()

    # ---------- PLOT ONLY FIRST 90 EPOCHS (LRs) ----------
    font = {'size': 20}
    plt.rc('font', **font)
    fig = plt.figure(figsize=(20, 15), dpi=250)
    plt.suptitle("Initialised" if init else "Uninitialised")
    gs = fig.add_gridspec(3, 2)
    ax_1 = fig.add_subplot(gs[0, :])
    ax_2 = fig.add_subplot(gs[1, 0])
    ax_3 = fig.add_subplot(gs[1, 1])
    ax_4 = fig.add_subplot(gs[2, 0])
    ax_5 = fig.add_subplot(gs[2, 1])
    axes = [ax_2, ax_3, ax_4, ax_5]

    losses = results["7"]["losses"]
    rnn = losses["total load actual"]["RNN"]

    ax_1.set_xticks(x)
    ax_1.set_xticklabels(labels, rotation=45)
    ax_1.plot(rnn[:95], label="RNN Losses")
    ax_1.legend(loc="best")

    for i, k in enumerate(losses.keys()):
        lvp = losses[k]["LVP"]
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(labels, rotation=45)
        axes[i].plot(lvp[:95], label=k + " LVP")
        axes[i].legend(loc="best")

    plt.show()

    # ---------- PLOT ONLY FIRST 80 EPOCHS (LRs) ----------
    font = {'size': 20}
    plt.rc('font', **font)
    fig = plt.figure(figsize=(20, 15), dpi=250)
    plt.suptitle("Initialised" if init else "Uninitialised")
    gs = fig.add_gridspec(3, 2)
    ax_1 = fig.add_subplot(gs[0, :])
    ax_2 = fig.add_subplot(gs[1, 0])
    ax_3 = fig.add_subplot(gs[1, 1])
    ax_4 = fig.add_subplot(gs[2, 0])
    ax_5 = fig.add_subplot(gs[2, 1])
    axes = [ax_2, ax_3, ax_4, ax_5]

    losses = results["7"]["losses"]
    rnn = losses["total load actual"]["RNN"]

    ax_1.set_xticks(x)
    ax_1.set_xticklabels(labels, rotation=45)
    ax_1.plot(rnn[:80], label="RNN Losses")
    ax_1.legend(loc="best")

    for i, k in enumerate(losses.keys()):
        lvp = losses[k]["LVP"]
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(labels, rotation=45)
        axes[i].plot(lvp[:80], label=k + " LVP")
        axes[i].legend(loc="best")

    plt.show()

    # ---------- PLOT ONLY FIRST 70 EPOCHS (LRs) ----------
    font = {'size': 20}
    plt.rc('font', **font)
    fig = plt.figure(figsize=(20, 15), dpi=250)
    plt.suptitle("Initialised" if init else "Uninitialised")
    gs = fig.add_gridspec(3, 2)
    ax_1 = fig.add_subplot(gs[0, :])
    ax_2 = fig.add_subplot(gs[1, 0])
    ax_3 = fig.add_subplot(gs[1, 1])
    ax_4 = fig.add_subplot(gs[2, 0])
    ax_5 = fig.add_subplot(gs[2, 1])
    axes = [ax_2, ax_3, ax_4, ax_5]

    losses = results["7"]["losses"]
    rnn = losses["total load actual"]["RNN"]

    ax_1.set_xticks(x)
    ax_1.set_xticklabels(labels, rotation=45)
    ax_1.plot(rnn[:70], label="RNN Losses")
    ax_1.legend(loc="best")

    for i, k in enumerate(losses.keys()):
        lvp = losses[k]["LVP"]
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(labels, rotation=45)
        axes[i].plot(lvp[:70], label=k + " LVP")
        axes[i].legend(loc="best")

    plt.show()


# Convert hPa to Pa
def hpa_to_pa(p):
    return 100 * p


# Convert kelvin to celcius
def k_to_c(t):
    return t - 273.15


# Convert celcius to kelvin
def c_to_k(t):
    return t + 273.15


# Calculate an approximation for the saturation vapor pressure for moist air
# using the Arden Buck equation (more optimised than the Goff-Gratch
# equation for -80C to 50C).
# Input: Temperature t in Celcius (C)
# Output: Saturation vapor pressire in Pascals (Pa)
# See: https://en.wikipedia.org/wiki/Arden_Buck_equation
def arden_buck(t):
    return 611.21 * exp((18.678 - t/234.5) * (t/(257.14 + t))) if t > 0 else\
        611.15 * exp((23.036 - t/333.7) * (t/(279.82 + t)))


# Calculate an approximation for the specific enthalpy using the above Arden
# Buck equation for the saturation vapor pressure for moist air. The formula is:
# h = Cpa * t + x * [Cpw * t + Hwe], where Cpa, Cpw and Hwe are constants, and
# x = 0.62198 * pw(t) / (pa - pw(t)) is the humidity ratio per mass, where:
# pa is the current pressure in Pascals, and
# pw(t) = r * ps(t) is the partial pressure of water vapor in moist air, where:
# r = relative humidity of the air (decimal), and
# ps(t) = saturation vapor pressure for the given temperature t, found using
# the arden_buck formula.
# See:
# https://www.engineeringtoolbox.com/enthalpy-moist-air-d_683.html
# https://www.engineeringtoolbox.com/water-vapor-saturation-pressure-air-d_689.html
# https://www.chegg.com/homework-help/determine-partial-pressure-water-vapor-moist-air-following-c-chapter-10-problem-3p-solution-9781305534094-exc
# Note that the formula given in this paper: https://www.sciencedirect.com/science/article/pii/S0378778815303492#eq0005
# is incorrect
# Used: https://www.psychrometric-calculator.com/humidairweb.aspx to check
# my implementation
# Input: Temperature in C, pressure in Pa, relative humidity ([0, 1])
# Output: Specific enthalpy in kilojoules per kilogram of dry hair (kJ/kg)
def specific_enthalpy(t, p, r):
    return 1.006 * t + (((0.62198 * r * arden_buck(t)) / (p - (r * arden_buck(
        t)))) * (2501 + (1.84 * t)))


# Receive a row of weather data (must include temperature, pressure and
# relative humidity) and calculate the latent enthalpy as defined in:
# https://ieeexplore.ieee.org/document/1525139
def latent_enthalpy(row, ref_temp):
    t = k_to_c(row["temp"])
    p = hpa_to_pa(row["pressure"])
    r = row["humidity"] / 100
    q = specific_enthalpy(t, p, r)
    qb = specific_enthalpy(ref_temp, p, r)

    return q - qb if t > ref_temp and q - qb > 0 else 0

# This func will loop through a directory and plot the results for every
# results file it finds in there, by calling plot_tests()
def plot_all_tests(window_size, output_size):
    test_path = "/Users/matt/Projects/AdvancedResearchProject/test"
    (_, _, filenames) = next(walk(test_path))

    for file in filenames:
        if "txt" in file:
            with open(file) as f:
                plot_test(json.load(f), window_size, output_size, True)


def plot_one_test():
    window_size = 336
    output_size = 48
    test_path = "/Users/matt/Projects/AdvancedResearchProject/test/"
    with open(test_path) as f:
        plot_test(json.load(f), window_size, output_size, True)


def plot_sliding_window(data_subset, rnn_in, data_out, out):
    font = {'size': 20}
    plt.rc('font', **font)
    fig, axes = plt.subplots(2, 1, figsize=(20, 15), dpi=250)

    for ax in axes:
        ax.set_xticks([])

    # Plot the input data
    axes[0].plot(data_subset.tolist(), label="Actual Data: " + r'$x_t$')

    # Add the sliding windows
    axes[0].plot([168, 504, 504, 168, 168],
                 [20000, 20000, 39500, 39500, 20000],
                 linestyle="--", label="Input Window", color="C1")
    axes[0].plot([504, 552, 552, 504, 504],
                [20000, 20000, 39500, 39500, 20000],
                linestyle="--", label="Output Window", color="C2")
    axes[0].legend(loc="upper left")

    axes[1].plot([i for i in range(336)], rnn_in.squeeze().tolist(),
                 label="dLSTM Input: " + r'$y_t$', color="C1")
    axes[1].plot([i for i in range(336, 384)], out.tolist(),
                 label="dLSTM Output: " + r'$\hat{y_t}$', color="C3")
    axes[1].plot([i for i in range(336, 384)], data_out.tolist(),
                 label="Actual Output: " + r'$y_t$', color="C2")
    axes[1].legend(loc="upper left")
    axes[0].set_xticks([])
    axes[1].set_xticks([])
    plt.show()


def plot_gen_forecast(input, output, seas_in, seas_out, level_in, level_out,
                      lstm_in, lstm_out, pred):

    # Set the font size
    font = {'size': 20}
    plt.rc('font', **font)
    fig, axes = plt.subplots(4, 1, figsize=(20, 15), dpi=250)
    for ax in axes:
        ax.set_xticks([])

    # Plot Actual Data:
    axes[0].plot([i for i in range(384)], input,
                 label="Actual Input: " + r'$x_t$')
    axes[0].plot([i for i in range(336, 384)], output,
                 label="Actual Output: " + r'$x_t$', color="C2")
    axes[0].plot([i for i in range(336, 384)], pred,
                 label="Forecast: " + r'$\hat{x_t}$', color="C1")
    axes[0].axvline(336, color="C5", linestyle="--")

    # Plot levels
    axes[1].plot([i for i in range(384)], level_in,
                 label="Input Level: " + r'$l_t$')
    axes[1].plot([i for i in range(336, 384)], level_out,
                 label="Extrapolated Output Level: " + r'$\hat{l_t}$')
    axes[1].axvline(336, color="C5", linestyle="--")

    # Plot seasonality
    axes[2].plot([i for i in range(384)], seas_in,
                 label="Input Seasonality: " + r'$s_t$')
    axes[2].plot([i for i in range(336, 384)], seas_out,
                 label="Repeated Output Seasonality: " + r'$\hat{s_t}$')
    axes[2].axvline(336, color="C5", linestyle="--")
    axes[2].plot([168, 216, 216, 168, 168], [2.05, 2.05, 3.3, 3.3, 2.05],
                 color="C1", linestyle="--")

    # Plot RNN
    axes[3].plot([i for i in range(336)], lstm_in,
                 label="dLSTM Input: " + r'$y_t$')
    axes[3].plot([i for i in range(336, 384)], lstm_out,
                 label="dLSTM Output: " + r'$\hat{y_t}$')
    axes[3].axvline(336, color="C5", linestyle="--")

    for ax in axes:
        ax.legend(loc="upper left")

    # Plot
    plt.show()


def plot_es_comp(data, seasonality, levels):
    # Create figure
    font = {'size': 20}
    plt.rc('font', **font)

    fig = plt.figure(figsize=(20, 15), dpi=250)

    # Grid of 4 rows by 5 columns
    gs = fig.add_gridspec(3, 5)
    for i in range(3):
        # In each column: before break occupies 3/5, after 2/5
        ax_1 = fig.add_subplot(gs[i, :3])
        ax_2 = fig.add_subplot(gs[i, 3:])
        ax_1.set_xticks([])
        ax_2.set_xticks([])
        if i == 2:
            # Remove right spline of first axis
            ax_1.spines['right'].set_visible(False)

            # Remove left spline, label and ticks of the second axis
            ax_2.spines['left'].set_visible(False)
            ax_2.tick_params(axis='y', which='both', left=False, right=False,
                             labelleft=False)

            # Ensure limits go to ends of x axis
            ax_1.set_xlim(0, 504)
            ax_2.set_xlim(504, 840)

            # Add dashed line with tick at end of initial seasonality
            ax_1.axvline(168, color="black", linestyle='--')
            ax_1.set_xticks(list(ax_1.get_xticks()) + [168])
            ax_1.set_xlim(0, 504)  # Need to do this

            # Add the diagonal line breaks. So painful
            d = .005
            kwargs = dict(transform=ax_1.transAxes, color='k', clip_on=False,
                          linewidth=1)
            ax_1.plot((1 - d + 0.004, 1 + d - 0.003),
                      (1 - d - 0.003, 1 + d + 0.003),
                      **kwargs)
            ax_1.plot((1 - d + 0.004, 1 + d - 0.003),
                      (-d - 0.003, d + + 0.003),
                      **kwargs)

            kwargs.update(transform=ax_2.transAxes)
            ax_2.plot((-d + 0.003, d - 0.003), (1 - d - 0.003, 1 + d + 0.003),
                      **kwargs)
            ax_2.plot((-d + 0.003, d - 0.003), (-d - 0.003, d + 0.003),
                      **kwargs)

            ax_1.plot([i for i in range(168)], seasonality[:168],
                      label="Initial Seasonality Parameters: " +
                            r'$s_1, s_2, ..., s_{168}$', color="C2")
            ax_1.plot([i for i in range(168, 504)],
                              seasonality[168:504],
                              color="C1")
            ax_2.plot([i for i in range(504, 840)], seasonality[-336:],
                      color="C1",
                      label="Seasonality: " + r'$s_{t+168} = \gamma \frac{'
                                              r'x_t}{l_{t-1}} + (1 - '
                                              r'\gamma)s_t$')

            ax_1.legend(loc="upper left")
            ax_2.legend(loc="upper right")
            # Create two legends for readability
            # first_legend = ax_1.legend(handles=[init], loc='upper left')
            # ax_1.add_artist(first_legend)
            # ax_1.legend(handles=[seas], loc='upper right')
        else:
            # Hacky. Left spine (with ticks) at 0, right spine at the left
            ax_1.spines['right'].set_position(('data', -168))
            ax_1.spines['left'].set_position('zero')

            # Remove left spline, label and ticks of the second axis
            ax_2.spines['left'].set_visible(False)
            ax_2.tick_params(axis='y', which='both', left=False, right=False,
                             labelleft=False)

            # Ensure limits go to ends of x axis
            ax_1.set_xlim(-168, 336)
            ax_2.set_xlim(336, 672)

            # Add the diagonal line breaks. So painful
            d = .005
            kwargs = dict(transform=ax_1.transAxes, color='k', clip_on=False,
                          linewidth=1)
            ax_1.plot((1 - d + 0.004, 1 + d - 0.003),
                      (1 - d - 0.003, 1 + d + 0.003),
                      **kwargs)
            ax_1.plot((1 - d + 0.004, 1 + d - 0.003),
                      (-d - 0.003, d + + 0.003),
                      **kwargs)

            kwargs.update(transform=ax_2.transAxes)
            ax_2.plot((-d + 0.003, d - 0.003), (1 - d - 0.003, 1 + d + 0.003),
                      **kwargs)
            ax_2.plot((-d + 0.003, d - 0.003), (-d - 0.003, d + 0.003),
                      **kwargs)

            if i == 0:
                ax_1.plot([i for i in range(336)], data[:336])
                ax_2.plot([i for i in range(336, 672)], data[-336:],
                          label="Actual Data: " + r'$x_t$')
            else:
                ax_1.plot([i for i in range(336)], levels[:336], color="C3")
                ax_2.plot([i for i in range(336, 672)], levels[-336:],
                          color="C3",
                          label="Level: " +
                                r'$l_t = \alpha \frac{x_t}{s_t} + (1 - '
                                r'\alpha)l_{t-1}$'
                          )
            ax_2.legend(loc="upper right")

    # Show the plots
    plt.show()


# This function take a single results dictionary (i.e the result from a week
# long test) and plots each of the 7 tests
def plot_test(results, window_size, output_size, print_results):
    print(results)
    font = {'size': 20}
    plt.rc('font', **font)

    for day in results.keys():
        if day == "overall":
            if print_results:
                print("***** OVERALL RESULTS *****")
                print("Average OWA:", results[day]["avg_owa"])
                print("No. Improved:", results[day]["num_improved"])
                print("Avg. Improvement:", results[day]["avg_improvement"])
                print("Avg. Decline:", results[day]["avg_decline"])
        # elif day not in [7, 6, 5]:
        #     continue
        else:
            # Note results (NCC)
            test_data = results[day]["test_data"]
            prediction = results[day]["ESRNN_prediction"]
            naive_prediction = results[day]["Naive2_prediction"]
            all_levels = results[day]["all_levels"]
            out_levels = results[day]["out_levels"]
            all_seasons = results[day]["all_seas"]
            out_seas = results[day]["out_seas"]
            rnn_out = results[day]["rnn_out"]
            lev_smooth = results[day]["level_smoothing"]
            seas_smooth = results[day]["seasonality_smoothing"]
            # losses = results[day]["losses"]

            fig, axes = plt.subplots(4, 1, figsize=(20, 15), dpi=250)
            axes[0].plot(test_data, label="Actual Data")
            axes[0].plot([i for i in range(window_size, window_size +
                                           output_size)],
                         prediction, label="ES_RNN")
            axes[0].plot([i for i in range(window_size, window_size +
                                           output_size)],
                         naive_prediction, label="Naive2")
            axes[0].axvline(x=window_size, linestyle=":", color="C5")
            axes[0].set_title("Actual Data and Forecasts")
            axes[0].legend(loc="best")

            axes[1].plot(all_levels, label="All Levels")
            axes[1].plot([i for i in range(window_size, window_size +
                                           output_size)],
                         out_levels, label="Output Levels")
            axes[1].axvline(x=window_size, linestyle=":", color="C5")
            axes[1].set_title("Levels")
            axes[1].legend(loc="best")

            axes[2].plot(all_seasons, label="All Seasons")
            axes[2].plot([i for i in range(window_size, window_size +
                                           output_size)],
                         out_seas, label="Output Seasons")
            axes[2].axvline(x=window_size, linestyle=":", color="C5")
            axes[2].set_title("Seasonality")
            axes[2].legend(loc="best")

            axes[3].plot([i for i in range(window_size + 1)],
                         [1 for _ in range(window_size + 1)])
            axes[3].plot([i for i in range(window_size, window_size +
                                           output_size)],
                         rnn_out, label="RNN Output")
            axes[3].axvline(x=window_size, linestyle=":", color="C5")
            axes[3].set_title("RNN Output")
            axes[3].legend(loc="best")

            plt.show()

            local_labels = [0.01 for _ in range(10)] + [0.005 for _ in range(10)] + [
                0.001 for _ in range(10)] + [0.0005 for _ in range(5)]
            global_labels = [0.005 for _ in range(10)] + [0.001 for _ in range(10)] + [
                0.0005 for _ in range(10)] + [0.0001 for _ in range(5)]
            x = [i for i in range(0, 35)]

            # ---------- PLOT ALL EPOCHS (LRs) ----------
            # fig = plt.figure(figsize=(20, 15), dpi=250)
            # plt.suptitle("Losses")
            # gs = fig.add_gridspec(3, 2)
            # ax_1 = fig.add_subplot(gs[0, :])
            # ax_2 = fig.add_subplot(gs[1, 0])
            # ax_3 = fig.add_subplot(gs[1, 1])
            # ax_4 = fig.add_subplot(gs[2, 0])
            # ax_5 = fig.add_subplot(gs[2, 1])
            # axes = [ax_2, ax_3, ax_4, ax_5]
            #
            # rnn = losses["total load actual"]["RNN"]
            #
            # ax_1.plot(rnn, label="RNN Losses")
            # ax_1.legend(loc="best")
            # ax_1.set_xticks(x)
            # ax_1.set_xticklabels(global_labels, rotation=45)
            #
            # for i, k in enumerate(losses.keys()):
            #     lvp = losses[k]["LVP"]
            #     axes[i].set_xticks(x)
            #     axes[i].set_xticklabels(local_labels, rotation=45)
            #     axes[i].plot(lvp, label=k + " LVP")
            #     axes[i].legend(loc="best")
            #
            # plt.show()


def create_pairs(data, train_hours, valid_hours, test_hours,
                 window_size, output_size, multiple):
    # Split up the data
    train_data = data[:train_hours]
    valid_data = data[train_hours - window_size:
                             train_hours + valid_hours]
    test_data = data[train_hours + valid_hours - window_size:
                            train_hours + valid_hours + test_hours]

    train_data = sliding_window(
        train_data, output_size, window_size, "train", multiple
    )

    valid_data = sliding_window(
        valid_data, output_size, window_size, "valid", multiple
    )

    test_data = sliding_window(
        test_data, output_size, window_size, "test", multiple
    )

    return train_data, valid_data, test_data


def sliding_window(train_data, output_size, window_size, section, multiple):
    inputs = []
    labels = []
    step = 1 if section == "train" else output_size

    for i in range(0, len(train_data) - window_size - output_size + 1, step):
        x = train_data[i: i + window_size]

        if multiple:
            # Use only the total load actual column (column 14) for the label
            # y = train_data[i + window_size: i + window_size + output_size,
        # 14]
            # TODO - CHANGE BACK when fixed the zero values
            y = train_data[i + window_size: i + window_size + output_size, 1]
        else:
            y = train_data[i + window_size: i + window_size + output_size]

        inputs.append(x)
        labels.append(y)

    inputs = torch.tensor(np.array(inputs), dtype=torch.double)
    labels = torch.tensor(np.array(labels), dtype=torch.double)

    return inputs, labels


def batch_data(training_data, batch_size):
    input_batches = []
    label_batches = []

    for i in range(0, len(training_data[0]), batch_size):
        input_batches.append(training_data[0][i:i + batch_size])
        label_batches.append(training_data[1][i:i + batch_size])

    return input_batches, label_batches


def pinball_loss(pred, actual, tau):
    return torch.mean(torch.where(
        actual >= pred,
        (actual - pred) * tau,
        (pred - actual) * (1 - tau)
    ))
