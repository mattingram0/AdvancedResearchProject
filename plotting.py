import json
import os
import sys
from os import walk

import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from numpy.polynomial.polynomial import polyfit

from ml.ml_helpers import latent_enthalpy
from stats.stats_helpers import split_data


# Generate plots to be used in the presentation
def plots_for_presentation(demand_df):
    font = {'size': 24}
    plt.rc('font', **font)

    all_data = split_data(demand_df)
    data = all_data["Spring"][2]

    fig, axes = plt.subplots(2, 2, figsize=(20, 15), dpi=250)
    axes = axes.flatten()
    axes[0].plot(data.loc["2017-03-06 00:00:00+01:00":"2017-03-12 23:00:00+01:00"]["total load actual"], color="C0")
    axes[0].set_title("Total Energy Demanded")
    axes[1].plot(
        data.loc["2017-03-06 00:00:00+01:00":"2017-03-12 23:00:00+01:00"][
            "generation fossil gas"], color="C1")
    axes[1].set_title("Energy Generated - Gas")
    axes[2].plot(
        data.loc["2017-03-06 00:00:00+01:00":"2017-03-12 23:00:00+01:00"][
            "generation fossil oil"], color="C2")
    axes[2].set_title("Energy Generated - Oil")
    axes[3].plot(
        data.loc["2017-03-06 00:00:00+01:00":"2017-03-12 23:00:00+01:00"][
            "price actual"], color="C3")
    axes[3].set_title("Energy Price")

    for ax in axes:
        ax.set_xticks([])
    plt.show()
    sys.exit(0)

    pre_end = 5 * 24
    inp_end = 14 * 24 + pre_end
    out_end = 2 * 24 + inp_end
    aft_end = 2 * 24 + out_end
    start = (64 - 19) * 24
    end = (64 + 4) * 24
    input_data = data["total load actual"][start:end].tolist()
    test_data = data[start + pre_end:start + out_end]


    model = torch.load("/Users/matt/Projects/AdvancedResearchProject/models"
                       "/model_all.pt")
    model.eval()

    levels = [l.item() for l in model.levels["total load actual"]]
    seasonals = [s.item() for s in model.seasonals["total load actual"]]

    fig, axes = plt.subplots(4, 1, figsize=(20, 15), dpi=250)
    axes[0].set_title("Sliding Window, Fitted ES Components, and Normalised "
                      "Data")
    axes[0].plot(input_data[:inp_end], color="C0", label="Input Data")
    axes[0].plot([i for i in range(inp_end, out_end)],
                 input_data[inp_end: out_end], color="C0", linestyle="--")
    axes[0].plot([i for i in range(out_end, aft_end)],
                 input_data[out_end: aft_end], color="C0")
    axes[0].plot([pre_end, inp_end, inp_end, pre_end, pre_end],
                 [19000, 19000, 35000, 35000, 19000],
                 linestyle=":", label="Input Window", color="darkorange")
    axes[0].plot([inp_end, out_end, out_end, inp_end, inp_end],
                 [19000, 19000, 35000, 35000, 19000],
                 linestyle=":", label="Output Window", color="teal")
    axes[1].plot(levels[start + pre_end:start + inp_end], color="C1",
                 label="Fitted ES Level Values")
    axes[1].plot([i for i in range(inp_end - pre_end, out_end - pre_end)],
                 levels[start + inp_end:start + out_end], color="C1",
                 linestyle="--")
    axes[1].axvline(x=336, c='grey', linestyle=":")
    axes[2].plot(seasonals[start + pre_end:start + inp_end], color="C2",
                 label="Fitted ES Seasonality Values")
    axes[2].plot([i for i in range(inp_end - pre_end, out_end - pre_end)],
                 seasonals[168 + start + inp_end:168 + start + out_end],
                 color="C2", linestyle="--")
    axes[2].axvline(x=336, c='grey', linestyle=":")

    normalised_inp = np.log(
        np.array(input_data[pre_end:inp_end]) /
        (np.array(seasonals[168 + start + pre_end: 168 + start + inp_end]) *
         levels[start + inp_end])
    )
    normalised_out = np.log(
        np.array(input_data[inp_end:out_end]) /
        (np.array(seasonals[168 + start + inp_end: 168 + start + out_end]) *
         levels[start + inp_end])
    )

    axes[3].plot(normalised_inp, color="C3", label="De-seasonalised and "
                                               "Normalised Data")
    axes[3].plot([i for i in range(inp_end - pre_end, out_end - pre_end)],
                 normalised_out, color="C5", linestyle="--")
    axes[3].axvline(x=336, c='grey', linestyle=":")
    for ax in axes:
        ax.legend(loc="upper left")
    plt.show()

    pred, out_actuals, out_levels, out_seas, all_levels, \
    all_seasonals, out = model.predict(test_data, window_size, output_size,
                                   weather)

    fig, axes = plt.subplots(4, 1, figsize=(20, 15), dpi=250)
    axes[0].set_title("dLSTM and Actual Output")
    axes[0].plot(normalised_out, label="Actual Output", color="C5", linestyle="--")
    axes[0].plot(torch.log(out).detach().view(-1).numpy(), label="dLSTM Output", color="C3")
    axes[0].legend(loc="upper right")
    plt.show()


# Generate plots to be used in the poster
def plots_for_poster(demand_df):
    font = {'size': 24}
    plt.rc('font', **font)

    window_size = 336
    output_size = 48
    weather = False

    all_data = split_data(demand_df)
    data = all_data["Spring"][2]
    start_test = -(15 * 24 + window_size)
    end_test = -(15 * 24 - output_size)
    test_data = data[start_test:end_test]

    model = torch.load("/Users/matt/Projects/AdvancedResearchProject/models"
                   "/model_all"
                   ".pt")
    model.eval()

    levels = [l.item() for l in model.levels["total load actual"]]
    seasonals = [s.item() for s in model.seasonals["total load actual"]]

    fig, axes = plt.subplots(4, 1, figsize=(20, 15), dpi=250)
    axes[0].set_title("Input Data, Fitted ES Components, and Normalised Data")
    axes[0].plot(test_data["total load actual"][:window_size], color="C0",
                 label="Input Data")
    axes[1].plot(levels[-window_size:], color="C1",
                 label="Fitted ES Level Values")
    axes[2].plot(seasonals[-window_size:], color="C2",
                 label="Fitted ES Seasonality Values")
    axes[2].plot([7 * 24, 9 * 24, 9 * 24, 7 * 24, 7 * 24],
                 [2.1, 2.1, 3.3, 3.3, 2.1], color="C2",
                 linestyle=":", label="Output Window")
    normalised = np.log(
        np.array(test_data["total load actual"][:window_size]) /
        (np.array(seasonals[-window_size:]) * levels[-1])
    )
    axes[3].plot(normalised, color="C3", label="De-seasonalised and "
                                               "Normalised Data")
    for ax in axes:
        ax.legend(loc="upper left")
    plt.show()

    pred, out_actuals, out_levels, out_seas, all_levels, \
    all_seasonals, out = model.predict(test_data, window_size, output_size,
                                   weather)

    fig, axes = plt.subplots(4, 1, figsize=(20, 15), dpi=250)
    axes[0].set_title("dLSTM Output, Repeated Level and Seasonality Values, "
                      "and Final Forecast")
    axes[0].plot(out.view(-1).detach(), color="C3",
                 label="dLSTM Output")
    axes[1].plot(out_levels.view(-1).detach(), color="C1",
                 label="Extrapolated Level Values")
    axes[2].plot(out_seas.view(-1).detach(), color="C2",
                 label="Repeated Seasonality Values")
    axes[3].plot(pred.view(-1).detach(), color="C4",
                 label="Final Forecast")
    axes[3].plot(out_actuals, color="C5",
                 label="Actual Data")
    for ax in axes:
        ax.legend(loc="upper left")
    plt.show()


# Plot the losses against learning rate for a given test
def plot_lr():
    init = True

    path = "/Users/matt/Projects/AdvancedResearchProject/test/" \
           "ingram_weather_year_2_summer.txt"

    with open(path) as f:
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


# Plot the result of the learning rate tests to identify learning rate schedule
def plot_learning_rates():
    path = "/Users/matt/Projects/AdvancedResearchProject/test/ingram_weather_" \
           "year_2_winter_-1_1.txt"

    with open(path) as f:
        results = json.load(f)

    for test in [1, 2, 3, 4, 5, 6, 7]:

        losses = results[str(test)]["losses"]
        rnn = losses["total load actual"]["RNN"]

        lr = [0.01 for _ in range(1, 10)] + \
             [0.005 for _ in range(1, 10)] + \
             [0.001 for _ in range(1, 10)] + \
             [0.0001 for _ in range(1, 5)]
        labels = [np.round(lr[i], decimals=5) for i in range(0, len(lr), 3)]
        x = [i for i in range(0, len(lr), 3)]

        fig, axes = plt.subplots(2, 1, figsize=(20, 15), dpi=250)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(labels, rotation=45)
        axes[0].plot(rnn, label="RNN Losses")
        axes[0].set_title("RNN Test " + str(test))
        axes[0].legend(loc="best")

        axes[1].set_xticks(x)
        axes[1].set_xticklabels(labels, rotation=45)
        axes[1].plot(rnn[:65], label="RNN Losses")
        axes[1].legend(loc="best")
        plt.show()

        # Plot LVPs for each
        col_list = ["generation fossil gas",
                    "generation fossil hard coal",
                    "generation fossil oil",
                    "generation hydro run-of-river and poundage",
                    "generation hydro water reservoir", "total load forecast",
                    "total load actual", "price day ahead", "price actual",
                    ]
        col_list = ["total load actual", "generation fossil gas",
                     "generation fossil hard coal", "generation fossil oil"]
        for f in col_list:
            lvp = losses[f]["LVP"]
            fig, axes = plt.subplots(2, 1, figsize=(20, 15), dpi=250)
            axes[0].set_xticks(x)
            axes[0].set_xticklabels(labels, rotation=45)
            axes[0].plot(lvp, label="LVP Losses")
            axes[0].set_title(f + " Test " + str(test))
            axes[0].legend(loc="best")

            axes[1].set_xticks(x)
            axes[1].set_xticklabels(labels, rotation=45)
            axes[1].plot(lvp[:65], label="LVP Losses")
            axes[1].legend(loc="best")
            plt.show()


# Plot all tests in a directory
def plot_all_tests(window_size, output_size):
    test_path = "/Users/matt/Projects/AdvancedResearchProject/test"
    (_, _, filenames) = next(walk(test_path))

    for file in filenames:
        if "txt" in file:
            with open(file) as f:
                plot_test(json.load(f), window_size, output_size, True)


# Plot the results of a single test
def plot_one_test():
    window_size = 336
    output_size = 48
    test_path = "/Users/matt/Projects/AdvancedResearchProject/test/"
    with open(test_path) as f:
        plot_test(json.load(f), window_size, output_size, True)


# Plot the sliding window graph used in the report
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


# Plot the graph in the report demonstrating how a forecast is generated
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


# Plot the graph in the report showing the ES decomposition
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


# Plot the result from a week long test
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
            losses = results[day]["losses"]

            fig, axes = plt.subplots(4, 1, figsize=(20, 15), dpi=250)
            axes[0].plot(test_data, label="Actual Data")
            axes[0].plot([i for i in range(window_size, window_size +
                                           output_size)],
                         prediction, label="ES_RNN_S")
            axes[0].plot([i for i in range(window_size, window_size +
                                           output_size)],
                         naive_prediction, label="Naive2")
            axes[0].axvline(x=window_size, linestyle=":", color="C5")
            axes[0].set_title("Actual Data and Forecasts Test " + day)
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
            fig = plt.figure(figsize=(20, 15), dpi=250)
            plt.suptitle("Losses")
            gs = fig.add_gridspec(3, 2)
            ax_1 = fig.add_subplot(gs[0, :])
            ax_2 = fig.add_subplot(gs[1, 0])
            ax_3 = fig.add_subplot(gs[1, 1])
            ax_4 = fig.add_subplot(gs[2, 0])
            ax_5 = fig.add_subplot(gs[2, 1])
            axes = [ax_2, ax_3, ax_4, ax_5]

            rnn = losses["total load actual"]["RNN"]

            ax_1.plot(rnn, label="RNN Losses")
            ax_1.legend(loc="best")
            ax_1.set_xticks(x)
            ax_1.set_xticklabels(global_labels, rotation=45)

            for i, k in enumerate(losses.keys()):
                lvp = losses[k]["LVP"]
                axes[i].set_xticks(x)
                axes[i].set_xticklabels(local_labels, rotation=45)
                axes[i].plot(lvp, label=k + " LVP")
                axes[i].legend(loc="best")

            plt.show()


# Generate plots to demonstrate the relationship between demand and weather
def weather_analysis():
    weather_file = "/Users/matt/Projects/AdvancedResearchProject/data/spain/" \
                   "weather_features.csv"
    demand_file = "/Users/matt/Projects/AdvancedResearchProject/data/spain/" \
             "energy_dataset.csv"
    weather_list = ["dt_iso", "city_name", "temp", "humidity", "wind_speed",
                    "pressure"]
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

    # Font size
    font = {'size': 20}
    plt.rc('font', **font)

    # Plot of average temperature v total load actual
    fig, axes = plt.subplots(2, 2, figsize=(20, 15), dpi=250)
    axes = axes.flatten()
    axes[0].scatter(avg_weather["temp"], demand["total load actual"])
    axes[0].set_title("Aggregated Demand against Temperature (K)")
    b, m, m2 = polyfit(avg_weather["temp"], demand["total load actual"], 2)
    x = np.arange(
        np.floor(avg_weather["temp"].min()),
        np.ceil(avg_weather["temp"].max()) + 1
    )
    axes[0].plot(x, b + m * x + m2 * x**2, color="#fc8403")
    # plt.show()

    # fig, ax = plt.subplots(1, 1, figsize=(20, 15), dpi=250)
    axes[1].scatter(avg_weather["HD"], demand["total load actual"])
    axes[1].set_title("Aggregated Demand against Heating Degree")
    b, m = polyfit(avg_weather["HD"], demand["total load actual"], 1)
    x = np.arange(np.ceil(avg_weather["HD"].max()) + 1)
    axes[1].plot(x, b + m * x, color="#fc8403")
    # plt.show()

    # fig, ax = plt.subplots(1, 1, figsize=(20, 15), dpi=250)
    axes[2].scatter(avg_weather["CD"], demand["total load actual"])
    axes[2].set_title("Aggregated Demand against Cooling Degree")
    b, m = polyfit(avg_weather["CD"], demand["total load actual"], 1)
    x = np.arange(np.ceil(avg_weather["CD"].max()) + 1)
    axes[2].plot(x, b + m * x, color="#fc8403")
    # plt.show()

    # fig, ax = plt.subplots(1, 1, figsize=(20, 15), dpi=250)
    axes[3].scatter(avg_weather["LE"], demand["total load actual"])
    axes[3].set_title("Aggregated Demand against Latent Enthalpy")
    b, m = polyfit(avg_weather["LE"], demand["total load actual"], 1)
    x = np.arange(np.ceil(avg_weather["LE"].max()) + 1)
    axes[3].plot(x, b + m * x, color="#fc8403")
    plt.show()

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

    # Plots of temperature v total load actual for each city
    for c in cities:
        c = c.set_index('dt_iso').asfreq('H')
        name = c.iloc[0]["city_name"].strip()
        color = city_color_map[name]

        fig, ax = plt.subplots(1, 1, figsize=(20, 15), dpi=250)
        ax.scatter(c["temp"], demand["total load actual"], color=color)
        ax.set_title(name + ": Demand against Temperature")
        plt.show()

        fig, ax = plt.subplots(1, 1, figsize=(20, 15), dpi=250)
        ax.scatter(c["HD"], demand["total load actual"], color=color)
        ax.set_title(name + ": Demand against Heating Degree")
        plt.show()

        fig, ax = plt.subplots(1, 1, figsize=(20, 15), dpi=250)
        ax.scatter(c["CD"], demand["total load actual"], color=color)
        ax.set_title(name + ": Demand against Cooling Degree")
        plt.show()

        fig, ax = plt.subplots(1, 1, figsize=(20, 15), dpi=250)
        ax.scatter(c["LE"], demand["total load actual"], color=color)
        ax.set_title(name + ": Demand against Latent Enthalpy")
        b, m = polyfit(c["LE"], demand["total load actual"], 1)
        x = np.arange(np.ceil(c["LE"].max()) + 1)
        ax.plot(x, b + m * x, color="#fc8403")
        plt.show()


# Plot the test result forecasts for all the stats models
def plot_forecasts(df, season, year):
    split = split_data(df)
    for test in range(8, 1, -1):
        fig, axes = plt.subplots(2, 1, figsize=(20, 15), dpi=250)

        test_path = "/Users/matt/Projects/AdvancedResearchProject/results" \
                    "/non_ensemble_results/res/"
        (_, _, filenames) = next(os.walk(test_path))

        train_end = -(test * 24)
        test_end = -(test * 24 - 48) if test > 2 else None
        actual = split[season][year]["total load actual"][train_end:test_end].tolist()

        axes[0].plot(actual, label="Actual")
        axes[1].plot(actual, label="Actual")

        for file in filenames:
            if "forecasts" not in file:
                continue

            seas, method, _ = file.split("_")

            if seas != season:
                continue

            # methods = ["Naive1", "Naive2", "NaiveS", "SES", "Holt", "Damped",
            #            "Holt-Winters", "Comb", "ARIMA", "SARIMA", "Auto", "Theta",
            #            "TSO", "ES-RNN-S", "ES-RNN-SW", "ES-RNN-D", "ES-RNN-DW",
            #            "ES-RNN-I", "ES-RNN-IW"]
            # good = [""]

            # top = ["NaiveS", "ES-RNN-I", "Comb", "Holt", "Naive2", "SES", "TSO"]
            top = ["NaiveS", "TSO"]
            bottom = ["Theta", "Damped", "ARIMA", "SARIMA", "Holt-Winters", "Naive1"]

            with open(test_path + file) as f:
                all_forecasts = json.load(f)
                forecast = all_forecasts[str(year)][str(1)][str(test - 1)]

            # Plot 6 tests on first axes
            if method in top:
                axes[0].plot(forecast, label=method, marker='o')
            elif method in bottom:
                axes[1].plot(forecast, label=method, marker='o')
            else:
                pass  # To handle the empty 'Auto' forecasts

        axes[0].legend(loc="best")
        axes[1].legend(loc="best")
        plt.show()


# Plot the results graphs used in the report
def plot_48_results():
    methods = ["Naive1", "Naive2", "NaiveS", "SES", "Holt", "Damped",
               "Holt-Winters", "Comb", "ARIMA", "SARIMA", "Auto", "Theta",
               "TSO", "ES-RNN-S", "ES-RNN-SW", "ES-RNN-D", "ES-RNN-DW",
               "ES-RNN-I", "ES-RNN-IW"]
    circles = ["ES-RNN-S", "ES-RNN-SW", "ES-RNN-D", "ES-RNN-DW",
               "ES-RNN-I", "ES-RNN-IW"]
    # exclude = ["Naive1", "Auto", "SARIMA"]
    exclude = ["TSO", "Auto", "Naive1"]
    sns.reset_orig()
    clrs = sns.color_palette('husl', n_colors=len(methods))
    x_tick_locs = [0] + [i for i in range(-1, 48, 5)][1:] + [47]
    x_tick_labels = [1] + [i + 1 for i in range(-1, 48, 5)][1:] + [48]

    # Load OWA
    with open("/Users/matt/Projects/AdvancedResearchProject/results/"
              "non_ensemble_results/res/results_48_owa.txt") as file:
        owas48 = json.load(file)

    # Convert {Lead: {Method: Val, ...}, ...} to {Method: [Vals], ...}
    owas = {m: [0] * 48 for m in methods}
    for lead, vals in owas48.items():
        for m, owa in vals.items():
            owas[m][int(lead) - 1] = owa

    # Load sMAPE
    with open("/Users/matt/Projects/AdvancedResearchProject/results/"
              "non_ensemble_results/res/results_48_smape.txt") as file:
        smapes48 = json.load(file)

    # Convert {Lead: {Method: Val, ...}, ...} to {Method: [Vals], ...}
    smapes = {m: [0] * 48 for m in methods}
    for lead, vals in smapes48.items():
        for m, smape in vals.items():
            smapes[m][int(lead) - 1] = smape

    # Load MASE
    with open("/Users/matt/Projects/AdvancedResearchProject/results/"
              "non_ensemble_results/res/results_48_mase.txt") as file:
        mases48 = json.load(file)

    # Convert {Lead: {Method: Val, ...}, ...} to {Method: [Vals], ...}
    mases = {m: [0] * 48 for m in methods}
    for lead, vals in mases48.items():
        for m, mase in vals.items():
            mases[m][int(lead) - 1] = mase

    font = {'size': 24}
    plt.rc('font', **font)

    # Plot OWAS
    fig, ax = plt.subplots(1, 1, figsize=(20, 15), dpi=250)
    for i, (m, e) in enumerate(owas.items()):
        if m in exclude:
            continue

        marker = 'o' if m in circles else 'x'
        ax.plot(e, label=m, marker=marker)


    ax.legend(loc="best")
    ax.axvline(x=24, linestyle=":")
    ax.set_title("OWA against Lead Time")
    ax.set_xticks(x_tick_locs)
    ax.set_xticklabels(x_tick_labels)
    plt.show()

    # Plot sMAPES - 48 hour
    legend_1_plots = []
    legend_2_plots = []
    fig, ax = plt.subplots(1, 1, figsize=(20, 15), dpi=250)
    for i, (m, e) in enumerate(smapes.items()):
        if m in exclude:
            continue

        marker = 'o' if m in circles else 'x'
        a, = ax.plot(e, label=m, marker=marker)
        ax.set_ylim(1, 5)

        if m in circles:
            legend_1_plots.append(a)
        else:
            legend_2_plots.append(a)

    # ax.legend(loc="lower right")
    ax.axvline(x=23, linestyle=":")
    ax.set_title("sMAPE against Lead Time: 48 Hour Forecast")
    ax.set_xticks(x_tick_locs)
    ax.set_xticklabels(x_tick_labels)
    ax.set_xlabel("Hour Ahead")
    ax.set_ylabel("sMAPE (%)")
    first_legend = ax.legend(handles=legend_1_plots, loc='upper left')
    ax.add_artist(first_legend)
    ax.legend(handles=legend_2_plots, loc='lower right')
    plt.show()

    # Plot sMAPES - 12 Hour
    max_lead = 12
    legend_1_plots = []
    legend_2_plots = []
    fig, ax = plt.subplots(1, 1, figsize=(20, 15), dpi=250)
    for i, (m, e) in enumerate(smapes.items()):
        if m in exclude:
            continue

        marker = 'o' if m in circles else 'x'
        a, = ax.plot(e[:max_lead], label=m, marker=marker)
        ax.set_ylim(1, 4)

        if m in circles:
            legend_1_plots.append(a)
        else:
            legend_2_plots.append(a)

    first_legend = ax.legend(handles=legend_1_plots, loc='upper left')
    ax.add_artist(first_legend)
    ax.legend(handles=legend_2_plots, loc='lower right')
    ax.set_xticks([i for i in range(max_lead)])
    ax.set_xticklabels([i + 1 for i in range(max_lead)])
    ax.set_title("sMAPE against Lead Time: " + str(max_lead) + " Hour "
                                                               "Forecast")
    ax.set_xlabel("Hour Ahead")
    ax.set_ylabel("sMAPE (%)")
    plt.show()

    print(pd.DataFrame(owas))

    # Plot MASES
    fig, ax = plt.subplots(1, 1, figsize=(20, 15), dpi=250)
    for i, (m, e) in enumerate(mases.items()):
        if m in exclude:
            continue

        marker = 'o' if m in circles else 'x'
        ax.plot(e, label=m, marker=marker)

    ax.legend(loc="best")
    ax.axvline(x=24, linestyle=":")
    ax.set_title("MASE against Lead Time")
    ax.set_xticks(x_tick_locs)
    ax.set_xticklabels(x_tick_labels)
    plt.show()
