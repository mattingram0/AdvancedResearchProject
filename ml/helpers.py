import numpy as np
import matplotlib.pyplot as plt
import torch
import json

from os import walk


# This func will loop through a directory and plot the results for every
# results file it finds in there, by calling plot_tests()
def plot_all_tests(window_size, output_size):
    test_path = "/Users/matt/Projects/AdvancedResearchProject/test"
    (_, _, filenames) = next(walk(test_path))

    for file in filenames:
        if "txt" in file:
            with open(file) as f:
                plot_test(json.load(f), window_size, output_size, True)


# This function take a single results dictionary (i.e the result from a week
# long test) and plots each of the 7 tests
def plot_test(results, window_size, output_size, print_results):
    print(results)
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
