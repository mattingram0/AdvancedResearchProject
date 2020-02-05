import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os


def load_results(date, multiple, test_hours):
    folder_str = "/Users/matt/Documents/Durham/4th Year/" \
           "Advanced Research Project/Results/" + date + "/run/results/"
    # folder = os.path.abspath(folder_str)

    folder = os.path.abspath(
        "/Users/matt/Projects/AdvancedResearchProject/run/results/"
    )
    all_result_files = sorted(os.listdir(folder_str))
    dataframes = {}

    for result_file in all_result_files:
        if multiple:
            if result_file[-5:] == "m.csv":
                train_days = int(
                    result_file[result_file.rfind("_") +
                                1:result_file.find(".") - 1]
                )
            else:
                continue
        else:
            if result_file[-5:] == "m.csv":
                continue
            else:
                train_days = int(
                    result_file[
                        result_file.rfind("_") + 1:result_file.find(".")]
                )

        temp_df = pd.read_csv(os.path.join(folder_str, result_file))
        method_name = str(temp_df.columns[-1]).capitalize()

        # Create one dataframe per training day value, then concat after
        if train_days not in dataframes.keys():
            if multiple:
                new_df = pd.DataFrame(
                    {
                        "Error": (["sMAPE"] * test_hours) +
                                 (["RMSE"] * test_hours) +
                                 (["MASE"] * test_hours) +
                                 (["MAE"] * test_hours) +
                                 (["OWA"] * test_hours),
                        "Training Days": [train_days] * 5 * test_hours
                    }
                ).set_index("Training Days")
            else:
                new_df = pd.DataFrame(
                    {"Error": ["sMAPE", "RMSE", "MASE", "MAE", "OWA"],
                     "Training Days": [train_days] * 5}
                ).set_index("Training Days")

            new_df[method_name] = temp_df[temp_df.columns[-1]].values
            dataframes[train_days] = new_df
        else:
            existing_df = dataframes[train_days]
            existing_df[method_name] = temp_df[temp_df.columns[-1]].values

    sorted_dataframes = {k: dataframes[k] for k in sorted(dataframes)}
    df = pd.concat(sorted_dataframes.values(), sort=True)
    owa = df[df["Error"] == "OWA"]

    # Plot the results
    if multiple:
        for t in dataframes.keys():
            owat = owa.loc[t]
            fig = plt.figure(figsize=(12.8, 9.6), dpi=250)
            ax = fig.add_subplot(1, 1, 1)
            ax.set_title(
                "OWA Error Results - Multiple Point Forecast - " +
                str(t) + " Training Days"
            )

            for method in owat.columns:
                if method == "Error" or method == "Naive2":
                    continue
                ax.plot(
                    [i for i in range(1, test_hours + 1)],
                    owat[method], label=method, marker='o'
                )

            ax.legend(loc="best")
            ax.set_xlabel("Hour Ahead Forecast")
            ax.set_ylabel("OWA")
            plt.show()

    else:
        fig = plt.figure(figsize=(12.8, 9.6), dpi=250)
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title("OWA Error Results - Single Point Forecast")

        for method in owa.columns:
            if method == "Error":
                continue
            ax.plot(owa.index, owa[method], label=method)

        ax.legend(loc="best")
        plt.show()


load_results("30_01", True, 48)
