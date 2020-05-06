import json

import numpy as np
import pandas as pd


# Display results for forecasts at a given lead hour for final tests
from stats.errors import sMAPE
from stats.stats_helpers import split_data


def display_results():
    methods = ["Naive1", "Naive2", "NaiveS", "SES", "Holt", "Damped",
               "Holt-Winters", "Comb", "ARIMA", "SARIMA", "Theta",
               "ES-RNN-S", "ES-RNN-SW", "ES-RNN-D", "ES-RNN-DW",
               "ES-RNN-I", "ES-RNN-IW"]
    seasons = ["Spring", "Summer", "Autumn", "Winter"]
    ten_reps = ["Naive1", "Naive2", "NaiveS", "SES", "Holt", "Damped",
               "Holt-Winters", "Comb", "ARIMA", "Theta"]

    smapes = {s: {m: [] for m in methods} for s in seasons + ["Average"]}
    mases = {s: {m: [] for m in methods} for s in seasons + ["Average"]}
    owas = {s: {m: [] for m in methods} for s in seasons + ["Average"]}
    hour_ahead = 2  # Which hour ahead do we want to average results for

    for m in methods:
        num_reps = 10 if m in ten_reps else 1
        means = []
        stds = []
        for s in seasons:
            filepath = s + "_" + m + "_results.txt"
            with open("/Users/matt/Projects/AdvancedResearchProject/results/"
              "non_ensemble_results/res/" + filepath) as file:
                results = json.load(file)

            all_res = []
            for r in range(1, num_reps + 1):
                for y in range(1, 5):
                    for t in range(1, 8):
                        smape = results["sMAPE"][str(r)][str(y)][str(t)][hour_ahead]
                        all_res.append(smape)

            if m == "ES-RNN-S" and s == "Summer":
                print(all_res)

            mean = np.around(np.mean(all_res), decimals=3)
            std = np.around(np.std(all_res), decimals=3)

            smapes[s][m] = [mean, std]
            means.append(mean)
            stds.append(std)

        avg_mean = np.around(np.mean(means), decimals=3)
        avg_std = np.around(np.mean(stds), decimals=3)

        smapes["Average"][m] = [avg_mean, avg_std]

    print("sMAPES:")
    print(pd.DataFrame(smapes))
    print()

    for m in methods:
        num_reps = 10 if m in ten_reps else 1
        means = []
        stds = []
        for s in seasons:
            filepath = s + "_" + m + "_results.txt"
            with open("/Users/matt/Projects/AdvancedResearchProject/results/"
              "non_ensemble_results/res/" + filepath) as file:
                results = json.load(file)

            all_res = []
            for r in range(1, num_reps + 1):
                for y in range(1, 5):
                    for t in range(1, 8):
                        mase = results["MASE"][str(r)][str(y)][str(t)][hour_ahead]
                        all_res.append(mase)

            if m == "ES-RNN-S" and s == "Summer":
                print(all_res)

            mean = np.around(np.mean(all_res), decimals=3)
            std = np.around(np.std(all_res), decimals=3)

            mases[s][m] = [mean, std]
            means.append(mean)
            stds.append(std)

        avg_mean = np.around(np.mean(means), decimals=3)
        avg_std = np.around(np.mean(stds), decimals=3)

        mases["Average"][m] = [avg_mean, avg_std]

    print("MASEs:")
    print(pd.DataFrame(mases))
    print()

    for m in methods:
        for s in seasons + ["Average"]:
            smape_mean_naive = smapes[s]["Naive2"][0]
            smape_std_naive = smapes[s]["Naive2"][1]
            mase_mean_naive = mases[s]["Naive2"][0]
            mase_std_naive = mases[s]["Naive2"][1]

            smape_mean = smapes[s][m][0]
            smape_std = smapes[s][m][1]
            mase_mean = mases[s][m][0]
            mase_std = mases[s][m][1]

            mean_owa = ((smape_mean/smape_mean_naive) + (mase_mean/mase_mean_naive)) * 0.5
            std_owa = ((smape_std / smape_std_naive) + (mase_std / mase_std_naive)) * 0.5
            owas[s][m] = [np.around(mean_owa, decimals=3),
                          np.around(std_owa, decimals=3)]

    print("OWAs:")
    print(pd.DataFrame(owas).sort_values("Average"))


# Average the test results across multiple seasons
def average_test():
    base = "/Users/matt/Projects/AdvancedResearchProject/test/"
    tests = ["smyl_multiple_weather_year_2_season_",
             "smyl_multiple_no_weather_year_2_season_",
             "smyl_1_weather_year_2_season_",
             "smyl_1_no_weather_year_2_season_",
             "ingram_weather_year_2_season_",
             "ingram_no_weather_year_2_season_"]
    test_names = ["ES_RNN_SW", "ES_RNN_S", "ES_RNN_DW", "ES_RNN_D",
                  "ES_RNN_IW", "ES_RNN_I"]

    init_vals = ["-1_-1.out", "-1_1.out", "1_-1.out", "1_1.out"]
    init_vals_names = ["-1, -1", "-1, 1", "1, -1", "1, 1"]
    seasons = ["0_smoothing_", "1_smoothing_", "2_smoothing_", "3_smoothing_"]

    results = {t: {iv: 0 for iv in init_vals_names} for t in test_names}
    for t, n in zip(tests, test_names):
        for iv, ivn in zip(init_vals, init_vals_names):
            owas = []
            for s in seasons:
                filename = base + t + s + iv

                with open(filename) as f:
                    avg_owa = list(f)[-4]
                    owas.append(float(avg_owa.split(":")[1].strip()))

            results[n][ivn] = np.mean(owas)

    results = pd.DataFrame(results)
    print(results, "\n")
    print(results.min(axis=0), "\n")
    print(results.min(axis=1))


# Function just to check that the errors calculated during training are correct
def check_errors(df):
    forecast_length = 48
    base = "/Users/matt/Projects/AdvancedResearchProject/results" \
           "/non_ensemble_results/res/"
    seasons = ["Spring_", "Summer_", "Autumn_", "Winter_"]
    methods = ["Naive2_forecasts.txt", "NaiveS_forecasts.txt", "TSO_forecasts.txt", "ES-RNN-I_forecasts.txt"]
    seas_dict = {1: "Spring", 2: "Summer", 3: "Autumn", 4: "Winter"}

    # Begin with a hard coded season:
    seas = 3  # 0: Spring, 1: Summer, 2: Autumn, 3: Winter
    seas_n = seas_dict[seas + 1]

    split = split_data(df)

    # Load forecasts
    with open(base + seasons[seas] + methods[0]) as f:
        naive2_forecasts = json.load(f)

    with open(base + seasons[seas] + methods[1]) as f:
        naives_forecasts = json.load(f)

    with open(base + seasons[seas] + methods[2]) as f:
        tso_forecasts = json.load(f)

    with open(base + seasons[seas] + methods[3]) as f:
        es_rnn_i_forecasts = json.load(f)

    # TSO Results
    with open(base + seasons[seas] + "TSO_results.txt") as f:
        tso_results = json.load(f)

    # Calculate sMAPES:
    tso_smapes = []
    es_rnn_smapes = []
    naive2_smapes = []
    naives_smapes = []
    tso_test_smapes = []
    for y in range(1, 5):
        for t in range(1, 8):
            train_end = -((t + 1) * 24)
            test_end = -((t + 1) * 24 - forecast_length) if (t + 1) > 2 else None
            naive2 = naive2_forecasts[str(y)][str(1)][str(t)]
            naives = naives_forecasts[str(y)][str(1)][str(t)]
            tso = tso_forecasts[str(y)][str(1)][str(t)]
            es_rnn = es_rnn_i_forecasts[str(y)][str(1)][str(t)]
            actual = split[seas_n][y - 1]["total load actual"][
                     train_end:test_end].tolist()
            tso_smapes.append(sMAPE(pd.Series(actual), pd.Series(tso)))
            es_rnn_smapes.append(sMAPE(pd.Series(actual), pd.Series(es_rnn)))
            naive2_smapes.append(sMAPE(pd.Series(actual), pd.Series(naive2)))
            naives_smapes.append(sMAPE(pd.Series(actual), pd.Series(naives)))
            tso_test_smapes.append(tso_results["sMAPE"][str(1)][str(y)][str(
                t)][47])

    print("Average ES-RNN-I sMAPE:", np.mean(es_rnn_smapes))
    print("Average TSO sMAPE:", np.mean(tso_smapes))
    print("Average TSO (Results):", np.mean(tso_test_smapes))
    print("Average Naive2 sMAPE:", np.mean(naive2_smapes))
    print("Average NaiveS sMAPE:", np.mean(naives_smapes))