import json
import os.path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.plotting import register_matplotlib_converters
from sklearn.preprocessing import MinMaxScaler

from hybrid import hybrid
from ml import ml_helpers
from stats import arima, exponential_smoothing, naive, theta, errors, \
    stats_helpers


# Entry point
def main():
    demand_df = load_demand_data()
    weather_df = load_weather_data()
    test(demand_df, weather_df, int(sys.argv[1]), int(sys.argv[2]))


# Loads the energy demand data and performs interpolation
def load_demand_data():
    base = os.path.abspath(os.path.dirname(__file__))
    filename = os.path.join(base, "data/spain/energy_dataset.csv")
    col_list = ["time", "generation fossil gas",
                "generation fossil hard coal",
                "generation fossil oil",
                "generation hydro run-of-river and poundage",
                "generation hydro water reservoir", "total load forecast",
                "total load actual", "price day ahead", "price actual",
                ]

    df = pd.read_csv(filename, parse_dates=["time"],
                     infer_datetime_format=True, usecols=col_list)
    df = df.set_index('time').asfreq('H')
    df.replace(0, np.NaN, inplace=True)
    df.interpolate(inplace=True)

    return df


# Loads weather data and calculates additional metrics
def load_weather_data():
    base = os.path.abspath(os.path.dirname(__file__))
    filename = os.path.join(base, "data/spain/weather_features.csv")
    weather_cols = ["dt_iso", "city_name", "temp", "humidity", "wind_speed",
                    "pressure"]
    weather_df = pd.read_csv(filename, parse_dates=["dt_iso"],
                             infer_datetime_format=True,
                             usecols=weather_cols)

    # Remove duplicates in the weather data
    weather_df = weather_df.drop_duplicates(["dt_iso", "city_name"])

    # Reference temperature (C) for latent enthalpy calculation
    hd_ref = ml_helpers.c_to_k(15.5)
    cd_ref = ml_helpers.c_to_k(23.5)
    le_ref = 25.6

    # Calculate the HD and CD for all
    weather_df["heating_degree"] = weather_df["temp"].apply(
        lambda x: hd_ref - x if x < hd_ref else 0
    )
    weather_df["cooling_degree"] = weather_df["temp"].apply(
        lambda x: x - cd_ref if x > cd_ref else 0
    )
    weather_df["latent_enthalpy"] = weather_df.apply(
        lambda row: ml_helpers.latent_enthalpy(row, le_ref), axis=1
    )

    # Drop the pressure column, only required to calculate latent_enthalpy
    weather_df.drop(["pressure"], axis=1, inplace=True)

    # Average weather across all cities
    avg_weather_df = weather_df.groupby(["dt_iso"]).mean()

    # Normalise all values into the correct range
    scaler = MinMaxScaler((-1, 1))
    avg_scaled_df = pd.DataFrame(scaler.fit_transform(avg_weather_df),
                                 columns=avg_weather_df.columns,
                                 index=avg_weather_df.index)

    return avg_scaled_df


# Testing procedure. Tests given model on given season and saves results
def test(demand_df, weather_df, season_no, model_no):
    demand_features = demand_df.columns
    weather_features = weather_df.columns

    # Add the weather data to the demand data
    for c in weather_df.columns:
        demand_df[c] = weather_df[c]

    # Testing hyper-parameters
    seasonality = 168
    forecast_length = 48

    # For the ES_RNN_S, for each test, train the model num_ensemble
    # times and average the predictions. Further, if internal ensembling is
    # also specified, each prediction from the model will actually be the
    # average of the predictions from the last 5 epochs
    ensemble = False
    num_ensemble = 3

    # True = use final week for testing, False = use penultimate week for
    # validation
    testing = True

    # Model No.: [Function, Name, Deseasonalise?, Additional Parameters,
    # Return Parameters, Number of Repetitions]
    test_dict = {
        1: [naive.naive_1, 'Naive1', False, None, False, 10],

        2: [naive.naive_2, 'Naive2', True, None, False, 10],

        3: [naive.naive_s, 'NaiveS', False, [seasonality], False, 10],

        4: [exponential_smoothing.ses, 'SES', True, None, True, 10],

        5: [exponential_smoothing.holt, 'Holt', True, None, True, 10],

        6: [exponential_smoothing.damped, 'Damped', True, None, True, 10],

        7: [exponential_smoothing.holt_winters, 'Holt-Winters', False,
            [seasonality], True, 10],

        8: [exponential_smoothing.comb, 'Comb', True, None, False, 10],

        9: [arima.arima, 'ARIMA', True, "-- See arima_orders --", True, 10],

        10: [arima.sarima, 'SARIMA', False, "-- See sarima_orders --", True,
             1],

        11: [arima.auto, 'Auto', False, [168], True, 1],

        12: [theta.theta, 'Theta', True, None, True, 10],

        13: [None, 'TSO', False, None, False, 1],

        14: [hybrid.es_rnn_s, 'ES-RNN-S', False, [
            seasonality, demand_features, weather_features, False, ensemble,
            True], False, 1],

        15: [hybrid.es_rnn_s, 'ES-RNN-SW', False, [
            seasonality, demand_features, weather_features, True, ensemble,
            True], False, 1],

        16: [hybrid.es_rnn_s, 'ES-RNN-D', False, [
            seasonality, demand_features, weather_features, False, ensemble,
            False], False, 1],

        17: [hybrid.es_rnn_s, 'ES-RNN-DW', False, [
            seasonality, demand_features, weather_features, True, ensemble,
            False], False, 1],

        18: [hybrid.es_rnn_i, 'ES-RNN-I', False, [
            seasonality, demand_features, weather_features, False, ensemble],
            False, 1],

        19: [hybrid.es_rnn_i, 'ES-RNN-IW', False, [
            seasonality, demand_features, weather_features, True, ensemble],
            False, 1],
    }

    # Optimal SARIMA orders for each season
    sarima_orders = {
        1: [(2, 0, 0), (1, 0, 1, 168)],
        2: [(2, 0, 1), (1, 0, 1, 168)],
        3: [(2, 0, 1), (1, 0, 1, 168)],
        4: [(1, 0, 2), (1, 0, 1, 168)]
    }

    # Optimum ARIMA Parameters (automatically checked, using the
    # identify_arima function)
    arima_orders = {
        1: [[(2, 0, 0)], [(2, 0, 0)], [(1, 0, 2)], [(2, 0, 2)]],
        2: [[(2, 0, 0)], [(2, 0, 0)], [(2, 0, 2)], [(2, 0, 2)]],
        3: [[(1, 0, 1)], [(2, 0, 2)], [(2, 0, 2)], [(2, 0, 2)]],
        4: [[(2, 0, 1)], [(2, 0, 2)], [(2, 0, 2)], [(2, 0, 2)]],
    }

    seas_dict = {1: "Spring", 2: "Summer", 3: "Autumn", 4: "Winter"}

    # Get the parameters for the model
    model_func, model_name, deseasonalise, params, ret_params, num_reps = \
        test_dict[model_no]
    error_pairs = [("sMAPE", errors.sMAPE), ("RMSE", errors.RMSE),
                   ("MASE", errors.MASE), ("MAE", errors.MAE)]

    # Build empty data structures to hold results, naive results, forecasts and
    # fitted parameters
    results = {e: {r: {y: {t: [0] * forecast_length for t in range(1, 8)
                           } for y in range(1, 5)
                       } for r in range(1, num_reps + 1)
                   } for e in list(zip(*error_pairs))[0] + tuple(["OWA"])
               }

    n_results = {e: {r: {y: {t: [0] * forecast_length for t in range(1, 8)
                             } for y in range(1, 5)
                         } for r in range(1, num_reps + 1)
                     } for e in list(zip(*error_pairs))[0] + tuple(["OWA"])
                 }

    forecasts = {y: {r: {t: [] for t in range(1, 8)
                         } for r in range(1, num_reps + 1)
                     } for y in range(1, 5)
                 }

    final_params = {y: [] for y in range(1, 5)}

    all_data = stats_helpers.split_data(demand_df)
    years_df = all_data[seas_dict[season_no]]

    # The final 7 days are reserved for final testing
    if testing:
        years = [years_df[i]["total load actual"] for i in range(4)]
    else:
        years = [years_df[i]["total load actual"][:-7 * 24] for i in range(4)]

    # Loop through the years
    for y_index, y in enumerate(years):

        # Specify correct ARIMA parameters
        if model_no == 9:
            params = arima_orders[season_no][y_index]
        if model_no == 10:
            params = sarima_orders[season_no]

        # Loop through the week of tests
        for t in range(8, 1, -1):
            # Get training and test data. Change y[:-0] to y[:None].
            train_end = -(t * 24)
            test_end = -(t * 24 - forecast_length) if t > 2 else None
            train_data = y[:train_end]
            test_data = y[train_end:test_end]
            tso_data = years_df[y_index]["total load forecast"][
                       train_end:test_end]

            # Deseasonalise, always required for Naive2
            train_deseas, indices = stats_helpers.deseasonalise(
                train_data, seasonality, "multiplicative"
            )

            # Generate naïve forecast for use in MASE calculation
            naive_fit_forecast = stats_helpers.reseasonalise(
                naive.naive_2(train_deseas, forecast_length),
                indices,
                "multiplicative"
            )
            naive_forecast = naive_fit_forecast[-forecast_length:]

            # Use deseasonalised data if needed
            if deseasonalise:
                train_data = train_deseas

            # Loop through the repetitions
            for r in range(1, num_reps + 1):

                # Handle the hybrid model individually
                if model_no > 13:
                    # Hybrid model requires the dataframe and extra data
                    if testing:
                        test_end = -((t - 2) * 24) if t > 2 else None
                    else:
                        test_end = -((t + 5) * 24)  # Think about it, see notes
                    train_data = years_df[y_index][:test_end]

                    # Generate ensemble if we are ensembling
                    if ensemble:
                        pred_ensemble = []
                        for i in range(num_ensemble):
                            pred = model_func(train_data, forecast_length,
                                              *params)
                            pred_ensemble.append(pred)

                        forec_results = pd.Series(np.mean(pred_ensemble,
                                                          axis=0))
                    else:
                        forec_results = model_func(train_data, forecast_length,
                                                   *params)

                # Handle the TSO forecast individually (no forecast method)
                elif model_no == 13:
                    forec_results = tso_data

                # Handle the statistical models. Fit the model and forecast,
                # with additional params if needed
                else:
                    if params is not None:
                        forec_results = model_func(train_data, forecast_length,
                                                   *params)
                    else:
                        forec_results = model_func(train_data, forecast_length)

                # Split results into fit-forecast and parameters if the
                # model also returned the values of its fitted parameters
                if ret_params:
                    fit_forecast, fit_params = forec_results
                else:
                    fit_forecast = forec_results

                # Reseasonalise if necessary
                if deseasonalise:
                    fit_forecast = stats_helpers.reseasonalise(
                        fit_forecast, indices, "multiplicative"
                    )

                # Select only the forecast, not the fitted values
                forecast = fit_forecast[-forecast_length:]

                # Loop through the error functions
                for e_name, e_func in error_pairs:

                    # Loop through the lead times
                    for l in range(1, forecast_length + 1):
                        if e_name == "MASE":
                            end = None if (t == 2 and l == 48) else -(t * 24
                                                                      - l)
                            error = e_func(forecast[:l], y[:end],
                                           seasonality, l)
                            n_error = e_func(naive_forecast[:l], y[:end],
                                             seasonality, l)
                        else:
                            error = e_func(forecast[:l], test_data[:l])
                            n_error = e_func(naive_forecast[:l], test_data[:l])

                        # Save error results for all lead times
                        results[e_name][r][y_index + 1][t - 1][l - 1] = error
                        n_results[e_name][r][y_index + 1][t - 1][l - 1] = \
                            n_error

                # Save 48 hour forecast
                forecasts[y_index + 1][r][t - 1] = forecast.to_list()

                # Save model params only for final repetition and train time
                if r == num_reps and t == 2 and ret_params:
                    final_params[y_index + 1] = fit_params
            print("Year:", str(y_index), "Test:", str(t), "Finished")

    # Calculate OWA for all forecasts
    for r in range(1, num_reps + 1):
        for y in range(1, 5):
            for t in range(1, 8):
                for l in range(0, forecast_length):
                    results["OWA"][r][y][t][l] = errors.OWA(
                        n_results["sMAPE"][r][y][t][l],
                        n_results["MASE"][r][y][t][l],
                        results["sMAPE"][r][y][t][l],
                        results["MASE"][r][y][t][l],
                    )

    # Average the single 48 hour forecast results
    all_res = []
    for r in range(1, num_reps + 1):
        for y in range(1, 5):
            for t in range(1, 8):
                all_res.append(results["OWA"][r][y][t][forecast_length - 1])

    mean = np.around(np.mean(all_res), decimals=3)
    std = np.around(np.std(all_res), decimals=3)

    # Save averaged single 48 forecast results
    file_path = os.path.abspath(os.path.dirname(__file__))
    res_path = os.path.join(file_path, "results/results_1.txt")
    with open(res_path) as file:
        results_1 = json.load(file)

    results_1[seas_dict[season_no]][model_name] = [mean, std]

    with open(res_path, "w") as file:
        json.dump(results_1, file)

    # Average the lead time results for OWA
    all_res_owa = {l: [] for l in range(1, forecast_length + 1)}
    for r in range(1, num_reps + 1):
        for y in range(1, 5):
            for t in range(1, 8):
                for l in range(1, forecast_length + 1):
                    all_res_owa[l].append(results["OWA"][r][y][t][l - 1])
    for l in all_res_owa.keys():
        all_res_owa[l] = np.around(np.mean(all_res_owa[l]), decimals=3)

    # Average the lead time results for sMAPE
    all_res_smape = {l: [] for l in range(1, forecast_length + 1)}
    for r in range(1, num_reps + 1):
        for y in range(1, 5):
            for t in range(1, 8):
                for l in range(1, forecast_length + 1):
                    all_res_smape[l].append(results["sMAPE"][r][y][t][l - 1])
    for l in all_res_smape.keys():
        all_res_smape[l] = np.around(np.mean(all_res_smape[l]), decimals=3)

    # Average the lead time results for MASE
    all_res_mase = {l: [] for l in range(1, forecast_length + 1)}
    for r in range(1, num_reps + 1):
        for y in range(1, 5):
            for t in range(1, 8):
                for l in range(1, forecast_length + 1):
                    all_res_mase[l].append(results["MASE"][r][y][t][l - 1])
    for l in all_res_mase.keys():
        all_res_mase[l] = np.around(np.mean(all_res_mase[l]), decimals=3)

    # Save the lead time results for OWA
    res_path = os.path.join(file_path, "results/results_48_seasons_owa.txt")
    with open(res_path) as file:
        results_48 = json.load(file)
    for l in all_res_owa.keys():
        results_48[str(l)][model_name][season_no - 1] = all_res_owa[l]
    with open(res_path, "w") as file:
        json.dump(results_48, file)

    # Save the lead time results for sMAPE
    res_path = os.path.join(file_path,
                            "results/results_48_seasons_smape.txt")
    with open(res_path) as file:
        results_48 = json.load(file)
    for l in all_res_smape.keys():
        results_48[str(l)][model_name][season_no - 1] = all_res_smape[l]
    with open(res_path, "w") as file:
        json.dump(results_48, file)

    # Save the lead time results for MASE
    res_path = os.path.join(file_path,
                            "results/results_48_seasons_mase.txt")
    with open(res_path) as file:
        results_48 = json.load(file)
    for l in all_res_mase.keys():
        results_48[str(l)][model_name][season_no - 1] = all_res_mase[l]
    with open(res_path, "w") as file:
        json.dump(results_48, file)

    # Save the raw forecasts and results
    res_filename = seas_dict[season_no] + "_" + model_name + "_results.txt"
    forec_filename = seas_dict[season_no] + "_" + model_name + "_forecasts.txt"
    res_path = os.path.join(file_path, "results/" + res_filename)
    forec_path = os.path.join(file_path, "results/" + forec_filename)

    with open(res_path, "w") as file:
        json.dump(results, file)
    with open(forec_path, "w") as file:
        json.dump(forecasts, file)

    # Save the parameters (if model returns parameters)
    if ret_params:
        param_path = os.path.join(file_path, "results/params.txt")
        with open(param_path) as file:
            saved_params = json.load(file)

        for y in range(1, 5):
            saved_params[model_name][str(season_no)][str(y)] = final_params[
                y]

        with open(param_path, "w") as file:
            json.dump(saved_params, file)


# Delete all files in results/, then run this to regenerate empty results files
def reset_results_files():
    file_path = os.path.abspath(os.path.dirname(__file__))

    # Parameters
    param_path = os.path.join(file_path, "results/params.txt")
    params = {m: {s: {y: [] for y in range(1, 5)} for s in range(1, 5)} for
              m in ["SES", "Holt", "Damped", "Holt-Winters", "ARIMA", "SARIMA",
                    "Auto", "Theta"]}
    with open(param_path, "w") as f:
        json.dump(params, f)

    # Results 48 (all seasons) OWA
    res48s_path = os.path.join(file_path, "results/results_48_seasons_owa.txt")
    methods = ["Naive1", "Naive2", "NaiveS", "SES", "Holt", "Damped",
               "Holt-Winters", "Comb", "ARIMA", "SARIMA", "Auto", "Theta",
               "TSO", "ES-RNN-S", "ES-RNN-SW", "ES-RNN-D", "ES-RNN-DW",
               "ES-RNN-I", "ES-RNN-IW"]
    res48s = {l: {m: [0, 0, 0, 0] for m in methods} for l in range(1, 49)}
    with open(res48s_path, "w") as f:
        json.dump(res48s, f)

    # Results 48 (all seasons) sMAPE
    res48s_path = os.path.join(file_path,
                               "results/results_48_seasons_smape.txt")
    res48s = {l: {m: [0, 0, 0, 0] for m in methods} for l in range(1, 49)}
    with open(res48s_path, "w") as f:
        json.dump(res48s, f)

    # Results 48 (all seasons) MASE
    res48s_path = os.path.join(file_path,
                               "results/results_48_seasons_mase.txt")
    res48s = {l: {m: [0, 0, 0, 0] for m in methods} for l in range(1, 49)}
    with open(res48s_path, "w") as f:
        json.dump(res48s, f)

    # Results 48 OWA
    res48_path = os.path.join(file_path, "results/results_48_owa.txt")
    res48 = {l: {m: 0 for m in methods} for l in range(1, 49)}
    with open(res48_path, "w") as f:
        json.dump(res48, f)

    # Results 48 sMAPE
    res48_path = os.path.join(file_path, "results/results_48_smape.txt")
    res48 = {l: {m: 0 for m in methods} for l in range(1, 49)}
    with open(res48_path, "w") as f:
        json.dump(res48, f)

    # Results 48 MASE
    res48_path = os.path.join(file_path, "results/results_48_mase.txt")
    res48 = {l: {m: 0 for m in methods} for l in range(1, 49)}
    with open(res48_path, "w") as f:
        json.dump(res48, f)

    # Results 1
    res1_path = os.path.join(file_path, "results/results_1.txt")
    seasons = ["Spring", "Summer", "Winter", "Autumn", "Average"]
    res1 = {s: {m: [0, 0] for m in methods} for s in seasons}
    with open(res1_path, "w") as f:
        json.dump(res1, f)


# Output options for pandas and matplotlib
register_matplotlib_converters()
pd.options.mode.chained_assignment = None
pd.set_option('display.max_columns', 500)  # Shows all columns
pd.set_option('display.expand_frame_repr', False)  # Prevents line break
plt.style.use('seaborn-deep')

# Run main()
main()
