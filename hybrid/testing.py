import pandas as pd
import numpy as np

import sys
import os
import torch
import json

from ml.ml_helpers import pinball_loss, plot_test
from stats.stats_helpers import split_data, deseasonalise, reseasonalise
from stats.errors import sMAPE, MASE, OWA
from stats.naive import naive_2
from hybrid.hybrid import train_and_predict_i, train_and_predict_s
from hybrid.es_rnn_s import ES_RNN_S
from hybrid.es_rnn_i import ES_RNN_I


# General function used to do testing and tweaking
def run(demand_df, weather_df):
    # Optional command line argumenkts to specify year/season
    year = -1 if len(sys.argv) < 3 else int(sys.argv[2])
    season = -1 if len(sys.argv) < 4 else int(sys.argv[3])

    # Testing parameters
    window_size = 336
    output_size = 48
    plot = False
    ensemble = False
    skip_lstm = False
    init_params = True
    write_results = True
    file_location = str(os.path.abspath(os.path.dirname(__file__)))
    model = True  # True = Ingram, False = Smyl
    multiple = False  # Use multiple time series in Smyl's model
    weather = True  # Include weather data in the chosen model
    valid = True  # True = use validation set, False = use test set
    batch_first = True

    demand_features = demand_df.columns
    weather_features = weather_df.columns

    # If using weather features, add them to the demand_df
    if weather:
        for c in weather_df.columns:
            demand_df[c] = weather_df[c]

    # all_data = {Season: [Year1, ...]}
    all_data = split_data(demand_df)

    # Each year = [<- 12 week Train -> | <- 1 week Val. -> | <- 1 week Test ->]
    valid_sets = [
        all_data["Winter"][year if year >= 0 else 1][:-(7 * 24)],
        all_data["Spring"][year if year >= 0 else 0][:-(7 * 24)],
        all_data["Summer"][year if year >= 0 else 3][:-(7 * 24)],
        all_data["Autumn"][year if year >= 0 else 2][:-(7 * 24)]
    ]

    test_sets = [
        all_data["Winter"][year if year >= 0 else 1],
        all_data["Spring"][year if year >= 0 else 0],
        all_data["Summer"][year if year >= 0 else 3],
        all_data["Autumn"][year if year >= 0 else 2]
    ]

    if file_location == "/ddn/home/gkxx72/AdvancedResearchProject/dev/hybrid":
        res_base = "/ddn/home/gkxx72/AdvancedResearchProject/run/test_res/"
    else:
        res_base = "/Users/matt/Projects/AdvancedResearchProject/test/"

    if valid:
        data = valid_sets[season if season >= 0 else 2]
    else:
        data = test_sets[season if season >= 0 else 2]

    # Set the number of features
    if model:
        input_size = len(data.columns)  # My model, with or without weather
    elif weather:
        input_size = 1 + len(weather_df.columns)  # Smyl's model, with weather
    else:
        input_size = 1  # Smyl's model, without weather

    global_rates_dict = {
        "ingram": {10: 5e-3, 20: 1e-3, 30: 5e-4},
        "smyl_1": {10: 5e-3, 20: 1e-3, 30: 5e-4},
        "smyl_m": {10: 5e-3, 20: 1e-3, 30: 5e-4}
    }
    global_init_rates_dict = {
        "ingram": 0.008,
        "smyl_1": 0.008,
        "smyl_m": 0.008
    }
    local_rates_dict = {
        "ingram": {10: 2e-3, 20: 1e-3, 30: 5e-4},
        "smyl_1": {10: 1e-3, 20: 5e-4, 30: 1e-4},
        "smyl_m": {10: 2e-3, 20: 1e-3, 30: 5e-4}
    }
    local_init_rates_dict = {
        "ingram": 0.01,
        "smyl_1": 0.005,
        "smyl_m": 0.01
    }

    if model:
        global_init_lr = global_init_rates_dict["ingram"]
        global_rates = global_rates_dict["ingram"]
        local_init_lr = local_init_rates_dict["ingram"]
        local_rates = local_rates_dict["ingram"]
    else:
        if multiple:
            global_init_lr = global_init_rates_dict["smyl_m"]
            global_rates = global_rates_dict["smyl_m"]
            local_init_lr = local_init_rates_dict["smyl_m"]
            local_rates = local_rates_dict["smyl_m"]
        else:
            global_init_lr = global_init_rates_dict["smyl_1"]
            global_rates = global_rates_dict["smyl_1"]
            local_init_lr = local_init_rates_dict["smyl_1"]
            local_rates = local_rates_dict["smyl_1"]

    # Model hyper parameters
    num_epochs = 35
    hidden_size = 40
    num_layers = 4
    dilations = [1, 4, 24, 168]
    level_variability_penalty = 80
    percentile = 0.49
    loss_func = pinball_loss
    grad_clipping = 20
    auto_lr = False  # Automatically adjust learning rates
    variable_lr = True  # Use list of epoch/rate pairs
    auto_rate_threshold = 1.005  # If loss(x - 1) < 1.005 * loss(x) reduce rate
    min_epochs_before_change = 2
    residuals = tuple([[1, 3]])  # Residual connection from 2nd out -> 4th out
    seasonality = 168
    init_level_smoothing = int(sys.argv[4]) if len(sys.argv) >= 5 else 0
    init_seasonal_smoothing = int(sys.argv[5]) if len(sys.argv) >= 5 else 0

    test_model_week(data, output_size, input_size, hidden_size,
                    num_layers, batch_first, dilations, demand_features,
                    weather_features, seasonality, residuals, window_size,
                    level_variability_penalty, loss_func, num_epochs,
                    local_init_lr, global_init_lr, init_level_smoothing,
                    init_seasonal_smoothing, percentile, auto_lr,
                    variable_lr, auto_rate_threshold,
                    min_epochs_before_change, local_rates, global_rates,
                    grad_clipping, write_results, plot, year, season,
                    ensemble, multiple, skip_lstm, model, init_params,
                    res_base, weather)


# If output_size != 48 then this is broken. Pass in valid data or test
# data!! (i.e all up to the end of the valid section/test section).
def test_model_week(data, output_size, input_size, hidden_size,
                    num_layers, batch_first, dilations, demand_features,
                    weather_features, seasonality, residuals, window_size,
                    level_variability_penalty, loss_func, num_epochs,
                    local_init_lr, global_init_lr,
                    init_level_smoothing, init_seasonal_smoothing, percentile,
                    auto_lr, variable_lr, auto_rate_threshold,
                    min_epochs_before_change, local_rates, global_rates,
                    grad_clipping, write_results, plot, year, season,
                    ensemble, multi_ts, skip_lstm, model, init_params,
                    res_base, weather):

    # Arrays and dictionaries to hold the results
    es_rnn_predictions = []
    es_rnn_smapes = []
    es_rnn_mases = []
    naive2_predictions = []
    naive2_smapes = []
    naive2_mases = []
    actuals_mases = []
    actuals = []
    owas = []
    results = {i: {} for i in range(1, 8)}

    # Loop through each day in the week
    for i in range(8, 1, -1):

        # Figure out start and end points of the training/test data
        end_train = -(i * 24)
        start_test = -(i * 24 + window_size)
        end_test = -(i * 24 - output_size) if i != 2 else None
        train_data = data[:end_train]
        test_data = data[start_test:end_test]
        mase_data = data["total load actual"][:end_test]

        # Initialise (or not) the parameters
        if init_params:
            init_seas = {}
            init_l_smooth = {}
            init_s_smooth = {}
            for f in demand_features:
                deseas, indic = deseasonalise(train_data[f], 168,
                                              "multiplicative")
                init_seas[f] = indic
                init_l_smooth[f] = init_level_smoothing
                init_s_smooth[f] = init_seasonal_smoothing

                if f == "total load actual":
                    train_deseas = deseas
                    indices = indic
        else:
            deseas, indic = deseasonalise(train_data["total load actual"],
                                          168, "multiplicative")
            train_deseas = deseas
            indices = indic
            init_seas = None
            init_l_smooth = None
            init_s_smooth = None

        # Calculate the batch size
        batch_size = len(train_data["total load actual"]) - window_size - \
                     output_size + 1

        # Create a new model. Either mine or Smyl's
        if model:
            lstm = ES_RNN_I(
                output_size, input_size, batch_size, hidden_size,
                num_layers, demand_features, weather_features, seasonality,
                dropout=0, cell_type='LSTM', batch_first=batch_first,
                dilations=dilations, residuals=residuals,
                init_seasonality=init_seas, init_level_smoothing=init_l_smooth,
                init_seas_smoothing=init_s_smooth
            ).double()
        else:
            lstm = ES_RNN_S(
                output_size, input_size, batch_size, hidden_size, num_layers,
                demand_features, weather_features, seasonality, dropout=0,
                cell_type='LSTM',
                batch_first=batch_first, dilations=dilations,
                residuals=residuals, init_seasonality=init_seas,
                init_level_smoothing=init_l_smooth,
                init_seas_smoothing=init_s_smooth
            ).double()

        # Register gradient clipping function
        for p in lstm.parameters():
            p.register_hook(
                lambda grad: torch.clamp(grad, -grad_clipping, grad_clipping))

        # Set model in training mode
        lstm.train()

        print("----- TEST", str(9 - i), "-----")

        # Train the model. Discard prediction here (used in proper function)
        if model:
            _, losses = train_and_predict_i(
                lstm, train_data, window_size,
                output_size,
                level_variability_penalty,
                loss_func,
                num_epochs, local_init_lr,
                global_init_lr,
                percentile,
                auto_lr, variable_lr,
                auto_rate_threshold,
                min_epochs_before_change,
                test_data, local_rates,
                global_rates,
                ensemble, weather
            )
        else:
            _, losses = train_and_predict_s(
                lstm, train_data, window_size, output_size,
                level_variability_penalty, loss_func, num_epochs,
                local_init_lr, global_init_lr, percentile, auto_lr,
                variable_lr, auto_rate_threshold, min_epochs_before_change,
                test_data, local_rates, global_rates, ensemble, multi_ts,
                skip_lstm, weather
            )

        # Set model into evaluation mode
        lstm.eval()

        # Make ES_RNN_S Prediction
        prediction, actual, out_levels, out_seas, all_levels, all_seasons, \
        rnn_out = lstm.predict(test_data, window_size, output_size, weather)

        # Convert test data to correct form for results saving
        test_data = torch.tensor(test_data["total load actual"],
                                 dtype=torch.double)

        # [[<- 48 ->],] generated, so remove the dimension
        prediction = pd.Series(prediction.squeeze(0).detach().tolist())
        actual = pd.Series(actual.squeeze(0).detach().tolist())
        out_levels = out_levels.squeeze(0).detach().tolist()
        out_seas = out_seas.squeeze(0).detach().tolist()
        all_levels = [l.detach().item() for l in all_levels]
        all_seasons = [s.detach().item() for s in all_seasons]
        rnn_out = rnn_out.squeeze(0).detach().tolist()

        # Make Naive2 Prediction
        naive_fit_forecast = reseasonalise(
            naive_2(train_deseas, output_size), indices, "multiplicative"
        )
        naive_prediction = naive_fit_forecast[-output_size:].reset_index(
            drop=True)

        # Calculate errors
        es_rnn_smape = sMAPE(prediction, actual)
        es_rnn_mase = MASE(prediction, mase_data, 168, output_size)
        naive_smape = sMAPE(naive_prediction, actual)
        naive_mase = MASE(naive_prediction, mase_data, 168, output_size)
        owa = OWA(naive_smape, naive_mase, es_rnn_smape, es_rnn_mase)

        # Save values
        es_rnn_smapes.append(es_rnn_smape)
        es_rnn_mases.append(es_rnn_mase)
        naive2_smapes.append(naive_smape)
        naive2_mases.append(naive_mase)
        es_rnn_predictions.append(prediction)
        naive2_predictions.append(naive_prediction)
        actuals.append(actual)
        actuals_mases.append(mase_data)
        owas.append(owa)

        # Print results
        print("***** Test Results *****")
        print("ES-RNN sMAPE:", es_rnn_smape)
        print("Naive2 sMAPE:", naive_smape)
        print("ES-RNN MASE:", es_rnn_mase)
        print("Naive2 MASE:", naive_mase)
        print("OWA", owa)
        print("")

        # Save all results
        results[9 - i]["test_data"] = test_data.tolist()
        results[9 - i]["ESRNN_prediction"] = prediction.to_list()
        results[9 - i]["Naive2_prediction"] = naive_prediction.to_list()
        results[9 - i]["all_levels"] = all_levels
        results[9 - i]["out_levels"] = out_levels
        results[9 - i]["all_seas"] = all_seasons
        results[9 - i]["out_seas"] = out_seas
        results[9 - i]["rnn_out"] = rnn_out
        results[9 - i]["level_smoothing"] = float(
            lstm.level_smoothing_coeffs["total load actual"].data
        )
        results[9 - i]["seasonality_smoothing"] = float(
            lstm.seasonality_smoothing_coeffs["total load actual"].data
        )
        results[9 - i]["losses"] = losses

        sys.stderr.flush()
        sys.stdout.flush()

    # Print final results
    owas_np = np.array(owas)
    num_improved = len(owas_np[owas_np < 1.0])
    avg_improve = float(np.around(owas_np[owas_np < 1.0].mean(), decimals=3))
    avg_decline = float(np.around(owas_np[owas_np >= 1.0].mean(), decimals=3))
    avg_owa = float(np.around(np.mean(owas), decimals=3))
    print("***** OVERALL RESULTS *****")
    print("Average OWA:", avg_owa)
    print("No. Improved:", num_improved)
    print("Avg. Improvement:", avg_improve)
    print("Avg. Decline:", avg_decline)

    sys.stderr.flush()
    sys.stdout.flush()

    # Make note of final results
    results["overall"] = {
        "avg_owa": avg_owa,
        "num_improved": num_improved,
        "avg_improvement": avg_improve,
        "avg_decline": avg_decline
    }

    # Write results (NCC)
    if write_results:
        season_dict = {0: "_winter", 1: "_spring", 2: "_summer", 3: "_autumn"}
        name = sys.argv[1]
        if len(sys.argv) == 2:
            filename = name + ".txt"
        elif len(sys.argv) == 3:
            filename = name + "_year_" + str(year) + ".txt"
        elif len(sys.argv) == 4:
            s = season_dict[season]
            filename = name + "_year_" + str(year) + s + ".txt"
        elif len(sys.argv) == 6:
            s = season_dict[season]
            filename = name + "_year_" + str(year) + s + "_" +\
                       str(init_level_smoothing) + "_" +\
                       str(init_seasonal_smoothing) + ".txt"
        else:
            filename = "test.txt"

        res_path = os.path.join(res_base, filename)
        with open(res_path, "w") as res:
            json.dump(results, res)

    if plot:
        plot_test(results, window_size, output_size, print_results=True)

