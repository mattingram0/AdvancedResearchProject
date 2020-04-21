import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import sys
import os
import json

from ml.ml_helpers import pinball_loss, plot_test, plot_es_comp, \
    plot_gen_forecast
from stats.stats_helpers import split_data, deseasonalise, reseasonalise
from stats.errors import sMAPE, MASE, OWA
from stats.naive import naive_2
from math import sqrt

from hybrid.es_rnn_s import ES_RNN_S
from hybrid.es_rnn_i import ES_RNN_I


# Give the model the training data, the forecast length and the seasonality,
# and this function trains the model and makes a single prediction
# Must pass in the training data plus the following forecast_length data.
# Note that the extra forecast_length of data (i.e the test data) doesn't
# get used, we just need the data to be the correct length (just the way
# I've coded it)
# TODO THIS OF COURSE IS WRONG
def es_rnn(data, forecast_length, seasonality, ensemble, multi_ts, skip_lstm):
    # Hyperparameters - determined experimentally
    num_epochs = 30
    init_learning_rate = 0.002
    input_size = 1
    hidden_size = 40
    num_layers = 4
    dilations = [1, 4, 24, 168]
    level_variability_penalty = 100
    percentile = 0.49
    loss_func = pinball_loss
    grad_clipping = 20
    auto_lr = False
    variable_lr = True
    variable_rates = {15: 1e-3, 30: 5e-4, 35: 3e-4}
    auto_rate_threshold = 1.005
    min_epochs_before_change = 2
    residuals = tuple([[1, 3]])
    window_size = 336
    batch_first = True

    # Split the data
    train_data = data[:-forecast_length]
    forecast_data = data[-(forecast_length + window_size):]
    forecast_input = torch.tensor(forecast_data["total load actual"],
                                  dtype=torch.double)
    batch_size = len(train_data) - window_size - forecast_length + 1
    features = train_data.columns

    # Estimate the seasonal indices and inital smoothing levels
    init_seas = {}
    init_l_smooth = {}
    init_s_smooth = {}
    for c in data.columns:
        deseas, indic = deseasonalise(train_data[c], 168, "multiplicative")
        init_seas[c] = indic
        init_l_smooth[c] = -0.8
        init_s_smooth[c] = -0.2

    # Create the model
    lstm = ES_RNN_S(
        forecast_length, input_size, batch_size, hidden_size, num_layers,
        batch_first=batch_first, dilations=dilations, demand_features=features,
        seasonality_1=None, seasonality=seasonality, residuals=residuals,
        init_seasonality=init_seas, init_level_smoothing=init_l_smooth,
        init_seas_smoothing=init_s_smooth
    ).double()

    # Register gradient clipping function
    for p in lstm.parameters():
        p.register_hook(
            lambda grad: torch.clamp(grad, -grad_clipping, grad_clipping))

    # Set model in training mode
    lstm.train()

    # Train the model, and make a (possibly ensembled) prediction.
    prediction = train_and_predict_s(
        lstm, train_data, window_size, forecast_length,
        level_variability_penalty, loss_func, num_epochs,
        init_learning_rate, percentile, auto_lr, variable_lr,
        auto_rate_threshold, min_epochs_before_change, forecast_input,
        variable_rates, ensemble, multi_ts, skip_lstm)

    # plot_es_comp(train_data["total load actual"].tolist(), lstm.w_seasons,
    #              lstm.levels)
    #
    # # Set model in evaluation (prediction) mode
    # lstm.eval()
    #
    # results = lstm.predict(forecast_input, window_size, 48)
    # pred, act, out_level, out_seas, all_levels, all_seas, lstm_out = results
    # print(type(forecast_input), forecast_input.size())
    # print(type(pred), pred.size())
    # print(type(act), act.size())
    # print(type(out_level), out_level.size())
    # print(type(out_seas), out_seas.size())
    # print(type(all_levels), len(all_levels))
    # print(type(all_seas), len(all_seas))
    # print(type(lstm_out), lstm_out.size())
    # print(type(inputs), inputs.size())

    # plot_gen_forecast(forecast_input.tolist(), act.squeeze().tolist(),
    #                   all_seas, out_seas.squeeze().tolist(), all_levels,
    #                   out_level.squeeze().tolist(), inputs.view(-1).tolist(),
    #                   lstm_out.squeeze().tolist(), pred.squeeze().tolist())

    # Return prediction
    return prediction


# General function used to do testing and tweaking
def run(demand_df, weather_df):
    # Optional command line arguments to specify year/season
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
    model = False  # True = Ingram, False = Smyl
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

    if file_location == "/ddn/home/gkxx72/AdvancedResearchProject/dev":
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

    # Model hyper parameters
    num_epochs = 108
    local_init_lr = 0.01
    global_init_lr = 0.005
    hidden_size = 40
    num_layers = 4
    dilations = [1, 4, 24, 168]
    level_variability_penalty = 80
    percentile = 0.49
    loss_func = pinball_loss
    grad_clipping = 20
    auto_lr = False  # Automatically adjust learning rates
    variable_lr = True  # Use list of epoch/rate pairs
    global_rates = {10: 1e-3, 20: 5e-4, 30: 1e-4}
    local_rates = {10: 5e-3, 20: 1e-3, 30: 5e-4}
    auto_rate_threshold = 1.005  # If loss(x - 1) < 1.005 * loss(x) reduce rate
    min_epochs_before_change = 2
    residuals = tuple([[1, 3]])  # Residual connection from 2nd out -> 4th out
    seasonality = 168
    init_level_smoothing = -1
    init_seasonal_smoothing = 1

    lr = [0.0001 * i for i in range(1, 10) for _ in range(3)] + \
         [0.001 * i for i in range(1, 10) for _ in range(3)] + \
         [0.01 * i for i in range(1, 10) for _ in range(3)] + \
         [0.1 * i for i in range(1, 10) for _ in range(3)]

    global_rates = {i: r for i, r in enumerate(lr)}
    local_rates = {i: r for i, r in enumerate(lr)}
    global_init_lr = global_rates[0]
    local_init_lr = local_rates[0]

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
    for i in range(2, 1, -1):

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
            lr = str(local_init_lr).split(".")[1]
            filename = name + "_year_" + str(year) + s + "_" + lr + ".txt"
        else:
            filename = "test.txt"

        res_path = os.path.join(res_base, filename)
        with open(res_path, "w") as res:
            json.dump(results, res)

    if plot:
        plot_test(results, window_size, output_size, print_results=True)


def train_and_predict_i(lstm, data, window_size, output_size, lvp,
                        loss_func, num_epochs, local_init_lr, global_init_lr,
                        percentile, auto_lr, variable_lr, auto_rt,
                        min_epochs_since_change, forecast_input,
                        local_rates, global_rates, ensemble, weather):

    local_params = []
    global_params = []

    for n, p in lstm.named_parameters():
        if n.startswith("drnn.rnn_layer") or n.startswith(
                "linear") or n.startswith("tanh"):
            global_params.append(p)  # Handle the global parameters
        else:
            local_params.append(p)  # Handle the time-series specific params

    local_optimizer = torch.optim.Adam(params=local_params, lr=local_init_lr)
    global_optimizer = torch.optim.Adam(params=global_params,
                                        lr=global_init_lr)

    num_epochs_since_change = 0
    rate_changed = False
    prev_loss = 0
    dynamic_learning_rate = local_init_lr
    losses = {f: {l: [] for l in ["RNN", "LVP", "Total"]} for f in
              lstm.demand_features}
    pred_ensemble = []

    # Loop through number of epochs amount of times
    for epoch in range(num_epochs):
        print("Epoch:", epoch)
        outs, labels, level_var_losses = lstm(
            data, window_size, output_size, weather, lvp
        )
        loss = loss_func(outs, labels, percentile)
        rnn_loss = loss.item()

        for f in lstm.demand_features:
            # Calculate the total loss
            loss += level_var_losses[f]

            # Save losses
            losses[f]["RNN"].append(rnn_loss)
            losses[f]["LVP"].append(level_var_losses[f].item())
            losses[f]["Total"].append(loss.item())

            # Print loss information
            print("%s: LVP - %1.5f, RNN Loss - %1.5f Total Loss "
                  "- %1.5f" % (f, level_var_losses[f].item(), rnn_loss,
                               loss.item()))

        # Update the parameters, local and global
        local_optimizer.zero_grad()
        global_optimizer.zero_grad()
        loss.backward()
        local_optimizer.step()
        global_optimizer.step()

        # Update local optimizer learning rates if using variable lrs
        if variable_lr:
            if epoch in list(local_rates.keys()):
                for param_group in local_optimizer.param_groups:
                    param_group['lr'] = local_rates[epoch]

                print("Changed Local Learning Rate to:",
                    str(local_rates[epoch]))

            if epoch in list(global_rates.keys()):
                for param_group in global_optimizer.param_groups:
                    param_group['lr'] = global_rates[epoch]

                print("Changed Global Learning Rate to:",
                      str(global_rates[epoch]))

        # Update local optimizer lr if using automatic learning rates
        if auto_lr and num_epochs_since_change >= min_epochs_since_change:
            if prev_loss < auto_rt * loss:
                dynamic_learning_rate /= sqrt(10)
                rate_changed = True
                num_epochs_since_change = 0
                print("Loss Ratio:", prev_loss / loss)
                print("Changed Learning Rate to:", dynamic_learning_rate)

                for param_group in local_optimizer.param_groups:
                    param_group['lr'] = dynamic_learning_rate

                for param_group in global_optimizer.param_groups:
                    param_group['lr'] = dynamic_learning_rate
            else:
                rate_changed = False
        else:
            rate_changed = False

        prev_loss = loss.item()

        # If we didn't change the learning rate, make note of this
        if not rate_changed:
            num_epochs_since_change += 1

        # If we are ensembling, generate the ensemble of forecasts
        if ensemble and (num_epochs - epoch <= 5):
            prediction, *_ = lstm.predict(forecast_input, window_size,
                                          output_size, weather)
            pred_ensemble.append(prediction.squeeze(0).detach().tolist())
        else:
            if epoch == num_epochs - 1:
                prediction, *_ = lstm.predict(forecast_input, window_size,
                                              output_size, weather)
                prediction = pd.Series(prediction.squeeze(0).detach().tolist())

        sys.stderr.flush()
        sys.stdout.flush()

    if ensemble:
        return pd.Series(np.mean(pred_ensemble, axis=0)), losses
    else:
        return prediction, losses


# Trains a model and generates a prediction. If ensemble=True, the forecasts
# from the final 5 epochs are used. If we wish to use the extra information
# that the ES_RNN_S.predict() function returns, we can call it directly.
def train_and_predict_s(lstm, data, window_size, output_size, lvp, loss_func,
                        num_epochs, local_init_lr, global_init_lr,
                        percentile, auto_lr, variable_lr, auto_rt,
                        min_epochs_since_change, forecast_input,
                        local_variable_rates, global_variable_rates,
                        ensemble, multi_ts, skip_lstm, weather):

    parameter_dict = {c: [] for c in lstm.demand_features}
    parameter_dict["global"] = []

    # Create a dictionary of time_series -> parameters
    for n, p in lstm.named_parameters():
        # Handle the time-series specific parameters
        for f in lstm.demand_features:
            if n.startswith(f):
                parameter_dict[f].append(p)

        # Handle the global parameters
        if n.startswith("drnn.rnn_layer") or n.startswith(
                "linear") or n.startswith("tanh"):
            parameter_dict["global"].append(p)

    # Create a dictionary of time_series -> optimiser(parameters)
    optimizer_dict = {}
    for k, v in parameter_dict.items():
        if k == "global":
            optimizer_dict[k] = torch.optim.Adam(params=v, lr=global_init_lr)
        else:
            optimizer_dict[k] = torch.optim.Adam(params=v, lr=local_init_lr)

    num_epochs_since_change = 0
    rate_changed = False
    prev_loss = 0
    dynamic_learning_rate = local_init_lr
    pred_ensemble = []
    losses = {f: {l: [] for l in ["RNN", "LVP", "Total"]} for f in
              lstm.demand_features}

    # Loop through number of epochs amount of times
    for epoch in range(num_epochs):
        print("Epoch:", epoch)

        # Loop through each time series
        for f in lstm.demand_features:

            # If we're not using multiple time series, then skip all the time
            # series that aren't the total load actual column
            if not multi_ts and f != "total load actual":
                continue

            outs, labels, level_var_loss = lstm(
                data, f, window_size, output_size, weather, lvp,
                skip_lstm=skip_lstm
            )
            loss = loss_func(outs, labels, percentile)
            total_loss = loss + level_var_loss

            # Get the per-time-series optimiser
            local_optimizer = optimizer_dict[f]
            global_optimizer = optimizer_dict["global"]

            # Save losses
            losses[f]["RNN"].append(loss.item())
            losses[f]["LVP"].append(level_var_loss.item())
            losses[f]["Total"].append(total_loss.item())

            # User defined, fixed, variable learning rates depending on epoch
            if variable_lr:
                # Change local rates
                if epoch in list(local_variable_rates.keys()):
                    for param_group in local_optimizer.param_groups:
                        param_group['lr'] = local_variable_rates[epoch]

                    if (multi_ts and f == "generation fossil gas") or not multi_ts:
                            print("Changed Local Learning Rate to: " + str(
                                local_variable_rates[epoch]))

                # Change global rates
                if epoch in list(global_variable_rates.keys()):
                    for param_group in global_optimizer.param_groups:
                        param_group['lr'] = local_variable_rates[epoch]

                    if (multi_ts and f == "generation fossil gas") or not multi_ts:
                            print("Changed Local Learning Rate to: " + str(
                                local_variable_rates[epoch]))

            # Automatically changed learning rates depending loss ratio
            if auto_lr and num_epochs_since_change >= min_epochs_since_change:
                if prev_loss < auto_rt * loss:
                    dynamic_learning_rate /= sqrt(10)
                    print("Loss Ratio:", prev_loss / loss)
                    print("Changed Learning Rate to:", dynamic_learning_rate)

                    for param_group in local_optimizer.param_groups:
                        param_group['lr'] = dynamic_learning_rate

                    for param_group in global_optimizer.param_groups:
                        param_group['lr'] = dynamic_learning_rate

            local_optimizer.zero_grad()
            global_optimizer.zero_grad()
            total_loss.backward()
            local_optimizer.step()
            global_optimizer.step()

            print("Name: %s: LVP - %1.5f, Loss - %1.5f, Total Loss "
                  "- %1.5f" % (f, level_var_loss.item(), loss.item(),
                               total_loss.item()))

            prev_loss = loss

            # If we didn't change the learning rate, make note of this
            if not rate_changed:
                num_epochs_since_change += 1

        # If we are ensembling, generate the ensemble of forecasts
        if ensemble and (num_epochs - epoch <= 5):
            prediction, *_ = lstm.predict(forecast_input, window_size,
                                          output_size, weather,
                                          skip_lstm=skip_lstm)
            pred_ensemble.append(prediction.squeeze(0).detach().tolist())
        else:
            if epoch == num_epochs - 1:
                prediction, *_ = lstm.predict(forecast_input, window_size,
                                              output_size, weather,
                                              skip_lstm=skip_lstm)
                prediction = pd.Series(prediction.squeeze(0).detach().tolist())

        sys.stderr.flush()
        sys.stdout.flush()

    if ensemble:
        return pd.Series(np.mean(pred_ensemble, axis=0)), losses
    else:
        return prediction, losses


# if name == "total load actual":
# Check if parameters are updating:
# for n, p in lstm.named_parameters():
#     if n == "total load actual level smoothing":
#         print("Level Smoothing:", torch.sigmoid(p))
#     if n == "total load actual seasonality2 smoothing":
#         print("Seasonality Smoothing:", torch.sigmoid(p))
