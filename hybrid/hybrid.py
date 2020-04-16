import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import sys
import os
import json

from ml.helpers import pinball_loss, plot_test, plot_es_comp, plot_gen_forecast
from stats.helpers import split_data, deseasonalise, reseasonalise
from stats.errors import sMAPE, MASE, OWA
from stats.naive import naive_2
from math import sqrt

from hybrid.es_rnn import ES_RNN
from hybrid.es_rnn_ex import ES_RNN_EX


# Give the model the training data, the forecast length and the seasonality,
# and this function trains the model and makes a single prediction
# Must pass in the training data plus the following forecast_length data.
# Note that the extra forecast_length of data (i.e the test data) doesn't
# get used, we just need the data to be the correct length (just the way
# I've coded it)
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
    lstm = ES_RNN(
        forecast_length, input_size, batch_size, hidden_size, num_layers,
        batch_first=batch_first, dilations=dilations, features=features,
        seasonality_1=None, seasonality_2=seasonality, residuals=residuals,
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
    prediction = train_and_predict(
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
def run(df, multi_ts, ex):
    # Optionally can supply a year as a command line argument
    year = -1 if len(sys.argv) < 3 else int(sys.argv[2])
    season = -1 if len(sys.argv) < 4 else int(sys.argv[3])

    # all_data = {Season: [Year1, ...]}
    all_data = split_data(df)

    # Each year = [<- 12 week Train -> | <- 1 week Val. -> | <- 1 week Test ->]
    training_sets = [
        all_data["Winter"][year if year >= 0 else 1][:-(15 * 24)],
        all_data["Spring"][year if year >= 0 else 0][:-(15 * 24)],
        all_data["Summer"][year if year >= 0 else 3][:-(15 * 24)],
        all_data["Autumn"][year if year >= 0 else 2][:-(15 * 24)]
    ]

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

    # Default to summer if no month provided in the input arguments
    data = valid_sets[season if season >= 0 else 2]

    # Plot a result #TODO HOW TO PLOT RUN_TEST RESULT FROM HAM
    # test_path = "/Users/matt/Projects/AdvancedResearchProject/test" \
    #             "/ind_winter_year_2.txt"
    # with open(test_path) as f:
    #     plot_test(json.load(f), window_size, output_size, True)
    # sys.exit(0)

    # For now, just use one of the training sets
    window_size = 336
    output_size = 48
    write_results = True
    plot = True
    ensemble = False
    skip_lstm = False
    init_params = True

    # Training parameters
    # num_epochs = 50
    #     # init_learning_rate = 0.1

    num_epochs = 35
    local_init_lr = 0.01
    global_init_lr = 0.005
    input_size = 4
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

    # TODO - FOR THE MULTI TS YOU NEED TO FIX THE LOCAL AND GLOBAL RATES STUFF

    auto_rate_threshold = 1.005  # If loss(x - 1) < 1.005 * loss(x) reduce rate
    min_epochs_before_change = 2
    residuals = tuple([[1, 3]])  # Residual connection from output of 2nd layer
    # to output of 4th layer

    test_model_week(data, output_size, input_size, hidden_size,
                    num_layers, True, dilations, data.columns, 24, 168,
                    residuals, window_size, level_variability_penalty,
                    loss_func, num_epochs, local_init_lr, global_init_lr,
                    percentile, auto_lr, variable_lr, auto_rate_threshold,
                    min_epochs_before_change, local_rates, global_rates,
                    grad_clipping, write_results, plot, year, season,
                    ensemble, multi_ts, skip_lstm, ex, init_params)


# If output_size != 48 then this is broken. Pass in valid data or test
# data!! (i.e all up to the end of the valid section/test section).
def test_model_week(data, output_size, input_size, hidden_size,
                    num_layers, batch_first, dilations, features, seasonality1,
                    seasonality2, residuals, window_size,
                    level_variability_penalty, loss_func, num_epochs,
                    local_init_lr, global_init_lr, percentile, auto_lr,
                    variable_lr, auto_rate_threshold, min_epochs_before_change,
                    local_rates, global_rates, grad_clipping, write_results,
                    plot, year, season, ensemble, multi_ts, skip_lstm, ex,
                    init_params):
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

    for i in range(8, 1, -1):
        # Figure out start and end points of the data
        end_train = -(i * 24)
        start_test = -(i * 24 + window_size)
        end_test = -(i * 24 - output_size) if i != 2 else None

        train_data = data[:end_train]
        mase_data = data["total load actual"][:end_test]

        if init_params:
            # Determine initial parameter values. Can probably improved the
            # inital smoothing coefficients to be time series specific
            init_seas = {}
            init_l_smooth = {}
            init_s_smooth = {}
            for c in data.columns:
                deseas, indic = deseasonalise(train_data["total load actual"],
                                              168, "multiplicative")
                init_seas[c] = indic
                init_l_smooth[c] = -2
                init_s_smooth[c] = -1

                if c == "total load actual":
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

        # Batch size properly calculated here, as training data size changes
        batch_size = len(train_data["total load actual"]) - window_size - \
                     output_size + 1

        # Create a fresh model
        if ex:
            lstm = ES_RNN_EX(
                output_size, input_size, batch_size, hidden_size,
                num_layers, features, seasonality2, batch_first=batch_first,
                dilations=dilations, residuals=residuals,
                init_seasonality=init_seas, init_level_smoothing=init_l_smooth,
                init_seas_smoothing=init_s_smooth
            ).double()
        else:
            lstm = ES_RNN(
                output_size, input_size, batch_size, hidden_size, num_layers,
                batch_first=batch_first, dilations=dilations,
                features=features,
                seasonality_1=seasonality1, seasonality_2=seasonality2,
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
        if ex:
            test_data = data[start_test:end_test]
            _, losses = train_and_predict_ex(lstm, data, window_size,
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
                                             ensemble)
        else:
            test_data = torch.tensor(
                data["total load actual"][start_test:end_test],
                dtype=torch.double
            )
            _, losses = train_and_predict(lstm, train_data, window_size,
                                          output_size,
                                          level_variability_penalty,
                                          loss_func, num_epochs,
                                          local_init_lr, percentile,
                                          auto_lr, variable_lr,
                                          auto_rate_threshold,
                                          min_epochs_before_change,
                                          test_data, local_rates, ensemble,
                                          multi_ts,
                                          skip_lstm)

        # Set model into evaluation mode
        lstm.eval()

        # Make ES_RNN Prediction
        prediction, actual, out_levels, out_seas, all_levels, all_seasons, \
        rnn_out = lstm.predict(test_data, window_size, output_size)

        # Convert to correct form for results saving
        if ex:
            test_data = torch.tensor(
                data["total load actual"][start_test:end_test],
                dtype=torch.double
            )

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

        if ex:
            results[9 - i]["seasonality_smoothing"] = float(
                lstm.seasonality_smoothing_coeffs["total load actual"].data
            )
        else:
            results[9 - i]["seasonality_smoothing"] = float(
                lstm.seasonality2_smoothing_coeffs["total load actual"].data
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

        # res_path = os.path.join(
        #     "/Users/matt/Projects/AdvancedResearchProject/test/",
        #     filename
        # )

        res_path = os.path.join(
            "/ddn/home/gkxx72/AdvancedResearchProject/run/test_res/",
            filename
        )

        with open(res_path, "w") as res:
            json.dump(results, res)

    if plot:
        plot_test(results, window_size, output_size, print_results=True)


def train_and_predict_ex(lstm, data, window_size, output_size, lvp,
                         loss_func, num_epochs, local_init_lr, global_init_lr,
                         percentile, auto_lr, variable_lr, auto_rt,
                         min_epochs_since_change, forecast_input,
                         local_rates, global_rates, ensemble=False):
    parameter_dict = {c: [] for c in data.columns}
    parameter_dict["global"] = []

    # Create a dictionary of time_series -> parameters
    for n, p in lstm.named_parameters():
        # Handle the time-series specific parameters
        for c in data.columns:
            if n.startswith(c):
                parameter_dict[c].append(p)

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
    losses = {f: {l: [] for l in ["RNN", "LVP", "Total"]} for f in
              data.columns}
    pred_ensemble = []

    # Loop through number of epochs amount of times
    for epoch in range(num_epochs):
        print("Epoch:", epoch)
        outs, labels, level_var_losses = lstm(
            data, window_size, output_size, lvp
        )
        loss = loss_func(outs, labels, percentile)
        global_optimizer = optimizer_dict["global"]

        for f in lstm.features:
            # Calculate the total loss
            total_loss = loss + level_var_losses[f]

            # Update the local parameters - updated only once per epoch
            local_optimizer = optimizer_dict[f]
            local_optimizer.zero_grad()
            total_loss.backward(retain_graph=True)
            local_optimizer.step()

            # Update the global parameters - updated for every feature
            global_optimizer.zero_grad()
            total_loss.backward(retain_graph=True)
            global_optimizer.step()

            # Save losses
            losses[f]["RNN"].append(loss.item())
            losses[f]["LVP"].append(level_var_losses[f].item())
            losses[f]["Total"].append(total_loss.item())

            # Print loss information
            print("%s: LVP - %1.5f, Loss - %1.5f Total Loss "
                  "- %1.5f" % (f, level_var_losses[f].item(), loss.item(),
                               total_loss.item()))
            # Update local optimizer learning rates if using variable lrs
            if variable_lr and epoch in list(local_rates.keys()):
                for param_group in local_optimizer.param_groups:
                    param_group['lr'] = local_rates[epoch]

                    print("Changed Learning Rate to:",
                          str(local_rates[epoch]))

            # Update local optimizer lr if using automatic learning rates
            if auto_lr and num_epochs_since_change >= min_epochs_since_change:
                if prev_loss < auto_rt * loss:
                    dynamic_learning_rate /= sqrt(10)
                    print("Loss Ratio:", prev_loss / loss)
                    print("Changed Learning Rate to:", dynamic_learning_rate)

                    for param_group in local_optimizer.param_groups:
                        param_group['lr'] = dynamic_learning_rate

            prev_loss = loss

        # Update global optimizer if using variable learning rates
        if variable_lr and epoch in list(global_rates.keys()):
            for param_group in global_optimizer.param_groups:
                param_group['lr'] = global_rates[epoch]

        # Update local optimizer if using automatic learning rates
        if auto_lr and num_epochs_since_change >= min_epochs_since_change:
            if prev_loss < auto_rt * loss:
                dynamic_learning_rate /= sqrt(10)
                print("Loss Ratio:", prev_loss / loss)
                print("Changed Learning Rate to:",
                      dynamic_learning_rate)

                for param_group in global_optimizer.param_groups:
                    param_group['lr'] = dynamic_learning_rate

        # If we didn't change the learning rate, make note of this
        if not rate_changed:
            num_epochs_since_change += 1

        # If we are ensembling, generate the ensemble of forecasts
        if ensemble and (num_epochs - epoch <= 5):
            prediction, *_ = lstm.predict(forecast_input, window_size,
                                          output_size)
            pred_ensemble.append(prediction.squeeze(0).detach().tolist())
        else:
            if epoch == num_epochs - 1:
                prediction, *_ = lstm.predict(forecast_input, window_size,
                                              output_size)
                prediction = pd.Series(prediction.squeeze(0).detach().tolist())

        sys.stderr.flush()
        sys.stdout.flush()

    if ensemble:
        return pd.Series(np.mean(pred_ensemble, axis=0)), losses
    else:
        return prediction, losses


# This function trains the model, and generates (a) prediction(s). If we
# pass in ensemble = False, then a single prediction after training is
# generated. If ensemble = True, the predictions from the final 5 epochs are
# averaged and returned. If we wish to make use of the extra information
# that the ES_RNN.predict() function returns, we can call it directly.
def train_and_predict(lstm, data, window_size, output_size, lvp, loss_func,
                      num_epochs, init_learning_rate, percentile, auto_lr,
                      variable_lr, auto_rt, min_epochs_since_change,
                      forecast_input, variable_rates=None, ensemble=False,
                      multi_ts=False, skip_lstm=False):
    # to make this stochastic gradient descent??? Output the final level and
    # seasonality of the chunk?? Then we could feed this in as the initial
    # seasonality and level. Would we then also need to make sure that the
    # hidden state of the RNN persists too??

    # Split input = [batch_size, seq_len, num_featurs] into a num_features
    # length dictionary of name:[batch_size, seq_len, 1] tensors
    input_list = {c: torch.tensor(data[c], dtype=torch.double) for c in
                  data.columns}
    parameter_dict = {c: [] for c in data.columns}

    # Create a dictionary of time_series -> parameters
    for n, p in lstm.named_parameters():
        # Handle the time-series specific parameters
        for c in data.columns:
            if n.startswith(c):
                parameter_dict[c].append(p)

        # Handle the global parameters
        if n.startswith("drnn.rnn_layer") or n.startswith(
                "linear") or n.startswith("tanh"):
            for c in data.columns:
                parameter_dict[c].append(p)

    # Create a dictionary of time_series -> optimiser(parameters)
    optimizer_dict = {}
    for k, v in parameter_dict.items():
        optimizer_dict[k] = torch.optim.Adam(params=v, lr=init_learning_rate)

    num_epochs_since_change = 0
    rate_changed = False
    prev_loss = 0
    dynamic_learning_rate = init_learning_rate
    pred_ensemble = []
    losses = {f: {l: [] for l in ["RNN", "LVP", "Total"]} for f in
              data.columns}

    # Print params
    # for n, p in lstm.named_parameters():
    #     print(n, p)
    #     # if n == "total load actual level smoothing":
    #     #     print("Level Smoothing:", torch.sigmoid(p))
    #     # if n == "total load actual seasonality2 smoothing":
    #     #     print("Seasonality Smoothing:", torch.sigmoid(p))

    # Loop through number of epochs amount of times
    for epoch in range(num_epochs):

        # Loop through each time series
        for name, inputs in input_list.items():

            # If we're not using multiple time series, then skip all the time
            # series that aren't the total load actual column
            if not multi_ts and name != "total load actual":
                continue

            outs, labels, level_var_loss = lstm(
                inputs, name, window_size, output_size, lvp,
                skip_lstm=skip_lstm
            )
            loss = loss_func(outs, labels, percentile)
            total_loss = loss + level_var_loss

            # Get the per-time-series optimiser
            optimizer = optimizer_dict[name]

            # Save losses
            losses[name]["RNN"].append(loss.item())
            losses[name]["LVP"].append(level_var_loss.item())
            losses[name]["Total"].append(total_loss.item())

            # User defined, fixed, variable learning rates depending on epoch
            if variable_lr and epoch in list(variable_rates.keys()):
                for param_group in optimizer.param_groups:
                    param_group['lr'] = variable_rates[epoch]

                # Generation fossil gas is the first time series loaded,
                # don't need to repeat the message for all time series
                if multi_ts and name == "generation fossil gas":
                    print("Changed Learning Rate to: " + str(variable_rates
                                                             [epoch]))
                else:
                    print("Changed Learning Rate to: " + str(variable_rates
                                                             [epoch]))

            # Automatically changed learning rates depending loss ratio
            if auto_lr and num_epochs_since_change >= min_epochs_since_change:
                if prev_loss < auto_rt * loss:
                    dynamic_learning_rate /= sqrt(10)
                    print("Loss Ratio:", prev_loss / loss)
                    print("Changed Learning Rate to:", dynamic_learning_rate)

                    for param_group in optimizer.param_groups:
                        param_group['lr'] = dynamic_learning_rate

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            print("Name: %s: Epoch %d: LVP - %1.5f, Loss - %1.5f Total Loss "
                  "- %1.5f" % (name, epoch, level_var_loss.item(), loss.item(),
                               total_loss.item()))

            # if name == "total load actual":
            # Check if parameters are updating:
            # for n, p in lstm.named_parameters():
            #     if n == "total load actual level smoothing":
            #         print("Level Smoothing:", torch.sigmoid(p))
            #     if n == "total load actual seasonality2 smoothing":
            #         print("Seasonality Smoothing:", torch.sigmoid(p))

            prev_loss = loss

            # If we didn't change the learning rate, make note of this
            if not rate_changed:
                num_epochs_since_change += 1

        # If we are ensembling, generate the ensemble of forecasts
        if ensemble and (num_epochs - epoch <= 5):
            prediction, *_ = lstm.predict(forecast_input, window_size,
                                          output_size, skip_lstm=skip_lstm)
            pred_ensemble.append(prediction.squeeze(0).detach().tolist())
        else:
            if epoch == num_epochs - 1:
                prediction, *_ = lstm.predict(forecast_input, window_size,
                                              output_size, skip_lstm=skip_lstm)
                prediction = pd.Series(prediction.squeeze(0).detach().tolist())

        sys.stderr.flush()
        sys.stdout.flush()

    if ensemble:
        return pd.Series(np.mean(pred_ensemble, axis=0)), losses
    else:
        return prediction, losses


# This function takes all the data up to the end of the section (i.e up to
# the end of the validation section or test section), then generates
# forecasts for the last week of data. For an example data set, the first 12
# weeks are for training, the next is for validation and the final one is
# for testing. So we pass 13 weeks in to test on the validation, and all 14
# to test on the test data. This is because, to calculate the MASE,
# the entire data set is required.
# Only going to work with an output size of 48 I think, sadly :((
# Returns a list of 7 OWA
def test_model(lstm, data, window_size, output_size):
    # Data = [<- Train -1 Week -> | <- 1 Week -> <- Test = 1 week -> ]
    # Extra week required because ES_RNN requires week of input to generate
    # an output

    # Number actually predicted is this value - 1
    num_pred = 2
    test_data = torch.tensor(
        data["total load actual"][-(num_pred * 24 + window_size):],
        dtype=torch.double
    )

    # Calculate ES_RNN predictions TODO - CHANGED WHAT PREDICT RETURNS!!
    predictions, actuals, out_levels, out_seas, all_levels, all_seasons, \
    rnn_out = lstm.predict(
        test_data, window_size, output_size
    )

    predictions = [pd.Series(p) for p in predictions.detach().tolist()]
    actuals = [pd.Series(a) for a in actuals.detach().tolist()]
    out_levels

    # Calculate Naive 2 predictions
    naive_predictions = []
    naive_actuals = []
    for i in range(num_pred, 1, -1):
        # Data = [<- Train -> | <- Test = 1 week ->]
        # 1)   = [<- Train -> | <- Actual = 2 days -> <- Unused: 5 days ->]
        # 2)   = [<- Train + 1 -> | <- Actual = 2 days -> <- Unused: 4 days ->]
        # ...
        # 7)   = [<-        Train + 6 days          -> | <- Actual = 2 days ->]
        train = data["total load actual"][:-(i * 24)]
        actual_end = -(i * 24 - output_size) if i > 2 else None
        actual = data["total load actual"][-(i * 24):actual_end]

        train_d, indices = deseasonalise(train, 168, "multiplicative")
        naive_fit_forecast = reseasonalise(
            naive_2(train_d, output_size),
            indices,
            "multiplicative"
        )
        naive_forecast = naive_fit_forecast[-output_size:]

        naive_predictions.append(naive_forecast.reset_index(drop=True))
        naive_actuals.append(actual)

    # Plot predictions and actual
    for i in range(num_pred - 1):
        fig = plt.figure(figsize=(20, 15), dpi=250)
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(predictions[i], label="ES_RNN Predicted")
        ax.plot(naive_predictions[i], label="Naive2 Predictions")
        ax.plot(actuals[i], label="Actual")
        ax.legend(loc="best")
        plt.show()

    # Create the required sections of the data for the MASE. To calculate
    # the MASE, [<- Train -> | <- Forecast ->] actual values are needed
    mase_actuals = []
    for i in range(num_pred - 2, 0, -1):
        mase_actuals.append(
            pd.Series(data["total load actual"][:-(i * 24)])
        )
    mase_actuals.append(pd.Series(data["total load actual"]))

    # Calculate the MASE for ES_RNN
    mase = []
    for a, p in zip(mase_actuals, predictions):
        m = MASE(p, a, 168, output_size)
        print("ES-RNN MASE:", m)
        mase.append(MASE(p, a, 168, output_size))

    # Calculate the sMAPE for ES_RNN
    smape = []
    for a, p in zip(actuals, predictions):
        m = sMAPE(p, a)
        print("ES-RNN sMAPE:", m)
        smape.append(m)

    # Calculate the MASE for Naive 2
    naive_mase = []
    for a, p in zip(mase_actuals, naive_predictions):
        m = MASE(p, a, 168, output_size)
        print("Naive MASE:", m)
        naive_mase.append(MASE(p, a, 168, output_size))

    # Calculate the sMAPE for Naive 2
    naive_smape = []
    for a, p in zip(naive_actuals, naive_predictions):
        m = sMAPE(p, a)
        print("Naive sMAPE:", m)
        smape.append(m)
        naive_smape.append(m)

    # Calculate the OWA
    owa = []
    for m, s, nm, ns in zip(mase, smape, naive_mase, naive_smape):
        owa.append(OWA(ns, nm, s, m))

    for n, p in lstm.named_parameters():
        if n == "total load actual level smoothing":
            print("Level Smoothing:", torch.sigmoid(p))
        if n == "total load actual seasonality2 smoothing":
            print("Seasonality Smoothing:", torch.sigmoid(p))

    # print("Final Level:", level[0])
    # print("Final Seasonality (Exponentiated):", out_seas2)
    # print("Final Seasonality:", torch.log(out_seas2))

    return owa, indices

    # test_prediction = lstm.predict(
    #     test_data["total load actual"], window_size, output_size
    # )

    # train_actual = np.array(data["total load actual"][
    #                         1008:train_hours]).reshape(-1)
    #
    # prediction = np.array(valid_pred.detach())
    # actual = np.concatenate((np.array(train_actual), np.array(valid_actual)))
    #
    # x1 = np.array([range(len(actual))]).reshape(-1)
    # x2 = np.array([range(len(actual) - len(prediction),
    #                      len(actual))]).reshape(-1)
    #
    # plt.figure(figsize=(12.8, 9.6), dpi=250)
    # plt.plot(x1, actual, label="Actual Data")
    # plt.plot(x2, prediction, label="Prediction")
    #
    # # plt.axvline(x=train_hours - window_size, c='orange', label="Training Data")
    # # plt.axvline(x=train_hours - window_size + len(valid_actual), c='purple',
    # #             label="Validation Data")
    # plt.gca().set_xlabel("Time")
    # plt.gca().set_ylabel("Total Energy Demand")
    # plt.gca().legend(loc="best")
    # plt.gca().set_title('Dilations: ' + str(dilations))
    # plt.show()
