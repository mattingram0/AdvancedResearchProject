import sys
from math import sqrt

import numpy as np
import pandas as pd
import torch

from hybrid.es_rnn_i import ES_RNN_I
from hybrid.es_rnn_s import ES_RNN_S
from ml.ml_helpers import pinball_loss
from stats.stats_helpers import deseasonalise


# Train the model and generate a single prediction. Training data must
# include test data to ensure it is of the correct length, it's not actually
# used
def es_rnn_s(data, forecast_length, seasonality,
             demand_features, weather_features, weather,
             ensemble, multi_ts):

    # Model hyper parameters
    window_size = 336
    num_epochs = 35
    hidden_size = 40
    num_layers = 4
    dilations = [1, 4, 24, 168]
    level_variability_penalty = 80
    percentile = 0.49
    loss_func = pinball_loss
    grad_clipping = 20
    auto_lr = False
    variable_lr = True
    auto_rate_threshold = 1.005
    min_epochs_before_change = 2
    residuals = tuple([[1, 3]])
    init_level_smoothing = -1
    init_seasonal_smoothing = -1.1
    input_size = 1 + len(weather_features) if weather else 1
    batch_first = True
    skip_lstm = False

    if multi_ts:
        local_init_lr = 0.01
        global_init_lr = 0.008
        local_rates = {10: 2e-3, 20: 1e-3, 30: 5e-4}
        global_rates = {10: 5e-3, 20: 1e-3, 30: 5e-4}
    else:
        local_init_lr = 0.005
        global_init_lr = 0.008
        local_rates = {10: 1e-3, 20: 5e-4, 30: 1e-4}
        global_rates = {10: 5e-3, 20: 1e-3, 30: 5e-4}

    # Split the data
    train_data = data[:-forecast_length]
    forecast_data = data[-(forecast_length + window_size):]
    batch_size = len(train_data) - window_size - forecast_length + 1

    # Estimate the seasonal indices and inital smoothing levels
    init_seas = {}
    init_l_smooth = {}
    init_s_smooth = {}
    for c in demand_features:
        deseas, indic = deseasonalise(train_data[c], 168, "multiplicative")
        init_seas[c] = indic
        init_l_smooth[c] = init_level_smoothing
        init_s_smooth[c] = init_seasonal_smoothing

    # Create the model
    lstm = ES_RNN_S(
        forecast_length, input_size, batch_size, hidden_size, num_layers,
        demand_features, weather_features, seasonality, dropout=0,
        cell_type='LSTM', batch_first=batch_first, dilations=dilations,
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

    # Train the model, and make a (possibly ensembled) prediction.
    prediction, _ = train_and_predict_s(
                lstm, train_data, window_size, forecast_length,
                level_variability_penalty, loss_func, num_epochs,
                local_init_lr, global_init_lr, percentile, auto_lr,
                variable_lr, auto_rate_threshold, min_epochs_before_change,
                forecast_data, local_rates, global_rates, ensemble, multi_ts,
                skip_lstm, weather
            )

    # Return prediction
    return prediction


def es_rnn_i(data, forecast_length, seasonality, demand_features,
             weather_features, weather, ensemble):

    # Model hyper parameters
    window_size = 336
    num_epochs = 35
    hidden_size = 40
    num_layers = 4
    input_size = len(demand_features) if not weather else len(
        demand_features) + len(weather_features)
    dilations = [1, 4, 24, 168]
    level_variability_penalty = 80
    percentile = 0.49
    loss_func = pinball_loss
    grad_clipping = 20
    auto_lr = False
    variable_lr = True
    auto_rate_threshold = 1.005
    min_epochs_before_change = 2
    residuals = tuple([[1, 3]])
    init_level_smoothing = -1
    init_seasonal_smoothing = -1.1
    batch_first = True
    local_init_lr = 0.01
    global_init_lr = 0.008
    local_rates = {10: 2e-3, 20: 1e-3, 30: 5e-4}
    global_rates = {10: 5e-3, 20: 1e-3, 30: 5e-4}

    # Split the data
    train_data = data[:-forecast_length]
    forecast_data = data[-(forecast_length + window_size):]
    batch_size = len(train_data) - window_size - forecast_length + 1

    # Estimate the seasonal indices and inital smoothing levels
    init_seas = {}
    init_l_smooth = {}
    init_s_smooth = {}
    for c in demand_features:
        deseas, indic = deseasonalise(train_data[c], 168, "multiplicative")
        init_seas[c] = indic
        init_l_smooth[c] = init_level_smoothing
        init_s_smooth[c] = init_seasonal_smoothing

    # Create the model
    lstm = ES_RNN_I(
        forecast_length, input_size, batch_size, hidden_size, num_layers,
        demand_features, weather_features, seasonality, dropout=0,
        cell_type='LSTM', batch_first=batch_first, dilations=dilations,
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

    # Train the model, and make a (possibly ensembled) prediction.
    prediction, _ = train_and_predict_i(
        lstm, train_data, window_size, forecast_length,
        level_variability_penalty, loss_func, num_epochs,
        local_init_lr, global_init_lr, percentile, auto_lr,
        variable_lr, auto_rate_threshold, min_epochs_before_change,
        forecast_data, local_rates, global_rates, ensemble, weather
    )

    # Save the model and exit - TODO REMOVE
    torch.save(lstm.state_dict(),
               "ddn/home/gkxx72/AdvancedResearchProject/run/model.pt")
    sys.exit()

    # Return prediction
    return prediction


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