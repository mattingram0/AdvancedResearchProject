import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import sys
import os
import json

from ml.helpers import pinball_loss, plot_test
from stats.helpers import split_data, deseasonalise, reseasonalise
from stats.errors import sMAPE, MASE, OWA
from stats.naive import naive_2
from math import sqrt

from hybrid.es_rnn import ES_RNN


# Give the model the training data, the forecast length and the seasonality,
# and this function trains the model and makes a single prediction
# Must pass in the training data plus the following forecast_length data.
# Note that the extra forecast_length of data (i.e the test data) doesn't
# get used, we just need the data to be the correct length (just the way
# I've coded it)
def es_rnn(data, forecast_length, seasonality, ensemble):
    # Hyperparameters - determined experimentally
    num_epochs = 27
    init_learning_rate = 0.01
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
    variable_rates = {7: 5e-3, 14: 1e-3, 22: 3e-4}
    auto_rate_threshold = 1.005
    min_epochs_before_change = 2
    residuals = tuple([[1, 3]])
    window_size = 336
    batch_first = True
    plot = False

    # Split the data
    train_data = data[:-forecast_length]
    forecast_data = data[-(forecast_length + window_size):]
    forecast_input = torch.tensor(forecast_data["total load actual"],
                                  dtype=torch.double)
    batch_size = len(train_data) - window_size - forecast_length + 1
    features = train_data.columns

    # Estimate the seasonal indices
    _, indices = deseasonalise(train_data["total load actual"], seasonality,
                               "multiplicative")

    # Create the model
    lstm = ES_RNN(
        forecast_length, input_size, batch_size, hidden_size, num_layers,
        batch_first=batch_first, dilations=dilations, features=features,
        seasonality_1=None, seasonality_2=seasonality, residuals=residuals,
        init_seasonality=indices, init_level_smoothing=-0.8,
        init_seas_smoothing=-0.2
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
        auto_rate_threshold, min_epochs_before_change, plot, forecast_input,
        variable_rates, ensemble)

    # Set model in evaluation (prediction) mode
    lstm.eval()

    # Return prediction
    return prediction


# General function used to do testing and tweaking
def run(df):
    # Optionally can supply a year as a command line argument
    year = -1 if len(sys.argv) == 2 else int(sys.argv[2])

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

    # For now, just use one of the training sets
    train_data = training_sets[0]
    window_size = 336
    output_size = 48
    batch_size = len(train_data) - window_size - output_size + 1
    write_results = False
    plot = True

    # Plot a result TODO - YOU ARE HERE - REMOVE BEFORE MORE TEST
    test_path = "/Users/matt/Projects/AdvancedResearchProject/test" \
                "/ind_winter_year_2.txt"
    with open(test_path) as f:
        plot_test(json.load(f), window_size, output_size, True)
    sys.exit(0)

    # Give the ssqueasonality parameters a helping hand
    _, indices = deseasonalise(train_data['total load actual'], 168,
                               "multiplicative")

    # Training parameters
    num_epochs = 1
    init_learning_rate = 0.01
    input_size = 1
    hidden_size = 40
    num_layers = 4
    dilations = [1, 4, 24, 168]
    level_variability_penalty = 100
    percentile = 0.49
    loss_func = pinball_loss
    grad_clipping = 20
    auto_lr = False  # Automatically adjust learning rates
    variable_lr = True  # Use list of epoch/rate pairs
    variable_rates = {7: 5e-3, 14: 1e-3, 22: 3e-4}
    auto_rate_threshold = 1.005  # If loss(x - 1) < 1.005 * loss(x) reduce rate
    min_epochs_before_change = 2
    residuals = tuple([[1, 3]])  # Residual connection from output of 2nd layer
    # to output of 4th layer

    data = valid_sets[1]
    test_model_week(data, output_size, input_size, batch_size, hidden_size,
                    num_layers, True, dilations, data.columns, 24, 168,
                    residuals, indices, -0.8, window_size,
                    level_variability_penalty, loss_func, num_epochs,
                    init_learning_rate, percentile, auto_lr, variable_lr,
                    auto_rate_threshold, min_epochs_before_change,
                    variable_rates, grad_clipping, write_results, plot, year)


# If output_size != 48 then this is broken. Pass in valid data or test
# data!! (i.e all up to the end of the valid section/test section).
def test_model_week(data, output_size, input_size, batch_size, hidden_size,
                    num_layers, batch_first, dilations, features, seasonality1,
                    seasonality2, residuals, init_seasonality,
                    init_level_smoothing, window_size,
                    level_variability_penalty, loss_func, num_epochs,
                    init_learning_rate, percentile, auto_lr, variable_lr,
                    auto_rate_threshold, min_epochs_before_change,
                    variable_rates, grad_clipping, write_results, plot, year):

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

    for i in range(2, 1, -1):
        # Figure out start and end points of the data
        end_train = -(i * 24)
        start_test = -(i * 24 + window_size)
        end_test = -(i * 24 - output_size) if i != 2 else None

        train_data = data[:end_train]
        test_data = torch.tensor(
            data["total load actual"][start_test:end_test],
            dtype=torch.double
        )
        mase_data = data["total load actual"][:end_test]

        # Deseasonalise data for naive forecast and to get initial indices
        train_d, indices = deseasonalise(train_data["total load actual"], 168,
                                         "multiplicative")

        # Create a fresh model
        lstm = ES_RNN(
            output_size, input_size, batch_size, hidden_size, num_layers,
            batch_first=batch_first, dilations=dilations, features=features,
            seasonality_1=seasonality1, seasonality_2=seasonality2,
            residuals=residuals, init_seasonality=indices,
            init_level_smoothing=-0.8, init_seas_smoothing=-0.2
        ).double()

        # Register gradient clipping function
        for p in lstm.parameters():
            p.register_hook(
                lambda grad: torch.clamp(grad, -grad_clipping, grad_clipping))

        # Set model in training mode
        lstm.train()

        print("----- TEST", str(9 - i), "-----")

        # Train the model.
        train_and_predict(lstm, train_data, window_size, output_size,
                    level_variability_penalty, loss_func, num_epochs,
                    init_learning_rate, percentile, auto_lr, variable_lr,
                    auto_rate_threshold, min_epochs_before_change, test_data,
                    plot, variable_rates, ensemble=False)

        # Set model into evaluation mode
        lstm.eval()

        # Make ES_RNN Prediction
        prediction, actual, out_levels, out_seas, all_levels, all_seasons, \
        rnn_out = lstm.predict(
            test_data, window_size, output_size
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
            naive_2(train_d, output_size),
            indices,
            "multiplicative"
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

        # Note results (NCC)
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
            lstm.seasonality2_smoothing_coeffs["total load actual"].data
        )

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

    # Make note of final results
    results["overall"] = {
        "avg_owa": avg_owa,
        "num_improved": num_improved,
        "avg_improvement": avg_improve,
        "avg_decline": avg_decline
    }

    # Write results (NCC)
    if write_results:
        # res_path = os.path.join("/Users/matt/", str(sys.argv[1]))
        filename = str(sys.argv[1]) + ".txt" if year == -1 else str(
            sys.argv[1]) + "_year_" + str(year) + ".txt"
        res_path = os.path.join(
            "/ddn/home/gkxx72/AdvancedResearchProject/run/", filename)
        # res_path = "/Users/matt/test.txt"
        with open(res_path, "w") as res:
            json.dump(results, res)

    if plot:
        plot_test(results, window_size, output_size, print_results=True)


# This function trains the model, and generates (a) prediction(s). If we
# pass in ensemble = False, then a single prediction after training is
# generated. If ensemble = True, the predictions from the final 5 epochs are
# averaged and returned. If we wish to make use of the extra information
# that the ES_RNN.predict() function returns, we can call it directly.
def train_and_predict(lstm, data, window_size, output_size, lvp, loss_func,
                      num_epochs, init_learning_rate, percentile, auto_lr,
                      variable_lr, auto_rt, min_epochs_since_change, plot,
                      forecast_input, variable_rates=None, ensemble=False):
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
        if n.startswith("drnn.rnn_layer") or n.startswith("linear") or n.startswith("tanh"):
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

    # Loop through number of epochs amount of times
    for epoch in range(num_epochs):

        # Loop through each time series
        for name, inputs in input_list.items():
            outs, labels, level_var_loss = lstm(
                inputs, name, window_size, output_size, lvp
            )
            loss = loss_func(outs, labels, percentile)
            total_loss = loss + level_var_loss

            # Get the per-time-series optimiser
            optimizer = optimizer_dict[name]

            # User defined, fixed, variable learning rates depending on epoch
            if variable_lr and epoch in list(variable_rates.keys()):
                for param_group in optimizer.param_groups:
                    param_group['lr'] = variable_rates[epoch]

                if name == "total load actual":
                    print("Changed Learning Rate to: " + str(variable_rates
                                                             [epoch]))

            # Automatically changed learning rates depending loss ratio
            if auto_lr and num_epochs_since_change >= min_epochs_since_change:
                if prev_loss < auto_rt * loss:
                    dynamic_learning_rate /= sqrt(10)
                    print("Loss Ratio:", prev_loss/loss)
                    print("Changed Learning Rate to:", dynamic_learning_rate)

                    for param_group in optimizer.param_groups:
                        param_group['lr'] = dynamic_learning_rate

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if name == "total load actual":
                print(
                    "Name: %s: Epoch %d: LVP - %1.5f, Loss - %1.5f, "
                    "Total Loss - %1.5f" %
                    (name, epoch, level_var_loss.item(), loss.item(),
                     total_loss.item())
                )

                # Check if parameters are updating:
                for n, p in lstm.named_parameters():
                    if n == "total load actual level smoothing":
                        print("Level Smoothing:", torch.sigmoid(p))
                    if n == "total load actual seasonality2 smoothing":
                        print("Seasonality Smoothing:", torch.sigmoid(p))

            prev_loss = loss

            # If we didn't change the learning rate, make note of this
            if not rate_changed:
                num_epochs_since_change += 1

        # If we are ensembling, generate the ensemble of forecasts
        if ensemble and (num_epochs - epoch <= 5):
            prediction, *_ = lstm.predict(forecast_input, window_size,
                                          output_size)
            pred_ensemble.append(prediction.squeeze(0).detach().tolist())
            pass
        else:
            if epoch == num_epochs - 1:
                prediction, *_ = lstm.predict(forecast_input, window_size,
                                              output_size)
                prediction, = pd.Series(
                    prediction.squeeze(0).detach().tolist()
                )

    if ensemble:
        return pd.Series(np.mean(pred_ensemble, axis=0))
    else:
        return prediction


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

    print("Final Level:", level[0])
    print("Final Seasonality (Exponentiated):", out_seas2)
    print("Final Seasonality:", torch.log(out_seas2))

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
