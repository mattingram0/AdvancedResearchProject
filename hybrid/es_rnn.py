import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import sys

from torch.distributions import Normal, Uniform
from ml import drnn, non_lin
from ml.helpers import pinball_loss
from stats.helpers import split_data, deseasonalise, reseasonalise
from stats.errors import sMAPE, MASE, OWA
from stats.naive import naive_2
from math import sqrt


def es_rnn(df):

    # all_data = {Season: [Year1, ...]}
    all_data = split_data(df)

    # Each year = [<- 12 week Train -> | <- 1 week Val. -> | <- 1 week Test ->]
    training_sets = [
        all_data["Winter"][1][:-(15 * 24)],
        all_data["Spring"][0][:-(15 * 24)],
        all_data["Summer"][3][:-(15 * 24)],
        all_data["Autumn"][2][:-(15 * 24)]
    ]

    # For now, just use one of the training sets
    train_data = training_sets[0]
    window_size = 336
    output_size = 48
    batch_size = len(train_data) - window_size - output_size + 1

    # Give the seasonality parameters a helping hand
    _, indices = deseasonalise(train_data['total load actual'], 168,
                               "multiplicative")

    # Training parameters
    num_epochs = 27
    init_learning_rate = 0.01
    input_size = 1
    hidden_size = 40
    num_layers = 4
    dilations = [1, 4, 24, 168]
    level_variability_penalty = 50
    percentile = 0.49
    loss_func = pinball_loss
    grad_clipping = 20
    auto_lr = False  # Automatically adjust learning rates
    variable_lr = True  # Use list of epoch/rate pairs
    variable_rates = {7: 5e-3, 18: 1e-3, 22: 3e-4}
    auto_rate_threshold = 1.005  # If loss(x - 1) < 1.005 * loss(x) reduce rate
    min_epochs_before_change = 2
    residuals = tuple([[1, 3]])  # Residual connection from output of 2nd layer
    # to output of 4th layer

    # Create model
    lstm = ES_RNN(
        output_size, input_size, batch_size, hidden_size, num_layers,
        batch_first=True, dilations=dilations, features=train_data.columns,
        seasonality_1=24, seasonality_2=168, residuals=residuals,
        init_seasonality=indices, init_level_smoothing=-0.8
    ).double()

    # Register gradient clipping function
    for p in lstm.parameters():
        p.register_hook(
            lambda grad: torch.clamp(grad, -grad_clipping, grad_clipping))

    # for n, p in lstm.named_parameters():
    #     print(n, p.shape)
    #     print(len(list(lstm.named_parameters())))

    #

    # Train model
    lstm.train()
    train_model(lstm, train_data, window_size, output_size,
                level_variability_penalty, loss_func, num_epochs,
                init_learning_rate, percentile, auto_lr, variable_lr,
                auto_rate_threshold, min_epochs_before_change, variable_rates)


    valid_sets = [
        all_data["Winter"][1][:-(8 * 24)],
        all_data["Spring"][0][:-(8 * 24)],
        all_data["Summer"][3][:-(8 * 24)],
        all_data["Autumn"][2][:-(8 * 24)]
    ]

    test_sets = [
        all_data["Winter"][1],
        all_data["Spring"][0],
        all_data["Summer"][3],
        all_data["Autumn"][2]
    ]

    # Make predictions
    lstm.eval()

    valid_data = valid_sets[0]
    validation_results, indices = test_model(lstm, valid_data, window_size,
                                     output_size)
    print("Validation Average OWA:", np.mean(validation_results))

    sys.exit(0)

    # Print the seasonality indices for ES_RNN and Classic Decomposition
    fig = plt.figure(figsize=(20, 15), dpi=250)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(indices, label="Seasonal Decomposition Seasonality")
    ax.plot(torch.tensor(lstm.w_seasons).tolist(), label="ES_RNN Seasonality")
    ax.legend(loc="best")
    plt.show()


    # test_data = test_sets[0]
    # test_results = test_model(lstm, valid_data, window_size, output_size)
    # print("Test Average OWA:", np.mean(test_results))


def train_model(lstm, data, window_size, output_size, lvp, loss_func, num_epochs,
                init_learning_rate, percentile, auto_lr, variable_lr,
                auto_rt, min_epochs_since_change, variable_rates=None):
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

    # Plots for debugging:
    fig = plt.figure(figsize=(20, 15), dpi=250)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(data["total load actual"].to_list())
    ax.set_xticks([])

    num_epochs_since_change = 0
    rate_changed = False
    prev_loss = 0
    dynamic_learning_rate = init_learning_rate
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

            if (epoch + 1) % 10 == 0:
                ax.plot(torch.tensor(lstm.levels).tolist(), label=str(epoch
                                                                      + 1))

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

            prev_loss = loss

            if not rate_changed:
                num_epochs_since_change += 1
    # ax.legend(loc="best")
    plt.show()

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

    # Calculate ES_RNN predictions
    predictions, actuals, level, out_seas2 = lstm.predict(
        test_data, window_size, output_size
    )

    predictions = [pd.Series(p) for p in predictions.detach().tolist()]
    actuals = [pd.Series(a) for a in actuals.detach().tolist()]

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


# Stateful, mini-batch trained DRNN. One feature.
class ES_RNN(nn.Module):
    def __init__(self, output_size, input_size, batch_size, hidden_size,
                 num_layers, features, seasonality_1, seasonality_2, dropout=0,
                 cell_type='LSTM', batch_first=False, dilations=None,
                 residuals=tuple([[]]), init_seasonality=None,
                 init_level_smoothing=None, init_seas_smoothing=None):

        # TODO - the initial smoothing coefficients will need to be a
        #  dictionary, and given for all the features, when I add the other
        #  features

        super().__init__()

        # RNN Hyperparameters
        self.output_size = output_size
        self.input_size = input_size
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cell_type = cell_type

        # The input data may not necessarily start on a Monday, so we need
        # to make a note of which day comes first in the seasonality params
        self.first_day = 0  # By default, assum data begins on a Monday

        # Seasonality values
        # self.seasonality_1 = seasonality_1
        self.seasonality_2 = seasonality_2

        # ES parameters
        self.level_smoothing_coeffs = {}
        # self.seasonality1_smoothing_coeffs = {}
        self.seasonality2_smoothing_coeffs = {}
        self.hourly_seasonality_params = {}
        self.weekly_seasonality_params = {}

        # ES level and seasonality values (for total load actual only)
        self.levels = []
        # self.h_seasons = []
        self.w_seasons = []

        # Add all parameters to the network
        u1 = Uniform(-1, 1)  # Smoothing coefficients
        u2 = Uniform(0.65, 1.35)  # Seasonality parameters
        for f in features:
            # Create the parameters
            if init_level_smoothing:
                self.level_smoothing_coeffs[f] = torch.nn.Parameter(
                    torch.tensor(init_level_smoothing, dtype=torch.double),
                    requires_grad=True)
            else:
                self.level_smoothing_coeffs[f] = torch.nn.Parameter(
                    u1.sample(), requires_grad=True)

            if init_seas_smoothing:
                self.seasonality2_smoothing_coeffs[f] = torch.nn.Parameter(
                    torch.tensor(init_seas_smoothing, dtype=torch.double),
                    requires_grad=True)
            else:
                self.seasonality2_smoothing_coeffs[f] = torch.nn.Parameter(
                    u1.sample(), requires_grad=True)
            # self.seasonality1_smoothing_coeffs[f] = torch.nn.Parameter(
            #     u1.sample(), requires_grad=True)
            self.seasonality2_smoothing_coeffs[f] = torch.nn.Parameter(
                u1.sample(), requires_grad=True)
            # self.hourly_seasonality_params[f] = [
            #     torch.nn.Parameter(u2.sample(), requires_grad=True)
            #     for _ in range(seasonality_1)
            # ]
            if init_seasonality:
                self.weekly_seasonality_params[f] = [
                    torch.nn.Parameter(
                        torch.tensor(s, dtype=torch.double), requires_grad=True
                    )
                    for s in init_seasonality
                ]
            else:
                self.weekly_seasonality_params[f] = [
                    torch.nn.Parameter(u2.sample(), requires_grad=True)
                    for _ in range(seasonality_2)
                ]

            # Register the parameters with the model
            self.register_parameter(f + " level smoothing",
                                    self.level_smoothing_coeffs[f])
            # self.register_parameter(f + " seasonality1 smoothing",
            #                         self.seasonality1_smoothing_coeffs[f])
            self.register_parameter(f + " seasonality2 smoothing",
                                    self.seasonality2_smoothing_coeffs[f])

            # for i, p in enumerate(self.hourly_seasonality_params[f]):
            #     self.register_parameter(f + " seasonality1 " + str(i), p)

            for i, p in enumerate(self.weekly_seasonality_params[f]):
                self.register_parameter(f + " seasonality2 " + str(i), p)

        self.drnn = drnn.DRNN(
            input_size,
            hidden_size,
            num_layers,
            dropout=dropout,
            cell_type=cell_type,
            batch_first=batch_first,
            dilations=dilations,
            residuals=residuals
        )

        self.tanh = non_lin.Tanh(hidden_size, hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)

    # Call once at the beginning of every epoch
    def init_hidden_states(self):
        self.hidden = (
            torch.zeros(self.num_layers, self.batch_size, self.hidden_size),
            torch.zeros(self.num_layers, self.batch_size, self.hidden_size)
        )

    # TODO - add Trend as well!! See if improved performance
    def forward(self, x, feature, window_size, output_size, lvp=0,
                hidden=None, std=0.001):
        # Forward receives the entire sequence x = [seq_len]

        # Get the parameters of the current feature
        lvl_smoothing = torch.sigmoid(
            self.level_smoothing_coeffs[feature]
        )
        w_smoothing = torch.sigmoid(
            self.seasonality2_smoothing_coeffs[feature]
        )

        # Create lists holding the ES values
        w_seasons = [
            torch.exp(p) for p in self.weekly_seasonality_params[feature]
        ]

        # Handle initial values
        levels = [x[0] / (w_seasons[0])]
        w_seasons.append(w_seasons[0])

        # List to hold the log differences in the levels, to calculate the LVP
        log_level_diffs = []
        level_var_loss = 0

        # Double seasonality ES-style smoothing formulae
        for i in range(1, len(x[1:]) + 1):
            xi = x[i]
            new_level = lvl_smoothing * xi / (w_seasons[i]) +\
                        (1 - lvl_smoothing) * levels[i - 1]
            new_w_season = w_smoothing * xi / (new_level) + \
                           (1 - w_smoothing) * w_seasons[i]

            levels.append(new_level)
            w_seasons.append(new_w_season)

            log_level_diffs.append(torch.log(new_level / levels[i - 1]))

        # Calculate the LVP
        if lvp > 0:
            log_level_diffs_squared = []

            for i in range(1, len(log_level_diffs)):
                log_level_diffs_squared.append(
                    (log_level_diffs[i] - log_level_diffs[i - 1]) ** 2
                )

            level_var_loss = torch.mean(
                torch.tensor(log_level_diffs_squared)
            ) * lvp

        # Create sliding window inputs into the RNN
        inputs = []
        labels = []

        for i in range(len(x) - window_size - output_size + 1):
            # Get the input/label windows of data
            inp = x[i: i + window_size]
            label = x[i + window_size: i + window_size + output_size]

            # Get the level/seasonality values
            level = levels[i + window_size]
            w_seas_in = torch.tensor(
                w_seasons[i: i + window_size], dtype=torch.double
            )
            w_seas_label = torch.tensor(
                w_seasons[i + window_size: i + window_size + output_size],
                dtype=torch.double
            )

            # De-seasonalise/de-level the input/label values and add noise
            norm_input = inp / (level * w_seas_in)
            squashed_input = torch.log(norm_input)
            n = Normal(torch.tensor([0.0]), torch.tensor([std]))
            n_sample = n.sample([squashed_input.shape[0]]).view(-1)
            # Remove extra dimension ^
            noisy_norm_input = n_sample + squashed_input

            norm_label = label / (level * w_seas_label)
            squashed_label = torch.log(norm_label)
            n = Normal(torch.tensor([0.0]), torch.tensor([std]))
            n_sample = n.sample([squashed_label.shape[0]]).view(-1)
            noisy_norm_label = n_sample + squashed_label

            inputs.append(noisy_norm_input.unsqueeze(0))  # Unsqueeze b4 cat
            labels.append(noisy_norm_label.unsqueeze(0))

        inputs = torch.cat(inputs).unsqueeze(2)  # Unsqueeze to correct dim
        labels = torch.cat(labels)

        # Feed inputs in to Dilated LSTM
        if hidden is None:
            lstm_out, hidden = self.drnn(inputs.double())
            h_out = hidden[0][-1]
            # hidden = (h, c), where h is a list of num_layers items,
            # where num_layers is the number of layers in the DRNN,
            # where each item is of shape (1, original_batch_size *
            # dilation, hidden_size), where dilation is the dilation for the
            # given layer. Each one of these tensors represents the final
            # hidden state (i.e the hidden state for t = seq_len, the last
            # input sequence), for each of the original_batch_size *
            # dilation inputs into the final layer. Therefore we get h
            # rather than c by specifying [0], and then get the (1, o_b_s *
            # dilation, hidden_size) of the final layer by specifying [-1]

        else:
            # TODO - this isn't used, but it's wrong
            lstm_out, h_out = self.drnn(inputs.double(), hidden)

        # Pass DLSTM output through non-linear and linear layers
        linear_in = h_out[:, -inputs.size(0):, :].view(-1, self.hidden_size)
        out = self.linear(self.tanh(linear_in))

        # Save the level and seasonality values so that we can use them to make
        # predictions
        if feature == "total load actual":
            self.levels = levels
            self.w_seasons = w_seasons

        # Return model out, actual out, Level variability loss
        return out, labels, level_var_loss

    # Assume that the training data used finished at the end of a season,
    # and that validation/testing data begins at the beginning of the next
    # new season. We feed in the last window_size values from the training
    # data set, plus any extra, to generate forecasts.

    # The data we input for prediction must finish at the end of the
    # training data, be continuous and be of length window_size. If we pass in
    # data that extends into the test, then the final seasonality values are
    # repeated, along with the final length. We ca neither specify to make
    # contiguous forecasts, or we can specify to make overlapping forecasts.
    #
    # Todo 1. Ensure that the seasonality is correctly handled in this way
    # Todo 2. Add the contiguous/non-contiguous functionality. If you pass
    #  in cont=True, you'll get a 1d tensor of contiguous forecasts. If
    #  cont=True, you'll get a 2d tensor of overlapping forecasts.
    def predict(self, x, window_size, output_size, hidden=None, cont=False):
        # Get the final ES level and seasonality values
        levels = self.levels[:]
        w_seasons = self.w_seasons[:]

        # Replicate final seasonality and level values. Remember that we have
        # self.seasonality_2 extra values from the forward function! See
        # piece of paper for data breakdown if you forget/get confused!
        if len(x) > self.seasonality_2 + window_size:
            # Get the final seasonality values
            start = len(w_seasons) - self.seasonality_2

            for i in range(len(x) - window_size - self.seasonality_2):
                w_seasons.append(w_seasons[start + (i % self.seasonality_2)])

        levels.extend([levels[-1] for _ in range(len(x) - window_size)])

        # Get only the final levels and seasonality values that we need
        levels = self.levels[-len(x):]
        w_seasons = self.w_seasons[-len(x):]

        inputs = []
        actuals = []
        output_levels = []
        output_wseas = []

        # If we want contiguous output, we need to skip by output size (=
        # two days). If not, we just move forward one day
        step_size = output_size if cont else 24

        for i in range(0, len(x) - window_size - output_size + 1, step_size):
            # Get the input/label windows of data
            inp = x[i: i + window_size]
            label = x[i + window_size: i + window_size + output_size]

            # Get the level/seasonality values
            level = levels[i + window_size]

            w_seas_in = torch.tensor(
                w_seasons[i: i + window_size], dtype=torch.double
            )

            w_seas_out = torch.tensor(
                w_seasons[i + window_size: i + window_size + output_size],
                dtype=torch.double
            )

            # De-seasonalise/de-level the input/values
            norm_input = torch.log(inp / (level * w_seas_in))
            inputs.append(norm_input.unsqueeze(0))

            # Add the output levels/seasonality to the output vectors,
            # to be used for reseasoning/relevelling the data
            output_wseas.append(w_seas_out.unsqueeze(0))
            output_levels.append(torch.tensor([level for _ in range(
                output_size)], dtype=torch.double).unsqueeze(0))

            # Keep track of the actual data too
            actuals.append(label.unsqueeze(0))

        # Turn list of inputs into a vector, then add another dimension (
        # required for the feature input to the LSTM)
        inputs = torch.cat(inputs).unsqueeze(2)

        # Feed inputs in to Dilated LSTM
        if hidden is None:
            lstm_out, hidden = self.drnn(inputs.double())
            h_out = hidden[0][-1]

        else:
            lstm_out, h_out = self.drnn(inputs.double(), hidden)

        # Pass DLSTM output through linear layer. See piece of paper for why
        # we take the final 'original batch size' outputs. Don't get
        # confused by this again, you understand it now!!
        linear_in = h_out[:, -inputs.size(0):, :].view(-1, self.hidden_size)

        # If we have been generating contiguous outputs, then we transform
        # num_input_sequences x 48 output vector into a 1 x (
        # num_input_sequences x 48) vector
        if cont:
            out = self.linear(self.tanh(linear_in)).view(-1)
            output_wseas = torch.cat(output_wseas).view(-1)
            output_levels = torch.cat(output_levels).view(-1)
            actuals = torch.cat(actuals).view(-1)
        else:
            out = self.linear(self.tanh(linear_in))
            output_wseas = torch.cat(output_wseas)
            output_levels = torch.cat(output_levels)
            actuals = torch.cat(actuals)

        # Unsquash the output, and re-add the final level and seasonality
        pred = torch.exp(out) * output_levels * output_wseas

        # TODO - CHECK WHY NOT ENOUGH FINAL SEASONALITY ARE BEING OUTPUTTED,
        #  AND WHY THE MASE IS SO MUCH WORSE FOR THE ES-RNN WHEN IT CLEARLY
        #  SHOULD BE BETTER (I THINK, UNLESS IT IS JUST SHIFTING??)
        # Return the prediction, and also the final seasonalities and levels
        return (
            pred,
            actuals,
            output_levels[0],
            output_wseas[-self.seasonality_2:]
        )
