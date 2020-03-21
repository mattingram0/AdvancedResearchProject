import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.distributions import Normal, Uniform
import torch
import sys

from sklearn.preprocessing import MinMaxScaler

from ml import drnn, non_lin
from ml.helpers import pinball_loss


def forecast(data, train_hours, valid_hours, test_hours, window_size,
             output_size, batch_size, in_place):

    # Training parameters
    num_epochs = 50
    init_learning_rate = 0.001
    input_size = 1
    hidden_size = 40
    num_layers = 4
    dilations = [1, 4, 24, 168]
    level_variability_penalty = 10
    percentile = 0.49
    loss_func = pinball_loss
    grad_clipping = 20
    auto_lr = False  # Automatically adjust learning rates
    variable_lr = True  # Use list of epoch/rate pairs
    variable_rates = {7: 5e-3, 18: 1e-3, 22: 3e-4}
    auto_rate_threshold = 1.005  # If loss(x - 1) < 1.005 * loss(x) reduce rate
    min_epochs_before_change = 2

    # Create model
    lstm = ES_RNN(
        output_size, input_size, batch_size, hidden_size, num_layers,
        batch_first=True, dilations=dilations, features=data.columns,
        seasonality_1=24, seasonality_2=168
    ).double()

    # Register gradient clipping function
    for p in lstm.parameters():
        p.register_hook(
            lambda grad: torch.clamp(grad, -grad_clipping, grad_clipping))

    for n, p in lstm.named_parameters():
        print(n, p.shape)
        print(len(list(lstm.named_parameters())))

    # Train model
    lstm.train()
    train_model(lstm, data[:train_hours], window_size, output_size,
                level_variability_penalty, loss_func, num_epochs,
                init_learning_rate, percentile, auto_lr, variable_lr,
                auto_rate_threshold, min_epochs_before_change, variable_rates)

    # Make predictions
    lstm.eval()
    test_model(lstm, data, train_hours, valid_hours, test_hours,
               window_size, output_size, dilations)


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

    num_epochs_since_change = 0
    rate_changed = False
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

            if variable_lr and epoch in list(variable_rates.keys()):
                for param_group in optimizer.param_groups:
                    param_group['lr'] = variable_rates[epoch]

                if name == "total load actual":
                    print("Changed Learning Rate to: " + str(variable_rates
                                                             [epoch]))

            if auto_lr and num_epochs_since_change >= min_epochs_since_change:
                if j

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

            if not rate_changed:
                num_epochs_since_change += 1


def test_model(lstm, data, train_hours, valid_hours, test_hours,
               window_size, output_size, dilations):
    train_data = data[window_size:train_hours]
    valid_data = data[train_hours - window_size:
                      train_hours + valid_hours]
    test_data = data[train_hours + valid_hours - window_size:
                     train_hours + valid_hours + test_hours]

    valid_pred, valid_actual, out_levels, out_seas1, out_seas2 = lstm.predict(
        torch.tensor(valid_data["total load actual"], dtype=torch.double),
        window_size, output_size
    )

    # test_prediction = lstm.predict(
    #     test_data["total load actual"], window_size, output_size
    # )

    train_actual = np.array(data["total load actual"][
                            1008:train_hours]).reshape(-1)

    prediction = np.array(valid_pred.detach())
    actual = np.concatenate((np.array(train_actual), np.array(valid_actual)))

    x1 = np.array([range(len(actual))]).reshape(-1)
    x2 = np.array([range(len(actual) - len(prediction),
                         len(actual))]).reshape(-1)

    plt.figure(figsize=(12.8, 9.6), dpi=250)
    plt.plot(x1, actual, label="Actual Data")
    plt.plot(x2, prediction, label="Prediction")

    # plt.axvline(x=train_hours - window_size, c='orange', label="Training Data")
    # plt.axvline(x=train_hours - window_size + len(valid_actual), c='purple',
    #             label="Validation Data")
    plt.gca().set_xlabel("Time")
    plt.gca().set_ylabel("Total Energy Demand")
    plt.gca().legend(loc="best")
    plt.gca().set_title('Dilations: ' + str(dilations))
    plt.show()


# Stateful, mini-batch trained DRNN. One feature.
class ES_RNN(nn.Module):
    def __init__(self, output_size, input_size, batch_size, hidden_size,
                 num_layers, features, seasonality_1, seasonality_2, dropout=0,
                 cell_type='LSTM', batch_first=False, dilations=None):

        super().__init__()

        # RNN Hyperparameters
        self.output_size = output_size
        self.input_size = input_size
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cell_type = cell_type

        # Seasonality values
        self.seasonality_1 = seasonality_1
        self.seasonality_2 = seasonality_2

        # ES parameters
        self.level_smoothing_coeffs = {}
        self.seasonality1_smoothing_coeffs = {}
        self.seasonality2_smoothing_coeffs = {}
        self.hourly_seasonality_params = {}
        self.weekly_seasonality_params = {}

        # ES level and seasonality values (for total load actual only)
        self.levels = []
        self.h_seasons = []
        self.w_seasons = []

        # Add all parameters to the network
        u = Uniform(-0.5, 0.5)  # Random initialisation of parameters
        for f in features:
            # Create the parameters
            self.level_smoothing_coeffs[f] = torch.nn.Parameter(
                u.sample(), requires_grad=True)
            self.seasonality1_smoothing_coeffs[f] = torch.nn.Parameter(
                u.sample(), requires_grad=True)
            self.seasonality2_smoothing_coeffs[f] = torch.nn.Parameter(
                u.sample(), requires_grad=True)
            self.hourly_seasonality_params[f] = [
                torch.nn.Parameter(u.sample(), requires_grad=True)
                for _ in range(seasonality_1)
            ]
            self.weekly_seasonality_params[f] = [
                torch.nn.Parameter(u.sample(), requires_grad=True)
                for _ in range(seasonality_2)
            ]

            # Register the parameters with the model
            self.register_parameter(f + " level smoothing",
                                    self.level_smoothing_coeffs[f])
            self.register_parameter(f + " seasonality1 smoothing",
                                    self.seasonality1_smoothing_coeffs[f])
            self.register_parameter(f + " seasonality2 smoothing",
                                    self.seasonality2_smoothing_coeffs[f])

            for i, p in enumerate(self.hourly_seasonality_params[f]):
                self.register_parameter(f + " seasonality1 " + str(i), p)

            for i, p in enumerate(self.weekly_seasonality_params[f]):
                self.register_parameter(f + " seasonality2 " + str(i), p)

        self.drnn = drnn.DRNN(
            input_size,
            hidden_size,
            num_layers,
            dropout=dropout,
            cell_type=cell_type,
            batch_first=batch_first,
            dilations=dilations
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
        h_smoothing = torch.sigmoid(
            self.seasonality1_smoothing_coeffs[feature]
        )
        w_smoothing = torch.sigmoid(
            self.seasonality2_smoothing_coeffs[feature]
        )

        # Create lists holding the ES values
        h_seasons = [
            torch.exp(p) for p in self.hourly_seasonality_params[feature]
        ]
        w_seasons = [
            torch.exp(p) for p in self.weekly_seasonality_params[feature]
        ]

        # Handle initial values
        levels = [x[0] / (h_seasons[0] * w_seasons[0])]
        h_seasons.append(h_seasons[0])
        w_seasons.append(w_seasons[0])

        # List to hold the log differences in the levels, to calculate the LVP
        log_level_diffs = []
        level_var_loss = 0

        # Double seasonality ES-style smoothing formulae
        for i in range(1, len(x[1:]) + 1):
            xi = x[i]

            new_level = lvl_smoothing * xi / (h_seasons[i] * w_seasons[i]) + \
                        (1 - lvl_smoothing) * levels[i - 1]
            new_h_season = h_smoothing * xi / (new_level * w_seasons[i]) + \
                        (1 - h_smoothing) * h_seasons[i]
            new_w_season = w_smoothing * xi / (new_level * h_seasons[i]) + \
                           (1 - w_smoothing) * w_seasons[i]

            levels.append(new_level)
            h_seasons.append(new_h_season)
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
            h_seas_in = torch.tensor(
                h_seasons[i: i + window_size], dtype=torch.double
            )
            w_seas_in = torch.tensor(
                w_seasons[i: i + window_size], dtype=torch.double
            )
            h_seas_label = torch.tensor(
                h_seasons[i + window_size: i + window_size + output_size],
                dtype=torch.double
            )
            w_seas_label = torch.tensor(
                w_seasons[i + window_size: i + window_size + output_size],
                dtype=torch.double
            )

            # De-seasonalise/de-level the input/label values and add noise
            norm_input = inp / (level * h_seas_in * w_seas_in)
            squashed_input = torch.log(norm_input)
            n = Normal(torch.tensor([0.0]), torch.tensor([std]))
            n_sample = n.sample([squashed_input.shape[0]]).view(-1)
            # Remove extra dimension ^
            noisy_norm_input = n_sample + squashed_input

            norm_label = label / (level * h_seas_label * w_seas_label)
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

        else:
            lstm_out, h_out = self.drnn(inputs.double(), hidden)

        # Pass DLSTM output through non-linear and linear layers
        linear_in = h_out[:, -inputs.size(0):, :].view(-1, self.hidden_size)
        out = self.linear(self.tanh(linear_in))

        # Save the level and seasonality values so that we can use them to make
        # predictions
        if feature == "total load actual":
            self.levels = levels
            self.h_seasons = h_seasons
            self.w_seasons = w_seasons

        # Return model out, actual out, Level variability loss
        return out, labels, level_var_loss

    # Assume that the training data used finished at the end of a season,
    # and that validation/testing data begins at the beginning of the next
    # new season. We feed in the last window_size values from the training
    # data set, plus any extra, to generate forecasts.
    def predict(self, x, window_size, output_size, hidden=None):
        # Get the final ES level and seasonality values
        levels = self.levels[:]
        h_seasons = self.h_seasons[:]
        w_seasons = self.w_seasons[:]

        # Replicate final seasonality and level values
        if self.seasonality_1 < len(x) - window_size:
            start = len(h_seasons) - self.seasonality_1
            for i in range(len(x) - window_size - self.seasonality_1):
                h_seasons.append(h_seasons[start + (i % self.seasonality_1)])

        if self.seasonality_2 < len(x) - window_size:
            start = len(w_seasons) - self.seasonality_2
            for i in range(len(x) - window_size - self.seasonality_2):
                w_seasons.append(w_seasons[start + (i % self.seasonality_2)])

        levels.extend([levels[-1] for _ in range(len(x) - window_size)])

        # Get only the final levels and seasonality values that we need
        levels = self.levels[-(len(x) + window_size):]
        h_seasons = self.h_seasons[-(len(x) + window_size):]
        w_seasons = self.w_seasons[-(len(x) + window_size):]

        inputs = []
        actuals = []
        output_levels = []
        output_hseas = []
        output_wseas = []

        for i in range(0, len(x) - window_size - output_size + 1, output_size):
            # Get the input/label windows of data
            inp = x[i: i + window_size]
            label = x[i + window_size: i + window_size + output_size]

            # Get the level/seasonality values
            level = levels[i + window_size]
            h_seas_in = torch.tensor(
                h_seasons[i: i + window_size], dtype=torch.double
            )
            w_seas_in = torch.tensor(
                w_seasons[i: i + window_size], dtype=torch.double
            )
            h_seas_out = torch.tensor(
                h_seasons[i + window_size: i + window_size + output_size],
                dtype=torch.double
            )
            w_seas_out = torch.tensor(
                w_seasons[i + window_size: i + window_size + output_size],
                dtype=torch.double
            )

            # De-seasonalise/de-level the input/values
            norm_input = torch.log(inp / (level * h_seas_in * w_seas_in))
            inputs.append(norm_input.unsqueeze(0))

            # Add the output levels/seasonality to the output vectors,
            # to be used for reseasoning/relevelling the data
            output_hseas.append(h_seas_out.unsqueeze(0))
            output_wseas.append(w_seas_out.unsqueeze(0))
            output_levels.append(torch.tensor([level for _ in range(
                output_size)], dtype=torch.double).unsqueeze(0))

            # Keep track of the actual data too
            actuals.append(label.unsqueeze(0))

        inputs = torch.cat(inputs).unsqueeze(2)
        output_hseas = torch.cat(output_hseas).view(-1)
        output_wseas = torch.cat(output_wseas).view(-1)
        output_levels = torch.cat(output_levels).view(-1)
        actuals = torch.cat(actuals).view(-1)

        # Feed inputs in to Dilated LSTM
        if hidden is None:
            lstm_out, hidden = self.drnn(inputs.double())
            h_out = hidden[0][-1]

        else:
            lstm_out, h_out = self.drnn(inputs.double(), hidden)

        # Pass DLSTM output through linear layer
        linear_in = h_out[:, -inputs.size(0):, :].view(-1, self.hidden_size)
        out = self.linear(self.tanh(linear_in)).view(-1)

        # Unsquash the output, and re-add the final level and seasonality
        pred = torch.exp(out) * output_levels * output_hseas * output_wseas

        # Return the prediction, and also the final seasonalities and levels
        return (
            pred,
            actuals,
            output_levels[0],
            output_hseas[:self.seasonality_1],
            output_wseas[:self.seasonality_2]
        )

