# Stateful, mini-batch trained DRNN. One feature.
import torch
import torch.nn as nn
from torch.distributions import Normal, Uniform
from ml import drnn, non_lin, helpers


class ES_RNN_MULT(nn.Module):
    def __init__(self, output_size, input_size, batch_size, hidden_size,
                 num_layers, features, seasonality, dropout=0,
                 cell_type='LSTM', batch_first=False, dilations=None,
                 residuals=tuple([[]]), init_seasonality=None,
                 init_level_smoothing=None, init_seas_smoothing=None):

        super().__init__()

        # RNN Hyperparameters
        self.output_size = output_size
        self.input_size = input_size
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cell_type = cell_type

        # List of names of the features
        self.features = features

        # Seasonality values
        self.seasonality = seasonality

        # ES parameters
        self.level_smoothing_coeffs = {}
        self.seasonality_smoothing_coeffs = {}
        self.init_seasonality_params = {}

        # ES Saved values
        self.levels = {}
        self.seasonals = {}

        # Add all parameters to the network
        u1 = Uniform(-1, 1)  # Smoothing coefficients
        u2 = Uniform(0.65, 1.35)  # Seasonality parameters
        for f in features:
            # Level smoothing parameters
            if init_level_smoothing:
                self.level_smoothing_coeffs[f] = torch.nn.Parameter(
                    torch.tensor(init_level_smoothing[f], dtype=torch.double),
                    requires_grad=True)
            else:
                self.level_smoothing_coeffs[f] = torch.nn.Parameter(
                    u1.sample(), requires_grad=True)

            # Seasonality smoothing parameters
            if init_seas_smoothing:
                self.seasonality_smoothing_coeffs[f] = torch.nn.Parameter(
                    torch.tensor(init_seas_smoothing[f], dtype=torch.double),
                    requires_grad=True)
            else:
                self.seasonality_smoothing_coeffs[f] = torch.nn.Parameter(
                    u1.sample(), requires_grad=True)

            # Initial seasonality parameters
            if init_seasonality:
                self.init_seasonality_params[f] = [
                    torch.nn.Parameter(torch.tensor(s, dtype=torch.double),
                                       requires_grad=True)
                    for s in init_seasonality[f]
                ]
            else:
                self.init_seasonality_params[f] = [
                    torch.nn.Parameter(u2.sample(), requires_grad=True)
                    for _ in range(seasonality)
                ]

            # Register all parameters
            self.register_parameter(f + " level smoothing",
                                    self.level_smoothing_coeffs[f])
            self.register_parameter(f + " seasonality smoothing",
                                    self.seasonality_smoothing_coeffs[f])
            for i, p in enumerate(self.init_seasonality_params[f]):
                self.register_parameter(f + " seasonality " + str(i), p)

        # dLSTM Component
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

        # Non-linear and linear components
        self.tanh = non_lin.Tanh(hidden_size, hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, data, window_size, output_size, lvp=0, std=0.001):
        all_levels = {}
        all_seasonals = {}
        all_inputs = []
        labels = []
        level_var_losses = {}

        for f in self.features:
            # Get the data for the current features
            x = torch.tensor(data[f], dtype=torch.double)

            # Get the parameters of the current feature
            alpha = torch.sigmoid(self.level_smoothing_coeffs[f])
            gamma = torch.sigmoid(self.seasonality_smoothing_coeffs[f])

            # Create lists holding the ES values
            seasonals = [torch.exp(p) for p in self.init_seasonality_params[f]]

            # Handle initial values
            levels = [x[0] / (seasonals[0])]
            seasonals.append(seasonals[0])

            # List to hold the log differences in the levels to calculate LVP
            log_level_diffs = []

            # Fit the ES model to the feature

            # Double seasonality ES-style smoothing formulae
            for i in range(1, len(x[1:]) + 1):
                xi = x[i]
                new_level = alpha * xi / (seasonals[i]) + (1 - alpha) \
                            * levels[i - 1]
                new_seasonal = gamma * xi / (new_level) + (1 - gamma) \
                            * seasonals[i]

                levels.append(new_level)
                seasonals.append(new_seasonal)
                log_level_diffs.append(torch.log(new_level / levels[i - 1]))

            # Calculate the LVP
            if lvp > 0:
                log_level_diffs_squared = []

                for i in range(1, len(log_level_diffs)):
                    log_level_diffs_squared.append(
                        (log_level_diffs[i] - log_level_diffs[i - 1]) ** 2
                    )

                level_var_losses[f] = torch.mean(
                    torch.tensor(log_level_diffs_squared)
                ) * lvp

            # Save levels and seasonality values
            all_levels[f] = levels
            all_seasonals[f] = seasonals

            # Gaussian noise generator
            n = Normal(torch.tensor([0.0]), torch.tensor([std]))

            inputs = []

            # Sliding window method to create inputs for this feature
            for i in range(len(x) - window_size - output_size + 1):
                # Create input for each feature
                inp = x[i: i + window_size]

                # Get the level/seasonality values
                level = levels[i + window_size]
                seas_in = torch.tensor(seasonals[i: i + window_size],
                                       dtype=torch.double)

                # De-seasonalise/de-level the input
                norm_input = inp / (level * seas_in)
                squashed_input = torch.log(norm_input)
                n_sample = n.sample([squashed_input.shape[0]]).view(-1)
                noisy_norm_inp = n_sample + squashed_input

                # Save input. Unsqueeze before concat
                inputs.append(noisy_norm_inp.unsqueeze(0))

                # Create label:
                if f == "total load actual":
                    # Get output
                    label = x[i + window_size: i + window_size + output_size]

                    # Get corresponding seasonality
                    seas_label = torch.tensor(
                        seasonals[
                        i + window_size: i + window_size + output_size],
                        dtype=torch.double
                    )
                    # De-seasonalise, normalise, and take logs
                    norm_label = torch.log(label / (level * seas_label))

                    # Save label. Unsqueeze before concat
                    labels.append(norm_label.unsqueeze(0))

            # Concatenate all the inputs for this feature into one tensor,
            # then append to the list of all the inputs for all the features
            all_inputs.append(torch.cat(inputs).unsqueeze(2))

        # Concatenate the tensor from each feature into a single tensor
        labels = torch.cat(labels, dim=0)
        inputs = torch.cat(all_inputs, dim=2)

        # Feed inputs in to Dilated LSTM
        _, hidden = self.drnn(inputs.double())
        lstm_out = hidden[0][-1]

        # Pass DLSTM output through non-linear and linear layers
        linear_in = lstm_out[:, -inputs.size(0):, :].view(-1, self.hidden_size)
        out = self.linear(self.tanh(linear_in))

        # Save levels and seasonality values for all of the features
        self.levels = all_levels
        self.seasonals = all_seasonals

        # print(len(all_levels))
        # print(len(all_seasonals))
        # print(all_levels.keys())
        # print(all_seasonals.keys())
        # print(len(all_levels["total load actual"]))
        # print(len(all_seasonals["total load actual"]))
        # print(all_levels)
        # print(all_seasonals)
        # import sys
        # sys.exit(0)

        # Return model out, actual out, Level variability loss
        return out, labels, level_var_losses

    # Stripped down version. Assumes x = [<window-size|<out size>].
    def predict(self, data, window_size, output_size):
        all_inputs = []

        # Return the levels and seasonality corresponding to all x, and the
        # actual values corresponding to the output values
        out_actuals = []

        for f in self.features:
            x = torch.tensor(data[f], dtype=torch.double)

            #  Get the final ES level and seasonality values
            levels = self.levels[f][:]
            seasonals = self.seasonals[f][:]

            # Replicate level and seasonality values
            start = len(seasonals) - self.seasonality

            for i in range(len(x) - window_size):
                seasonals.append(seasonals[start + (i % self.seasonality)])

            levels.extend([levels[-1] for _ in range(len(x) - window_size)])

            # Get only the final levels and seasonality values that we need
            levels = levels[-len(x):]
            seasonals = seasonals[-len(x):]
            inputs = []

            for i in range(0, len(x) - window_size - output_size + 1, 24):
                # Get the input/label windows of data
                inp = x[i: i + window_size]
                label = x[i + window_size: i + window_size + output_size]

                # Get the level/seasonality values
                level = levels[i + window_size]

                seas_in = torch.tensor(
                    seasonals[i: i + window_size], dtype=torch.double
                )

                seas_out = torch.tensor(
                    seasonals[i + window_size: i + window_size + output_size],
                    dtype=torch.double
                )

                # De-seasonalise/de-level the input/values
                norm_input = torch.log(inp / (level * seas_in))
                inputs.append(norm_input.unsqueeze(0))

                if f == "total load actual":
                    out_levels = torch.tensor(
                        [level for _ in range(output_size)],
                        dtype=torch.double
                    )
                    out_seas = seas_out
                    out_actuals = label

            all_inputs.append(torch.cat(inputs).unsqueeze(2))

        # Convert into tensor
        inputs = torch.cat(all_inputs, dim=2)

        # Input into LSTM
        _, hidden = self.drnn(inputs.double())
        lstm_out = hidden[0][-1]

        # Pass DLSTM output through tanh and linear layers
        linear_in = lstm_out[:, -inputs.size(0):, :].view(-1, self.hidden_size)
        out = self.linear(self.tanh(linear_in))

        # Unsquash the output, and re-add the final level and seasonality
        pred = torch.exp(out) * out_levels * out_seas

        # Return the prediction, and also the final seasonalities and levels
        return (
            pred,           # 48 hour prediction
            out_actuals,    # 48 hour actual
            out_levels,     # 48 hour levels
            out_seas,       # 48 hour seasonality
            levels,         # Levels for all of x
            seasonals,      # Seasonality for all x
            torch.exp(out),            # Output from LSTM
        )