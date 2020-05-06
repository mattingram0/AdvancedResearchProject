import torch
import torch.nn as nn
from torch.distributions import Normal, Uniform
from ml import drnn, non_lin


# Final implementation of the ES-RNN-S/D (W) model
class ES_RNN_S(nn.Module):
    def __init__(self, output_size, input_size, batch_size, hidden_size,
                 num_layers, demand_features, weather_features, seasonality,
                 dropout=0, cell_type='LSTM', batch_first=False,
                 dilations=None, residuals=tuple([[]]), init_seasonality=None,
                 init_level_smoothing=None, init_seas_smoothing=None):

        super().__init__()

        # RNN Hyperparameters
        self.output_size = output_size
        self.input_size = input_size
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cell_type = cell_type

        # List of features
        self.demand_features = demand_features
        self.weather_features = weather_features

        # Seasonality values
        self.seasonality = seasonality

        # ES parameters
        self.level_smoothing_coeffs = {}
        self.seasonality_smoothing_coeffs = {}
        self.init_seasonality_params = {}

        # ES level and seasonality values (for total load actual only)
        self.levels = []
        self.seasonals = []

        # Add all parameters to the network
        u1 = Uniform(-1, 1)  # Smoothing coefficients
        u2 = Uniform(0.65, 1.35)  # Seasonality parameters
        for f in demand_features:
            # Create the parameters
            if init_level_smoothing:
                self.level_smoothing_coeffs[f] = torch.nn.Parameter(
                    torch.tensor(init_level_smoothing[f], dtype=torch.double),
                    requires_grad=True)
            else:
                self.level_smoothing_coeffs[f] = torch.nn.Parameter(
                    u1.sample(), requires_grad=True)

            if init_seas_smoothing:
                self.seasonality_smoothing_coeffs[f] = torch.nn.Parameter(
                    torch.tensor(init_seas_smoothing[f], dtype=torch.double),
                    requires_grad=True)
            else:
                self.seasonality_smoothing_coeffs[f] = torch.nn.Parameter(
                    u1.sample(), requires_grad=True)

            if init_seasonality:
                self.init_seasonality_params[f] = [
                    torch.nn.Parameter(
                        torch.tensor(s, dtype=torch.double), requires_grad=True
                    )
                    for s in init_seasonality[f]
                ]
            else:
                self.init_seasonality_params[f] = [
                    torch.nn.Parameter(u2.sample(), requires_grad=True)
                    for _ in range(seasonality)
                ]

            # Register the parameters with the model
            self.register_parameter(f + " level smoothing",
                                    self.level_smoothing_coeffs[f])
            self.register_parameter(f + " seasonality smoothing",
                                    self.seasonality_smoothing_coeffs[f])
            for i, p in enumerate(self.init_seasonality_params[f]):
                self.register_parameter(f + " seasonality " + str(i), p)

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

        # Used for testing whether the LSTM was actually beneficial
        self.tanh_no_lstm = non_lin.Tanh(336, 336)
        self.linear_no_lstm = nn.Linear(336, output_size)

    def forward(self, data, feature, window_size, output_size, weather, lvp=0,
                std=0.001, skip_lstm=False):
        n = Normal(torch.tensor([0.0]), torch.tensor([std]))
        x = torch.tensor(data[feature], dtype=torch.double)

        # Forward receives the entire sequence x = [seq_len]

        # Get the parameters of the current feature
        lvl_smoothing = torch.sigmoid(
            self.level_smoothing_coeffs[feature]
        )
        seas_smoothing = torch.sigmoid(
            self.seasonality_smoothing_coeffs[feature]
        )

        # Create lists holding the ES values
        seasonals = [
            torch.exp(p) for p in self.init_seasonality_params[feature]
        ]

        # Handle initial values
        levels = [x[0] / (seasonals[0])]
        seasonals.append(seasonals[0])

        # List to hold the log differences in the levels, to calculate the LVP
        log_level_diffs = []
        level_var_loss = 0

        # Double seasonality ES-style smoothing formulae
        for i in range(1, len(x[1:]) + 1):
            xi = x[i]
            new_level = lvl_smoothing * xi / (seasonals[i]) +\
                        (1 - lvl_smoothing) * levels[i - 1]
            new_w_season = seas_smoothing * xi / (new_level) + \
                           (1 - seas_smoothing) * seasonals[i]

            levels.append(new_level)
            seasonals.append(new_w_season)

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
                seasonals[i: i + window_size], dtype=torch.double
            )
            w_seas_label = torch.tensor(
                seasonals[i + window_size: i + window_size + output_size],
                dtype=torch.double
            )

            # De-seasonalise/de-level the input/label values and add noise
            norm_input = inp / (level * w_seas_in)
            squashed_input = torch.log(norm_input)
            n_sample = n.sample([squashed_input.shape[0]]).view(-1)
            noisy_norm_input = n_sample + squashed_input
            squashed_norm_label = torch.log(label / (level * w_seas_label))

            inputs.append(noisy_norm_input.unsqueeze(0))  # Unsqueeze b4 cat
            labels.append(squashed_norm_label.unsqueeze(0))

        labels = torch.cat(labels)

        if skip_lstm:
            inputs = torch.cat(inputs)
            out = self.linear_no_lstm(self.tanh_no_lstm(inputs.double()))
        else:

            # Sliding window approach to add the weather features if including
            if weather:
                all_inputs = [torch.cat(inputs).unsqueeze(2)]

                for f in self.weather_features:
                    x = torch.tensor(data[f], dtype=torch.double)
                    inputs = []

                    for i in range(len(x) - window_size - output_size + 1):
                        inputs.append(x[i: i + window_size].unsqueeze(0))

                    all_inputs.append(torch.cat(inputs).unsqueeze(2))

                inputs = torch.cat(all_inputs, dim=2)
            else:
                inputs = torch.cat(inputs).unsqueeze(2)
                # Have to unsqueeze to add the single 'feature' dimension

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


            # Pass DLSTM output through non-linear and linear layers
            linear_in = h_out[:, -inputs.size(0):, :].view(-1, self.hidden_size)
            out = self.linear(self.tanh(linear_in))

        # Save the level and seasonality values so that we can use them to make
        # predictions
        if feature == "total load actual":
            self.levels = levels
            self.seasonals = seasonals

        # Return model out, actual out, Level variability loss
        return out, labels, level_var_loss

    # The data we input for prediction must finish at the end of the
    # training data, be continuous and be of length window_size. If we pass in
    # data that extends into the test, then the final seasonality values are
    # repeated, along with the final length. We ca neither specify to make
    # contiguous forecasts, or we can specify to make overlapping forecasts.
    def predict(self, data, window_size, output_size, weather,
                cont=False, skip_lstm=False):
        x = torch.tensor(data["total load actual"], dtype=torch.double)

        # Get the final ES level and seasonality values
        levels = self.levels[:]
        seasonals = self.seasonals[:]

        # Replicate final seasonality and level values. Remember that we have
        # self.seasonality extra initial values
        start = len(seasonals) - self.seasonality

        for i in range(len(x) - window_size):
            seasonals.append(seasonals[start + (i % self.seasonality)])

        levels.extend([levels[-1] for _ in range(len(x) - window_size)])

        # Get only the final levels and seasonality values that we need
        levels = levels[-len(x):]
        seasonals = seasonals[-len(x):]

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
                seasonals[i: i + window_size], dtype=torch.double
            )

            w_seas_out = torch.tensor(
                seasonals[i + window_size: i + window_size + output_size],
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

        # Instead of feeding the inputs through the LSTM, skip and input
        # them directly through the tanh layer into the linear layer
        if skip_lstm:
            inputs = torch.cat(inputs)
            out = self.linear_no_lstm(self.tanh_no_lstm(inputs.double()))
        else:

            # Sliding window approach to add the weather features if including
            if weather:
                all_inputs = [torch.cat(inputs).unsqueeze(2)]

                for f in self.weather_features:
                    x = torch.tensor(data[f], dtype=torch.double)
                    inputs = []

                    for i in range(len(x) - window_size - output_size + 1):
                        inputs.append(x[i: i + window_size].unsqueeze(0))

                    all_inputs.append(torch.cat(inputs).unsqueeze(2))

                inputs = torch.cat(all_inputs, dim=2)
            else:
                inputs = torch.cat(inputs).unsqueeze(2)
                # Have to unsqueeze to add the single 'feature' dimension

            # Feed inputs in to Dilated LSTM
            lstm_out, hidden = self.drnn(inputs.double())
            h_out = hidden[0][-1]

            # Pass DLSTM output through linear layer. See piece of paper for why
            # we take the final 'original batch size' outputs.
            linear_in = h_out[:, -inputs.size(0):, :].view(-1, self.hidden_size)
            out = self.linear(self.tanh(linear_in))

        # If we have been generating contiguous outputs, then we transform
        # num_input_sequences x 48 output vector into a 1 x (
        # num_input_sequences x 48) vector
        if cont:
            out = out.view(-1)
            output_wseas = torch.cat(output_wseas).view(-1)
            output_levels = torch.cat(output_levels).view(-1)
            actuals = torch.cat(actuals).view(-1)
        else:
            output_wseas = torch.cat(output_wseas)
            output_levels = torch.cat(output_levels)
            actuals = torch.cat(actuals)

        # Unsquash the output, and re-add the final level and seasonality
        pred = torch.exp(out) * output_levels * output_wseas

        # Return the prediction and other debugging information
        return (
            pred,                       # 48 hour prediction
            actuals,                    # 48 hour actual
            output_levels,              # 48 hour levels
            output_wseas,               # 48 hour seasonality
            levels,                     # Levels for all of x
            seasonals,                  # Seasonality for all x
            torch.exp(out),            # Output from LSTM
        )