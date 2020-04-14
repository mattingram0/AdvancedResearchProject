# Stateful, mini-batch trained DRNN. One feature.
import torch
import torch.nn as nn
from torch.distributions import Normal, Uniform
from ml import drnn, non_lin, helpers


class ES_RNN_EX(nn.Module):
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

    def forward(self, x, window_size, output_size, lvp=0, std=0.001):
        all_levels = {}
        all_seasonals = {}
        level_var_losses = {}

        # Fit the ES model to every feature
        for f in self.features:
            # Get the parameters of the current feature
            alpha = torch.sigmoid(self.level_smoothing_coeffs[f])
            gamma = torch.sigmoid(self.seasonality_smoothing_coeffs[f])

            # Create lists holding the ES values
            seasonals = [torch.exp(p)
                           for p in self.init_seasonality_params[f]]

            # Handle initial values
            levels = [x[0] / (seasonals[0])]
            seasonals.append(seasonals[0])

            # List to hold the log differences in the levels to calculate LVP
            log_level_diffs = []

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

        # Create sliding window inputs into the RNN
        all_inputs = []
        labels = []

        # Gaussian noise generator
        n = Normal(torch.tensor([0.0]), torch.tensor([std]))

        for f in self.features:
            levels = all_levels[f]
            seasonals = all_seasonals[f]
            data = x[f]
            inputs = []

            for i in range(len(data) - window_size - output_size + 1):
                # Create input for each feature
                inp = data[i: i + window_size]

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
                    label = data[i + window_size: i + window_size + output_size]

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

            all_inputs.append(torch.cat(inputs).unsqueeze(2))

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

        # Return model out, actual out, Level variability loss
        return out, labels, level_var_losses

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
    def predict(self, x, window_size, output_size, hidden=None, cont=False,
                skip_lstm=False):
        # Get the final ES level and seasonality values
        levels = self.levels[:]
        w_seasons = self.w_seasons[:]

        # Replicate final seasonality and level values. Remember that we have
        # self.seasonality_2 extra values from the forward function! See
        # piece of paper for data breakdown if you forget/get confused!
        # Get the final seasonality values
        start = len(w_seasons) - self.seasonality_2

        for i in range(len(x) - window_size):
            w_seasons.append(w_seasons[start + (i % self.seasonality_2)])

        levels.extend([levels[-1] for _ in range(len(x) - window_size)])

        # Get only the final levels and seasonality values that we need
        levels = levels[-len(x):]
        w_seasons = w_seasons[-len(x):]

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

        # Instead of feeding the inputs through the LSTM, skip and input
        # them directly through the tanh layer into the linear layer
        if skip_lstm:
            inputs = torch.cat(inputs)
            out = self.linear_no_lstm(self.tanh_no_lstm(inputs.double()))
        else:
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

        # Return the prediction, and also the final seasonalities and levels
        return (
            pred,           # 48 hour prediction
            actuals,        # 48 hour actual
            output_levels,  # 48 hour levels
            output_wseas,   # 48 hour seasonality
            levels,         # Levels for all of x
            w_seasons,      # Seasonality for all x
            out,            # Output from LSTM
            inputs  # TODO MAKE SURE TO REMOVE!!!
        )