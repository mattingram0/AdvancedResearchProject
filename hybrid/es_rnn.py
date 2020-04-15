# Stateful, mini-batch trained DRNN. One feature.
import torch
import torch.nn as nn
from torch.distributions import Normal, Uniform
from ml import drnn, non_lin, helpers
import sys

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
                    torch.tensor(init_level_smoothing[f], dtype=torch.double),
                    requires_grad=True)
            else:
                self.level_smoothing_coeffs[f] = torch.nn.Parameter(
                    u1.sample(), requires_grad=True)

            if init_seas_smoothing:
                self.seasonality2_smoothing_coeffs[f] = torch.nn.Parameter(
                    torch.tensor(init_seas_smoothing[f], dtype=torch.double),
                    requires_grad=True)
            else:
                self.seasonality2_smoothing_coeffs[f] = torch.nn.Parameter(
                    u1.sample(), requires_grad=True)

            if init_seasonality:
                self.weekly_seasonality_params[f] = [
                    torch.nn.Parameter(
                        torch.tensor(s, dtype=torch.double), requires_grad=True
                    )
                    for s in init_seasonality[f]
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
            # TODO YOU ARE HERE????/ SEE OUTPUT AND THE 1417??
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

        # Used for testing whether the LSTM was actually beneficial
        self.tanh_no_lstm = non_lin.Tanh(336, 336)
        self.linear_no_lstm = nn.Linear(336, output_size)

        # TODO REMOVE
        self.counter = 0

    # Call once at the beginning of every epoch
    def init_hidden_states(self):
        self.hidden = (
            torch.zeros(self.num_layers, self.batch_size, self.hidden_size),
            torch.zeros(self.num_layers, self.batch_size, self.hidden_size)
        )

    def create_batch(self, x, feature, window_size, output_size, lvp):
        pass

    # TODO - add Trend as well!! See if improved performance
    def forward(self, x, feature, window_size, output_size, lvp=0,
                hidden=None, std=0.001, skip_lstm=False):
        self.counter += 1

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

            # Used for one of the plotting functions, ignore
            if i == 7 * 24 * 4:
                j = i
                data_subset = x[i - (7 * 24): i + (3 * 7 * 24)]

        labels = torch.cat(labels)

        if skip_lstm:
            # Instead of feeding the inputs through the LSTM, skip and input
            # them directly through the tanh layer into the linear layer
            inputs = torch.cat(inputs)
            out = self.linear_no_lstm(self.tanh_no_lstm(inputs.double()))
        else:
            inputs = torch.cat(inputs).unsqueeze(2)  # Unsqueeze to correct dim
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

        # Used for one of the plotting functions, ignore
        rnn_in = inputs[j]
        data_out = labels[j]
        rnn_out = out[j]
        if self.counter == 25:
            helpers.plot_sliding_window(data_subset, rnn_in, data_out, rnn_out)

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
            torch.exp(out),            # Output from LSTM
        )