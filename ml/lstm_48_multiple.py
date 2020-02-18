import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import sys

from ml.helpers import create_pairs, batch_data


def forecast(data, train_hours, valid_hours, test_hours, window_size,
             output_size, batch_size, in_place):
    # Scale data - scales all 16 features independently
    scaler = MinMaxScaler()
    scaler.fit(data[:train_hours])
    transf_data = scaler.transform(data)

    # Generate training, validation and testing pairs
    training_data, valid_data, test_data = create_pairs(
        transf_data, train_hours, valid_hours, test_hours, window_size,
        output_size, True
    )

    # Training parameters
    num_epochs = 100
    learning_rate = 0.01
    input_size = 16  # 16 features now
    hidden_size = 40
    num_layers = 1
    output_size = 48

    # Create model
    lstm = LSTM(
        output_size, input_size, batch_size, hidden_size, num_layers
    ).double()

    loss_func = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)

    # Train model
    lstm.train()
    train_model(
        lstm, optimizer, loss_func, num_epochs, training_data, batch_size
    )

    # Make predictions
    lstm.eval()
    test_model(lstm, data, valid_data, test_data, train_hours, window_size)


def train_model(lstm, optimizer, loss_func, num_epochs, training_data,
                batch_size):
    # Split into batches
    input_batches, label_batches = batch_data(training_data, batch_size)

    # Epoch loop
    for epoch in range(num_epochs):
        # Reset hidden states at beginning of each epoch
        lstm.init_hidden_states()

        # Mini-batch training
        for i in range(len(input_batches)):
            loss = train_batch(
                lstm, optimizer, loss_func, input_batches[i], label_batches[i]
            )

        if epoch % 10 == 0:
            print("Epoch %d: Loss - %1.5f" % (epoch, loss.item()))


def train_batch(lstm, optimizer, loss_func, inputs, labels):
    outputs = lstm(inputs.double())
    # labels = labels.view(labels.size(0), -1)  # Remove redundant dimension

    optimizer.zero_grad()
    loss = loss_func(outputs, labels)
    loss.backward()
    optimizer.step()
    return loss


def test_model(lstm, data, valid_data, test_data, train_hours, window_size):
    # Fit scaler to just one feature now, to inverse transform the data
    scaler = MinMaxScaler()
    scaler.fit(
        data["total load actual"][:train_hours].to_numpy().reshape(-1, 1)
    )

    # The LSTM requires the first window_size values to make a prediction,
    # and so we don't include them when plotting our actual data
    train_actual = np.array(data["total load actual"][
                                 window_size:train_hours]).reshape(-1)

    valid_norm_prediction = lstm(valid_data[0]).view(-1, 1).detach()
    valid_norm_actual = valid_data[1].view(-1, 1).detach()
    valid_prediction = scaler.inverse_transform(
        valid_norm_prediction).reshape(-1)
    valid_actual = scaler.inverse_transform(valid_norm_actual).reshape(-1)

    test_norm_prediction = lstm(test_data[0]).view(-1, 1).detach()
    test_norm_actual = test_data[1].view(-1, 1).detach()
    test_prediction = scaler.inverse_transform(
        test_norm_prediction).reshape(-1)
    test_actual = scaler.inverse_transform(test_norm_actual).reshape(-1)

    prediction = np.concatenate((valid_prediction, test_prediction))
    actual = np.concatenate((train_actual, valid_actual, test_actual))

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
    plt.show()


# Stateful, mini-batch trained LSTM. One feature.
class LSTM(nn.Module):
    def __init__(self, output_size, input_size, batch_size, hidden_size,
                 num_layers):
        super().__init__()

        self.output_size = output_size
        self.input_size = input_size
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Allows the LSTM to be stateful between mini-batches
        self.hidden = (torch.zeros(num_layers, batch_size, hidden_size,
                                   dtype=torch.double),
                       torch.zeros(num_layers, batch_size, hidden_size,
                                   dtype=torch.double))

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.linear = nn.Linear(hidden_size, output_size)

    # Call once at the beginning of every epoch
    def init_hidden_states(self):
        self.hidden = (
            torch.zeros(self.num_layers, self.batch_size, self.hidden_size,
                        dtype=torch.double),
            torch.zeros(self.num_layers, self.batch_size, self.hidden_size,
                        dtype=torch.double)
        )

    def forward(self, x):
        # print("********")
        # print("LSTM Input: ", x.size())
        lstm_out, (h_out, c_out) = self.lstm(x.double())

        # The LSTM model does not model the temporal dependency between
        # samples within a batch, only the dependency of time steps within a
        # sequence. So possibly need to include sequences which are more
        # than one week long, to get the weekly seasonality. We can
        # arbitrarily select any batch size to form a batch, and the
        # sequences within a batch fo not need to be from continuous time
        # segments. No need to have a batch of data as input when making
        # prediction, just a single set of seq_len data points.
        self.hidden = (h_out, c_out)
        # print("LSTM Output:", lstm_out.size())
        # print("Hidden Output:", h_out.size())
        # print("Internal Output:", c_out.size())
        linear_in = h_out.view(-1, self.hidden_size)  # Just gets rid of
        # extra dimension added by having more than one layer
        # print("Linear Input:", linear_in.size())
        out = self.linear(linear_in)
        # print("Linear Output", out.size())
        # print("********")

        return out
