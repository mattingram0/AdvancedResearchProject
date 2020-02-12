import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from stats import helpers


def forecast(data, train_hours, test_hours, in_place):
    helpers.indices_adjust(
        data, len(data) - test_hours, test_hours, "multiplicative"
    )

    train_data = torch.log(
        torch.tensor(data[:-test_hours]["seasonally adjusted"].values)
    )

    test_data = torch.log(
        torch.tensor(data[-test_hours:]["seasonally adjusted"].values)
    ).tolist()

    training_pairs = create_pairs(train_data, train_hours)
    model = LSTM()
    loss_function = nn.MSELoss()
    learning_rate = 0.0001
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    epochs = 150

    # Training loop
    for i in range(epochs):
        for input, label in training_pairs:
            optimizer.zero_grad()
            model.hidden_cell = (
                torch.zeros(1, 1, model.hidden_layer_size).double(),
                torch.zeros(1, 1, model.hidden_layer_size).double()
            )
            model = model.double()
            pred = model(input.double())
            loss = loss_function(pred, label)
            loss.backward()
            optimizer.step()

        # Every 25 epochs print the loss
        if i % 25 == 0:
            print("Epoch:", i, ", Loss:", loss.item())

    # Prediction:
    model.eval()

    for i in range(test_hours):
        input = torch.tensor(test_data[-test_hours:])
        with torch.no_grad():
            model.hidden_cell = (
                torch.zeros(1, 1, model.hidden_layer_size).double(),
                torch.zeros(1, 1, model.hidden_layer_size).double()
            )
            test_data.append(model(input.double()).item())

    norm_forecast = test_data[-test_hours:]
    forec = torch.exp(
        torch.tensor(norm_forecast)
    ) * torch.tensor(data['seasonal indices'][-test_hours:])

    if in_place:
        pass
    else:
        return forec

    # The first dimension is always the outer one. The second is the next inner
    # one, and so forth. When passing dimensions to view, the first element
    # is the outer dimension, the second is the next inner, and so on.
    # So the final view above converts the 1 dimension of train_hours data into
    # train_hours number of data along the outer dimension, each of 1 x 1 2d
    # array. This can be thought of a train_hours-depth cuboid, where the
    # width and height of the cuboid is 1 x 1


def create_pairs(train_data, window_size):
    pairs = []

    for i in range(len(train_data) - window_size):
        input = train_data[i: i + window_size]
        output = train_data[i + window_size: i + window_size + 1]
        pairs.append((input, output))

    return pairs


class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=40, output_size=1):
        super().__init__()  # Call __init__() of nn.Module()

        self.hidden_layer_size = hidden_layer_size

        # Create basic LSTM
        self.lstm = nn.LSTM(input_size, hidden_layer_size)

        # Add linear adapter layer
        self.linear = nn.Linear(hidden_layer_size, output_size)

        # Hidden cell is (a, a) where a is a 1 x 1 x h_l_s dimension tensor
        # of 1s. This variable will hold the previous hidden state and cell
        # state, which are the two internal parts of an LSTM block
        self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size).double(),
                            torch.zeros(1, 1, self.hidden_layer_size).double())

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(
            input_seq.view(len(input_seq), 1, -1),  # Required format
            self.hidden_cell
        )
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]

