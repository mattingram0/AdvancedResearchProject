import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch

from sklearn.preprocessing import MinMaxScaler

from ml import drnn
from ml.ml_helpers import create_pairs, batch_data


# Generate a forecast
def forecast(data, train_hours, valid_hours, test_hours, window_size,
             output_size, batch_size, in_place):

    scaler = MinMaxScaler()
    scaler.fit(data[:train_hours])
    transf_data = scaler.transform(data)

    training_data, valid_data, test_data = create_pairs(
        transf_data, train_hours, valid_hours, test_hours, window_size,
        output_size, True
    )

    # Training parameters
    num_epochs = 200
    learning_rate = 0.01
    input_size = 16
    hidden_size = 40
    num_layers = 1
    output_size = 48
    dilations = [1]

    # Create model
    lstm = DRNN_48(
        output_size, input_size, batch_size, hidden_size, num_layers,
        batch_first=True, dilations=dilations
    )

    lstm = lstm.double()

    loss_func = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)

    # for n, p in lstm.named_parameters():
    #     print(n, p.shape)

    # Train model
    lstm.train()
    train_model(
        lstm, optimizer, loss_func, num_epochs, training_data, batch_size
    )

    # Make predictions
    lstm.eval()
    test_model(lstm, data, valid_data, test_data, train_hours, window_size)


# Entire training procedure
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


# Train the model on a single batch
def train_batch(lstm, optimizer, loss_func, inputs, labels):
    outputs = lstm(inputs.double())
    labels = labels.view(labels.size(0), -1)  # Remove redundant dimension

    optimizer.zero_grad()
    loss = loss_func(outputs, labels)
    loss.backward()
    optimizer.step()
    return loss


# Test the model on unseen data
def test_model(lstm, data, valid_data, test_data, train_hours, window_size):
    scaler = MinMaxScaler()
    scaler.fit(
        data["total load actual"][:train_hours].to_numpy().reshape(-1, 1)
    )

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
    plt.gca().set_title('Dilations: [1]')
    plt.show()


# Stateful, mini-batch trained DRNN. Multiple features. 48 hour forecast
class DRNN_48(nn.Module):
    def __init__(self, output_size, input_size, batch_size, hidden_size,
                 num_layers, dropout=0, cell_type='LSTM', batch_first=False,
                 dilations=None):

        super().__init__()

        self.output_size = output_size
        self.input_size = input_size
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cell_type = cell_type

        # Allows the LSTM to be stateful between mini-batches
        # moment as I'm not using stateful LSTM. init_hidden_states needs
        # changing too
        if self.cell_type == "LSTM":
            self.hidden = (torch.zeros(num_layers, batch_size, hidden_size),
                           torch.zeros(num_layers, batch_size, hidden_size))
        else:
            self.hidden = torch.zeros(num_layers, batch_size, hidden_size)

        self.drnn = drnn.DRNN(
            input_size,
            hidden_size,
            num_layers,
            dropout=dropout,
            cell_type=cell_type,
            batch_first=batch_first,
            dilations=dilations
        )

        self.linear = nn.Linear(hidden_size, output_size)

    # Call once at the beginning of every epoch
    def init_hidden_states(self):
        self.hidden = (
            torch.zeros(self.num_layers, self.batch_size, self.hidden_size),
            torch.zeros(self.num_layers, self.batch_size, self.hidden_size)
        )

    def forward(self, x, hidden=None):
        if hidden is None:
            lstm_out, hidden = self.drnn(x.double())
            h_out = hidden[0][-1]
            self.hidden = hidden

        else:
            lstm_out, h_out = self.drnn(x.double(), hidden)
            self.hidden = h_out

        linear_in = h_out[:, -x.size(0):, :].view(-1, self.hidden_size)
        out = self.linear(linear_in)

        return out
