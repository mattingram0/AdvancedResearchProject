import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler


# Here, train_hours is the window size, and test_hours is the size of the
# data to be reserved for testing
def forecast(data, train_hours, valid_hours, test_hours, window_size,
             in_place):
    # Scale the training data into the range (0, 1), and generate training set
    scaler = MinMaxScaler()
    transf_data = scaler.fit_transform(data)

    # Split up the data
    train_data = transf_data[:train_hours]
    valid_data = transf_data[train_hours - window_size: train_hours + valid_hours]
    test_data = transf_data[
        train_hours + valid_hours - window_size: train_hours + valid_hours +
                                          test_hours
    ]

    # Create the inputs and labels, and convert to tensors
    train_inputs, train_labels = (
        torch.tensor(i,  dtype=torch.double)
        for i in create_training_set(train_data, window_size)
    )

    valid_inputs, valid_labels = (
        torch.tensor(i, dtype=torch.double)
        for i in create_training_set(valid_data, window_size)
    )

    test_inputs, test_labels = (
        torch.tensor(i, dtype=torch.double)
        for i in create_training_set(test_data, window_size)
    )

    all_inputs = torch.cat((train_inputs, valid_inputs, test_inputs))
    all_labels = torch.cat((train_labels, valid_labels, test_labels))


    # Here we have a single batch of training examples. Our batch size is
    # 96, as our single batch has 96 elements in it. Each of the elements is
    # of length 48, which is our sequence length and corresponds to the
    # window size used. Finally, as we only have one feature for every
    # sample, the final dimension size is 1.
    # Remember that we read the dimensions in a torch.size value from left
    # to right, and these correspond from outer to inner dimension.
    # Therefore, our data is in the correct format because we are using
    # batch_first=True, and so our input needs to be in the shape:
    # (batch_size, sequence_length, num_features).
    # AFAIK nothing changes about the PyTorch functionality (i.e it doesn't
    # perform batching first), but rather this parameter is merely for
    # convenience as data often comes with the batch_size dimension first)

    # Training parameters
    num_epochs = 1000
    learning_rate = 0.01  # Static learning rate
    input_size = 1  # Our only feature is the time series itself (possibly
    # include weather information here?)
    hidden_size = 2  # Only two nodes in the hidden layer
    num_layers = 1  # Just one layer of LSTMs
    output_size = 1  # One step prediction

    # Create model
    lstm = LSTM(output_size, input_size, hidden_size, num_layers).double()
    loss_func = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)

    # Here we are performing ADAM batch gradient descent. That is, we feed in
    # one batch of data containing our entire dataset, and then update the
    # parameters after this. In the LSTM case, we therefore have that the
    # number of "time steps" = batch_size = training_size = 96 hours here.
    # We then perform 2000 repetitions (epochs), hoping that the loss converges
    for epoch in range(num_epochs):

        # Pass in (96, 24, 1) - entire batch of 96 elements each of which is
        # 24 values long with only 1 feature, and get out (96, 1) - entire
        # batch of outputs which is a single prediction for each of the 96
        # inputs
        outputs = lstm(train_inputs.double())

        # Zero the gradients of the optimizer - what happens if we don't do
        # this, do they not get reset? Might be useful with Slawek's model
        optimizer.zero_grad()

        # Pass the predicted outputs (96, 1) and the labels (96, 1) to the
        # function, and calculate the mean squared error
        loss = loss_func(outputs, train_labels)

        # Backprop the gradients
        loss.backward()

        # Change weights in the correct direction
        optimizer.step()

        if epoch % 100 == 0:
            print("Epoch %d: Loss - %1.5f" % (epoch, loss.item()))

    lstm.eval()
    norm_prediction = lstm(all_inputs).data.numpy()
    norm_actual = all_labels.data.numpy()
    prediction = scaler.inverse_transform(norm_prediction)
    actual = scaler.inverse_transform(norm_actual)

    plt.figure(figsize=(12.8, 9.6), dpi=250)
    plt.plot(prediction, label="Prediction")
    plt.plot(actual, label="Actual Data")
    plt.axvline(x=len(train_labels), c='orange', label="Training Data")
    plt.axvline(x=len(train_labels) + len(valid_labels), c='purple',
                label="Validation Data")
    plt.gca().set_xlabel("Time")
    plt.gca().set_ylabel("Total Energy Demand")
    plt.gca().legend(loc="best")
    plt.show()


def create_training_set(train_data, window_size):
    inputs = []
    labels = []

    # Sliding window
    # Inputs = [[ <- window_size -> ], ...]
    # Labels = [ , , , ... ]
    for i in range(len(train_data) - window_size):
        x = train_data[i: i + window_size]
        y = train_data[i + window_size]
        inputs.append(x)
        labels.append(y)

    return np.array(inputs), np.array(labels)


class LSTM(nn.Module):
    def __init__(self, output_size, input_size, hidden_size, num_layers):
        super().__init__()

        self.output_size = output_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Create the LSTM model, using the built in LSTM class. By
        # specifying batch_first to be True, we must supply input tensors as
        # (batch_size, seq_length, num_features), instead of the usual (
        # seq_length, batch_size, input_size (aka num_features))
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        # We then simply add a linear layer which is effectively just a map
        # from a (hidden_size x 1) row to a (1 x output_size) column. In matrix
        # terminology. A Linear(in, out) will simply map an (n, *, in) tensor
        # into a (n, *, out) tensor

        # In general, a linear layer is just maps an (h * n) matrix to a (n
        # * o) matrix, where n is fixed and h and o and chosen.
        # Say currently we have (4 x 2). Then we have a (2 x 3) matrix. We
        # then get a (4 x 3) output. The number of columns of the first much
        # match the number of rows of the second

        # a b       i j k    1 2 3
        # c d   x   l m n =  5 6 7
        # e f                8 9 10
        # g h                11 12 13
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initial values for the hidden layers
        # For a uni-direction LSTM, these are of size:
        # (num_layers, batch_size, hidden_size).
        # As our input x is of size (batch_size, seq_length, num_features),
        # x.size(0) will get the batch_size for us, and so can be varied once
        # the model has been built as a hyperparameter, without the need to
        # create a new model for each
        init_h = torch.zeros(self.num_layers, x.size(0),
                             self.hidden_size).double()
        init_c = torch.zeros(self.num_layers, x.size(0),
                             self.hidden_size).double()

        # Feed the input into the network. This will feed all of the input
        # elements of our batch into the network sequentially. Each of the
        # elements is a sequence of 48 values, represented by a (48 * 1) vector
        # as the data itself is the only 'feature' of the LSTM
        lstm_out, (final_h, final_c) = self.lstm(x.double(), (init_h, init_c))

        # The linear layer expects an input of shape (N, *, hidden_size)
        # and outputs a tensor of shape (N, *, output_size), so we convert out
        # lstm_out variable which has shape (for unidirectional LSTM)
        # (num_layers, batch_size, hidden_size)
        # = (1, 96, 2)
        # to a tensor of shape (96, 2)
        linear_in = final_h.view(-1, self.hidden_size)

        # We then pass the lstm_out into the linear layer, to get a final
        # output of (96, 1)
        return self.linear(linear_in)




# In LSTMs, we typically have:
# An series of input sequences:
# "The cow jumped", "over the fox"
# This is input as:
# [["The", "over"], ["cow", "the"], ["jumped", "fox"]]

# Divide your dataset into batches, and each elemenet of the batch contains a
# seq_length of windows of samples


# input_size is the number of expected features in the input x. For us,
# we just have one feature:

# [[1],
#  [2],
#  ...
#  [n]]

# When we have more than one time series, the input size will increase. We
# now have two features
# [[1, a],
#  [2, b],
#  ...
#  [n, p]]

# The hidden feature size can be of any shape we want.