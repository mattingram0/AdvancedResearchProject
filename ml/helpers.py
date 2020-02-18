import numpy as np
import torch


def create_pairs(data, train_hours, valid_hours, test_hours,
                 window_size, output_size):
    # Split up the data
    train_data = data[:train_hours]
    valid_data = data[train_hours - window_size:
                             train_hours + valid_hours]
    test_data = data[train_hours + valid_hours - window_size:
                            train_hours + valid_hours + test_hours]

    # Create the inputs and labels, and convert to tensors
    training_data = list(
        torch.tensor(i, dtype=torch.double)
        for i in sliding_window(train_data, output_size, window_size, "train")
    )

    valid_data = list(
        torch.tensor(i, dtype=torch.double)
        for i in sliding_window(valid_data, output_size, window_size, "valid")
    )

    test_data = list(
        torch.tensor(i, dtype=torch.double)
        for i in sliding_window(test_data, output_size, window_size, "test")
    )

    return training_data, valid_data, test_data


def sliding_window(train_data, output_size, window_size, section):
    inputs = []
    labels = []
    step = 1 if section == "train" else output_size

    for i in range(0, len(train_data) - window_size - output_size + 1, step):
        x = train_data[i: i + window_size]
        y = train_data[i + window_size: i + window_size + output_size]
        inputs.append(x)
        labels.append(y)

    return np.array(inputs), np.array(labels)


def batch_data(training_data, batch_size):
    input_batches = []
    label_batches = []

    for i in range(0, len(training_data[0]), batch_size):
        input_batches.append(training_data[0][i:i + batch_size])
        label_batches.append(training_data[1][i:i + batch_size])

    return input_batches, label_batches