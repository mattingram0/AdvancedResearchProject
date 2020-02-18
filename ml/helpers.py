import numpy as np
import torch


def create_pairs(data, train_hours, valid_hours, test_hours,
                 window_size, output_size, multiple):
    # Split up the data
    train_data = data[:train_hours]
    valid_data = data[train_hours - window_size:
                             train_hours + valid_hours]
    test_data = data[train_hours + valid_hours - window_size:
                            train_hours + valid_hours + test_hours]

    train_data = sliding_window(
        train_data, output_size, window_size, "train", multiple
    )

    valid_data = sliding_window(
        valid_data, output_size, window_size, "valid", multiple
    )

    test_data = sliding_window(
        test_data, output_size, window_size, "test", multiple
    )

    return train_data, valid_data, test_data


def sliding_window(train_data, output_size, window_size, section, multiple):
    inputs = []
    labels = []
    step = 1 if section == "train" else output_size

    for i in range(0, len(train_data) - window_size - output_size + 1, step):
        x = train_data[i: i + window_size]

        if multiple:
            # Use only the total load actual column (column 14) for the label
            y = train_data[i + window_size: i + window_size + output_size, 14]
        else:
            y = train_data[i + window_size: i + window_size + output_size]

        inputs.append(x)
        labels.append(y)

    inputs = torch.tensor(np.array(inputs), dtype=torch.double)
    labels = torch.tensor(np.array(labels), dtype=torch.double)

    return inputs, labels


def batch_data(training_data, batch_size):
    input_batches = []
    label_batches = []

    for i in range(0, len(training_data[0]), batch_size):
        input_batches.append(training_data[0][i:i + batch_size])
        label_batches.append(training_data[1][i:i + batch_size])

    return input_batches, label_batches
