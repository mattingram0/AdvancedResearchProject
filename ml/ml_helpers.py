import sys
from functools import reduce
from math import exp, fabs

import numpy as np
import torch


# Convert hPa to Pa
def hpa_to_pa(p):
    return 100 * p


# Convert kelvin to celcius
def k_to_c(t):
    return t - 273.15


# Convert celcius to kelvin
def c_to_k(t):
    return t + 273.15


# Calculate an approximation for the saturation vapor pressure for moist air
# using the Arden Buck equation (more optimised than the Goff-Gratch
# equation for -80C to 50C).
# Input: Temperature t in Celcius (C)
# Output: Saturation vapor pressire in Pascals (Pa)
# See: https://en.wikipedia.org/wiki/Arden_Buck_equation
def arden_buck(t):
    return 611.21 * exp((18.678 - t/234.5) * (t/(257.14 + t))) if t > 0 else\
        611.15 * exp((23.036 - t/333.7) * (t/(279.82 + t)))


# Calculate an approximation for the specific enthalpy using the above Arden
# Buck equation for the saturation vapor pressure for moist air. The formula is:
# h = Cpa * t + x * [Cpw * t + Hwe], where Cpa, Cpw and Hwe are constants, and
# x = 0.62198 * pw(t) / (pa - pw(t)) is the humidity ratio per mass, where:
# pa is the current pressure in Pascals, and
# pw(t) = r * ps(t) is the partial pressure of water vapor in moist air, where:
# r = relative humidity of the air (decimal), and
# ps(t) = saturation vapor pressure for the given temperature t, found using
# the arden_buck formula.
# See:
# https://www.engineeringtoolbox.com/enthalpy-moist-air-d_683.html
# https://www.engineeringtoolbox.com/water-vapor-saturation-pressure-air-d_689.html
# https://www.chegg.com/homework-help/determine-partial-pressure-water-vapor-moist-air-following-c-chapter-10-problem-3p-solution-9781305534094-exc
# Note that the formula given in this paper: https://www.sciencedirect.com/science/article/pii/S0378778815303492#eq0005
# is incorrect
# Used: https://www.psychrometric-calculator.com/humidairweb.aspx to check
# my implementation
# Input: Temperature in C, pressure in Pa, relative humidity ([0, 1])
# Output: Specific enthalpy in kilojoules per kilogram of dry hair (kJ/kg)
def specific_enthalpy(t, p, r):
    return 1.006 * t + (((0.62198 * r * arden_buck(t)) / (p - (r * arden_buck(
        t)))) * (2501 + (1.84 * t)))


# Receive a row of weather data (must include temperature, pressure and
# relative humidity) and calculate the latent enthalpy as defined in:
# https://ieeexplore.ieee.org/document/1525139
def latent_enthalpy(row, ref_temp):
    t = k_to_c(row["temp"])
    p = hpa_to_pa(row["pressure"])
    r = row["humidity"] / 100
    q = specific_enthalpy(t, p, r)
    qb = specific_enthalpy(ref_temp, p, r)

    return q - qb if t > ref_temp and q - qb > 0 else 0


# Create input, target pairs for the basic ML models
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


# Use sliding window approach to generate input, target pairs
def sliding_window(train_data, output_size, window_size, section, multiple):
    inputs = []
    labels = []
    step = 1 if section == "train" else output_size

    for i in range(0, len(train_data) - window_size - output_size + 1, step):
        x = train_data[i: i + window_size]

        if multiple:
            # Use only the total load actual column (column 14) for the label
            # y = train_data[i + window_size: i + window_size + output_size,
        # 14]
            # TODO - CHANGE BACK when fixed the zero values
            y = train_data[i + window_size: i + window_size + output_size, 1]
        else:
            y = train_data[i + window_size: i + window_size + output_size]

        inputs.append(x)
        labels.append(y)

    inputs = torch.tensor(np.array(inputs), dtype=torch.double)
    labels = torch.tensor(np.array(labels), dtype=torch.double)

    return inputs, labels


# Turn entire sequence of training data into batches
def batch_data(training_data, batch_size):
    input_batches = []
    label_batches = []

    for i in range(0, len(training_data[0]), batch_size):
        input_batches.append(training_data[0][i:i + batch_size])
        label_batches.append(training_data[1][i:i + batch_size])

    return input_batches, label_batches


# Calculate the pinball loss of a batch of training examples
def pinball_loss(pred, actual, tau):
    return torch.mean(torch.where(
        actual >= pred,
        (actual - pred) * tau,
        (pred - actual) * (1 - tau)
    ))


# Find optimal batch size for given number of data points
def calc_batch_size(n, bs):
    factors = list(set(
        reduce(
            list.__add__,
            ([i, n // i] for i in range(1, int(n ** 0.5) + 1) if n % i == 0)
        )
    ))
    return factors[np.argmin([fabs(v - bs) if v >= bs else sys.maxsize for v
                              in factors])]