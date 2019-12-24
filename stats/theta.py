import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
import statsmodels.api as sm


def forecast(data, train_hours, test_hours):
    theta0 = 0
    theta2 = 2

    def b_theta(theta, series):
        n = len(series)
        const = (6 * (1 - theta)) / ((n ** 2) - 1)
        tmp1 = 2 * np.array([(t + 1) * x_t for t, x_t in enumerate(
            series)]).mean()
        tmp2 = (n + 1) * series.mean()

        return const * (tmp1 - tmp2)

    def a_theta(theta, series):
        term1 = (1 - theta) * series.mean()
        term2 = b_theta(theta, series) * (len(series) - 1) / 2

        return term1 - term2

    a_theta0 = a_theta(theta0, data['seasonally adjusted'][:train_hours])
    a_theta2 = a_theta(theta2, data['seasonally adjusted'][:train_hours])
    b_theta0 = b_theta(theta0, data['seasonally adjusted'][:train_hours])
    b_theta2 = b_theta(theta2, data['seasonally adjusted'][:train_hours])

    data['theta0'] = 0
    data['theta2'] = 0

    # Calculate theta0 and theta2 lines for the train data
    data['theta0'][:train_hours] = [
        a_theta0 + (b_theta0 * t) for t in range(train_hours)
    ]
    data['theta2'][:train_hours] = [
        a_theta2 + (b_theta2 * t) + theta2 * data['seasonally adjusted'][t + 1]
        for t in range(train_hours)
    ]

    data['theta0'][train_hours:] = [
        a_theta0 + b_theta0 * (train_hours + h - 1) for h in range(test_hours)
    ]

    # Predict theta0 and theta2 lines for the test data
    model = SimpleExpSmoothing(
        np.asarray(data['theta2'][:train_hours])
    )
    fit = model.fit()
    pred = fit.forecast(test_hours)
    data['theta2'][train_hours:] = list(pred)
    alpha = int(float(str(fit.params['smoothing_level'])[:3]))

    # Take the arithmetic average of the two theta lines to get the final
    # forecast
    data['theta'] = (
        (data['theta0'] + data['theta2']) / 2
    ) * data['seasonal indices']