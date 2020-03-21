import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import SimpleExpSmoothing


def forecast(data, train_index, forecast_length):
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

    a_theta0 = a_theta(theta0, data[:train_index + 1])
    a_theta2 = a_theta(theta2, data[:train_index + 1])
    b_theta0 = b_theta(theta0, data[:train_index + 1])
    b_theta2 = b_theta(theta2, data[:train_index + 1])

    # Calculate theta0 and theta2 lines for the train data
    theta0_fitted = pd.Series(
        [pd.a_theta0 + (b_theta0 * t) for t in range(train_index + 1)]
    )
    theta2_fitted = pd.Series(
        [a_theta2 + (b_theta2 * t) + theta2 * data[t + 1]
         for t in range(train_index + 1)]
    )

    # Predict theta0 and theta2 lines for the test data
    theta0_predicted = pd.Series(
        [a_theta0 + b_theta0 * (train_index + h)
         for h in range(forecast_length)]
    )

    # Predict theta0 and theta2 lines for the test data
    theta2_model_fitted = SimpleExpSmoothing(theta2_fitted).fit()
    theta2_predicted = theta2_model_fitted.forecast(forecast_length)

    theta0 = theta0_fitted.append(theta0_predicted).reset_index(drop=True)
    theta2 = theta2_fitted.append(theta2_predicted).reset_index(drop=True)

    # Take the arithmetic average of the two theta lines to get the forecast
    return (theta0 + theta2) / 2
