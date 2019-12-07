import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import SimpleExpSmoothing


def forecast(data, train_hours, test_hours):
    train = data.iloc[25:train_hours]
    model = SimpleExpSmoothing(np.asarray(train['seasonally adjusted']))
    fit = model.fit()
    pred = fit.forecast(test_hours)

    data['ses adjusted'] = 0
    data['ses adjusted'][25:] = data['seasonal indices'][25:].multiply(
        list(fit.fittedvalues) + list(pred)
    )
