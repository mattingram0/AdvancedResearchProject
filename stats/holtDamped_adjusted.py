import numpy as np
from statsmodels.tsa.holtwinters import Holt


def forecast(data, train_hours, test_hours):
    train = data.iloc[25:train_hours]
    model = Holt(np.asarray(train['seasonally adjusted']), damped=True)
    fit = model.fit()
    pred = fit.forecast(test_hours)

    data['holtDamped adjusted'] = 0
    data['holtDamped adjusted'][25:] = data['seasonal indices'][25:].multiply(
        list(fit.fittedvalues) + list(pred)
    )
