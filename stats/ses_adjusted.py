import numpy as np
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
import pandas as pd


def forecast(data, train_hours, test_hours, in_place=True):
    train = data.iloc[25:train_hours]
    model = SimpleExpSmoothing(np.asarray(train['seasonally adjusted']))
    fit = model.fit()
    pred = fit.forecast(test_hours)

    fcst = pd.concat(
        [pd.Series([0] * 25), data['seasonal indices'][25:].multiply(
            list(fit.fittedvalues) + list(pred)
        )
         ]
    )

    if in_place:
        data['ses adjusted'] = fcst
    else:
        data['ses adjusted'] = fcst
        return fcst
