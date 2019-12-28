import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import Holt


def forecast(data, train_hours, test_hours, in_place=True):
    train = data.iloc[25:train_hours]
    model = Holt(np.asarray(train['seasonally adjusted']), damped=True)
    fit = model.fit()
    pred = fit.forecast(test_hours)

    fcst = pd.concat(
        [pd.Series([0] * 25), data['seasonal indices'][25:].multiply(
                list(fit.fittedvalues) + list(pred)
            )
         ]
    )

    if in_place:
        data['holtDamped adjusted'] = fcst
    else:
        data['holtDamped adjusted'] = fcst
        return fcst
