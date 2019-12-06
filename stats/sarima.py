import statsmodels.api as sm


def forecast(data, train_hours, test_hours):
    model2 = sm.tsa.statespace.SARIMAX(
        data['total load actual'][:(train_hours)],
        order=(0, 1, 1),
        seasonal_order=(0, 1, 1, 24)
    )
    res = model2.fit(disp=0)
    data['sarima'] = 0
    data['sarima'][train_hours:] = res.forecast(steps=test_hours)

    print(res.summary())
