def forecast(data, train_hours, test_hours):
    pred = list(
        data['total load actual'][train_hours - 24:train_hours]
    ) * int(test_hours / 24)

    data['naiveS'] = data['total load actual'].shift(24)
    data['naiveS'][train_hours:] = pred
