def forecast(data, train_hours):
    data['naive1'] = data['total load actual'].shift(1)
    data['naive1'][train_hours:] = data['total load actual'][train_hours - 1]
