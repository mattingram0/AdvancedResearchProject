def forecast(data, train_days):
    data['naive1'] = data['total load actual'].shift(1)
    data['naive1'][train_days * 24:] = data['total load actual'][(train_days
                                                                 * 24) - 1]
