def forecast(data, train_days):
    data['naive2'] = data['adjusted'].shift(1)
    data['naive2'][train_days * 24:] = data['adjusted'][(train_days * 24) - 1]
