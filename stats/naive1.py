def forecast(data):
    data['naive1'] = data['total load actual'].shift(1)
