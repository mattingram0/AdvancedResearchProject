def forecast(data):
    data['naive1'] = data['total load actual'].shift(1)
    data['naive1'][144:] = data['total load actual'][143]
