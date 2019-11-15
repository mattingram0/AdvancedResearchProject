def forecast(data):
    data['naive2'] = data['adjusted'].shift(1)
    data['naive2'][144:] = data['adjusted'][143]
