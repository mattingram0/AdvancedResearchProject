def forecast(data):
    data['forecast'] = data['total load actual'].shift(1)
    return data['forecast']
