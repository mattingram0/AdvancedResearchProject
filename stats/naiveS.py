def forecast(data):
    return data['total load actual'].shift(24)
