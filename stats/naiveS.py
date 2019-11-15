def forecast(data):
    data['naiveS'] = data['total load actual'].shift(24)
