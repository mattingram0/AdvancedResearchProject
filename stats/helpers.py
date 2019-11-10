def sma(data, window):
    return data['total load actual'].rolling(window=window).mean()

