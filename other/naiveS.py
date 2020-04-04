def forecast(data, train_hours, test_hours, in_place=True):
    pred = list(
        data['total load actual'][train_hours - 24:train_hours]
    ) * int(test_hours / 24)

    fcst = data['total load actual'].shift(24)
    fcst[train_hours:] = pred

    if in_place:
        data['naiveS'] = fcst
    else:
        return fcst
