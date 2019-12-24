def forecast(data, train_hours, test_hours, in_place=True):
    fcst = data['total load actual'].shift(1)
    fcst[train_hours:] = data['total load actual'][train_hours - 1]

    if in_place:
        data['naive1'] = fcst
    else:
        return fcst
