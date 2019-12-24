def forecast(data, train_hours, test_hours, in_place=True):
    fcst = (data['seasonally adjusted'].shift(1)) * data['seasonal indices']
    fcst[train_hours:] = data['naive2 adjusted'][train_hours - 1]

    if in_place:
        data['naive2 adjusted'] = fcst
    else:
        return fcst
