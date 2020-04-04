def forecast(data, train_hours, test_hours, in_place=True):
    fcst = (data['ses adjusted'] + data['holt adjusted'] + data[
                'holtDamped adjusted']) / 3

    if in_place:
        data['comb adjusted'] = fcst
    else:
        return fcst
