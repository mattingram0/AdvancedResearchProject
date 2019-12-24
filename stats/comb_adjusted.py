def forecast(data):
    data['comb adjusted'] = (
            data['ses adjusted'] + data['holt adjusted'] + data[
                'holtDamped adjusted'
            ]
        ) / 3
