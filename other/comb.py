def forecast(data):
    data['comb undiff'] = (data['ses undiff'] + data['holt undiff'] + data[
        'holtDamped undiff']) / 3
