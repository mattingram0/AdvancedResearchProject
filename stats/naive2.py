import numpy as np


def forecast(data, train_hours):
    data['naive2'] = data['seasonally differenced'].shift(1)
    data['naive2'][train_hours:] = data['seasonally differenced'][train_hours -
                                                                  1]


def undifference(data, train_hours, test_hours):
    start_value = [data['total load actual'][train_hours] for i in range(
        test_hours)]
    cum_forecast = np.cumsum(data['naive2'][train_hours:])
    data['naive2 undiff'] = 0
    data['naive2 undiff'][train_hours:] = start_value + cum_forecast



    # print("UNDIFFERENCE")
    # a = data['total load actual'].iloc[train_hours - 1]
    # b = data['total load actual'].iloc[train_hours - 24]
    # c = data['total load actual'].iloc[train_days * 25 - 25]
    #
    # print("Prev Value", a)
    # print("Prev Season", b)
    # print("Prev Season Value", c)
    # print("Start Value: ", a - c + b)
    #
    # data['naive2 undiff'] = 0
    # for i, v in enumerate(np.cumsum(data['naive2'][train_hours:])):
    #     data['naive2 undiff'].iloc[train_hours + i] = v + a - c + b


