import numpy as np
import pandas as pd
from stats import helpers


def forecast(data, train_hours):
    data['naive2 adjusted'] = (data['seasonally adjusted'].shift(1)) * data[
        'seasonal indices']
    data['naive2 adjusted'][train_hours:] = data['naive2 adjusted'][
        train_hours - 1]