import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import linalg  # SciPy sub-packages need to be imported separately

# Use the above import convections
# Help
np.info(linalg)

#SciPy builds on NumPy, so for all basic array handling needs NumPy
# functions are used. Top level of scipy contains some numpy functions,
# but its better to use them directly from the numpy module instead

# Some abuse of the slicing notation in numpy/scipy:
# [0:5:4j] means use 4 as the number of values, rather than the step size
# and can often be used as input to functions.

# The vectorise class can be used to return vectorised equivalents of scalar
# functions:


def addsub(a, b):
    if a > b:
        return a - b
    else:
        return a + b


vec_addsub = np.vectorize(addsub)
print(vec_addsub([0, 3, 4, 5], [3, 7, 1, 9]))  # We can now pass arrays

# Select - vectorized form of the multiple if-statement
x = np.arange(10)
condlist = [x < 3, x > 5]  # The conditions for which to keep values
choicelist = [x, x**2]  # The list (of list) of values for which to look
# through
print(np.select(condlist, choicelist))  # Perform the multi-d if statement

# scipy.special contians the definition of numerous special function of
# mathematical physics
