import numpy as np

# Information gleaned from:
# https://docs.scipy.org/doc/numpy/user/quickstart.html

# Dimensions are called axes. [1, 2, 1] has 1 axis, of 3 elements
# [[1, 0, 1], [2, 0, 3]] has 2 axes, the first of length 2, the second of
# length 3

# Fundamental data type - the homogeneous ndarray, also known just as an array

a = np.arange(15).reshape(3, 5)  # 3 rows, 5 columns
print(a.ndim)  # Number of axes
print(a.shape)  # Dimensions (n rows, m cols)
print(a.size)  # Total number of elements (n * m)
print(a.dtype)  # Data type
print(a.itemsize)  # Size of each of the elements
print(a.data)  # Buffer containins actual elements (don't use)

# Creating arrays
a = np.array([1, 2, 3, 4])  # Must use brackets
b = np.array([[1, 2], [3, 4]], dtype=complex)  # Specify datatype
c = np.zeros((4, 5))  # 4 x 5 array of 0s
d = np.arange(10, 30, 5)  # Range of numbers from 10 (inc) to 30 (exc), step 5
e = np.linspace(0, 2, 9)  # Final arg is how many numbers we want
f = np.arange(24).reshape(2, 3, 4)  # depth 2 (2 copies of), 3 rows, 4 columns

# Basic Operations
# Arithmetic operations are applied elementwise. A NEW ARRAY is created and
# filled with the result
g = 10 * np.sin(a)  # Apply 10 * sin(element) for all elements
h = f @ g  # Matrix multiplication
i = f.dot(g)  # Matrix multiplication
j = i.sum()  # Sums all elements
k = j.min(axis=0)  # Min of each column

# Splicing
a[:6:2] = 16  # From the start to position6, set every 2nd element = 16
l = k[1:3, 2:5]  # 2nd and 3rd row, 3rd to 5th column
# If fewer indices are provided, missing ones are considered " : "
# Equivalently, x[1, 2, ...] is equivalent to x[1, 2, :, :, :]

# Accessing
print(a[2, 3])

# Reshaping
print(list(l.flat))  # l.flat is an iterator of all elements
l.ravel()  # Returns the array, flattened
# reshape returns its argument with a modified shape, resize modifies the
# array itself.
# If a dimension is given as -1 in a reshaping operation, other dimensions
# are automatically calculated

# Copying/Views
# No copy:
l = np.arange(10)
m = l
print(b is a)  # True - refer to same object. Function calls also make no copy

# View/Shallow Copy
n = m.view()
print(n is m)  # False
print(n.base is m)  # True
# Changing the shape of a view doesn't modify the underlyin date,
# but changing the data itself will. Slicing returns a view of data

# Deep Copy
o = n.copy()  # Complete copy of the array and its data

# Didn't cover fancy indexing - see tutorials if necessary