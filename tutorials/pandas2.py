import pandas as pd
import numpy as np

# An Index is an immutable ndarry implementing an orderered sliceable set.
# The basic object for storing axis labels for all pandas objects
index = pd.date_range('1/1/2000', periods=8)

# A series is a 1D ndarray with axis labels. The labels need not be unique
# but must be hashable. The objext supports both integer and label indexing.
# 100 x 1 series of random numbers
long_series = pd.Series(np.random.randn(100))
ls = long_series.array  # Get the data in a list-like array

# A Dataframe is a 2D, size-mutable, potentially heterogeneous tabular data
# structure with labeled axes (rows and columns). Can be though of as a
# dict-like container for Series objects
# 5 x 5 table of random numbers
small_df = pd.DataFrame(np.random.randn(5, 5))

# TYPICALLY ALL PANDAS FUNCTIONS RETURN A NEW DATAFRAME, RATHER THAN APPLYING
# THE FUCNTIONARLITY IN SITU - INPLACE TYPICALLY FORCES THE ACTION TO HAPPEN
# TO THE DATAFRAME ITSELF

print(pd.MultiIndex.from_tuples([(1, 'a'), (1, 'b'), (1, 'c'), (2, 'a')],
                                names=['1st', '2nd']))

small_df.mean(0)  # Will calculate the mean of all the columns
small_df.mean(1)  #       "           "          "     rows
small_df.idxmin(axis=0, skipna=True)  # Calculates index of smallest value
# in each column, skipping NaNs

rn = small_df.rename(columns={0: 'Zero', 1: 'One'})  # Rename two of the
# columns
print(rn.drop(["Zero"], axis=1))  # Drop the zero'th column
