import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.style
matplotlib.style.use('seaborn-pastel')

# Matplotlib is the toolkit, PyPlot is an interactive way to use Matplotlib,
# PyLab is the same as PyPlot but with some extra shortcuts (discouraged now)

# PyPlot is a shell-like interface to Matplotlib, to make it easier to use
# for people who are used ot MATLAB.
# PyPlot maintains state across calls.

# Figure: The object that keeps the whole image output
# Axes: Represents a pair of axis the contains a single plot (x, y axis)

df = pd.DataFrame({
    'name': ['john', 'mary', 'peter', 'jeff', 'bill', 'lisa', 'jose'],
    'age': [23, 78, 22, 19, 45, 33, 20],
    'gender': ['M', 'F', 'M', 'M', 'M', 'F', 'M'],
    'state': ['california', 'dc', 'california', 'dc', 'california', 'texas',
              'texas'],
    'num_children': [2, 0, 0, 3, 2, 1, 4],
    'num_pets': [5, 1, 0, 5, 2, 2, 3]
})

# Pandas has tight integration with matplotlib, we can plot directly
# using the plot() method, and refernce fields in the pandas dataframe

df.plot(kind='scatter',x='num_children',y='num_pets',color='red')
plt.clf()  # Clears the entire current figure

# Line plot with multiple datapoints
fig = plt.gcf()  # Get current figure (demonstrative here)
ax = plt.gca()  # Get current axes

# Plotting one column versus the other using the x, y keywords
df.plot(kind='line',x='name',y='num_children',ax=ax)
df.plot(kind='line',x='name',y='num_pets', ax=ax)
plt.clf()

# Convenience method that plots all columns against the index
# column, with labels.
df.plot()

# Show the plot -
plt.show()

# Note - when plotting using Pandas and Matplotlib, keep the data and the
# plotting separate - use pandas for the data, and matplotlib for the plotting

