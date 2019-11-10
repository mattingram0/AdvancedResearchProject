import pandas as pd

# Dataframes are the key construct in Pandas - used to represent data with
# # rows and columns - it is a tabular data structure
df = pd.read_csv("/Users/matt/Projects/AdvancedResearchProject/data/spain"
                 "/energy_dataset.csv")
#
# Pandas supports: csv, excel, python dictionary of columns, python list of
# tuples (one tuple per row - col names not included), python list of
# dictionaries (one dict per row), among others

# Columns are accessed using standard dictionary indexing
print(df['price actual'].min())

# Return the time column of every entry where the forecasted price was over 99
print(df['time'][df['price day ahead']>=100])

# Calculate the average price
print(df['price actual'].mean())

# Fill any Not a Number values with 0
df.fillna(0)

# Create a dataframe from a dictionary
weather_data = {
    'day': ['1/1/2017', '1/2/2017', '3/4/2017'],
    'temperature': [32, 35, 64]
}
df2 = pd.DataFrame(weather_data)
print(df2)

# Shape = (rows, columns)
rows, cols = df2.shape
print(rows, cols)

# Print only the top n rows (ditto for tail)
print(df.head(3))

# Splicing works also
print(df[3:5])

# List of fields (column titles). Columns are of type series.
print(df.columns)

# Print the first 5 time values
print(df['time'].head(5))  # df.time syntax works also

# Print multiple columns
print(df[['time', 'total load actual']].head(3))  # Note the double brackets

# Print statistics
print(df.describe())

# Find the date/time (and price) of the row with the maximum price
print(df[['time', 'price actual']][df['price actual']==df['price '
                                                         'actual'].max()])

# By default, df's are indexed by the row number (an integer). We can change
# this
df.set_index('time', inplace=True)  # W/out inplace, returns new data frame
print(df.head(3))  # We can now index by the date/time, rather than a number

# Reset index
df.reset_index(inplace=True)

# Handling missing data
# fillna, interpolate, and dropna (among others)
df = pd.read_csv("/Users/matt/Projects/AdvancedResearchProject/data/spain"
                 "/energy_dataset.csv", parse_dates=["time"])  # Parse dates
print(type(df['time'][0]))  # We see we have a datetime.datetime value
# df.set_index('time', inplace=True)  # Index by the date/time

# Check to see if we have any missing data
print(df.isnull().values.any())

# Fill Na values with 0
new_df = df.fillna(0)

# Fill Na values with a dictionary of default values
new_df = df.fillna({
    'generation fossil gas': 0,
    'price day ahead': 0.0
})

# Fill Na values with the previous value. index specifies horiz/vert fill.
new_df = df.fillna(method="ffill")  # ffill = fill-forward. Ditto for bfill

# Fill Na values with (linearly by default) interpolated values. Lots of
# different interpolation methods (quadratic, etc)
new_df = df.interpolate()
print(df.shape)

# Drop the na rows with less than two valid values (not including the index)
new_df = df.dropna(threshold=2)
print(new_df.shape)

# Shifting (useful for difference/calculating percentage change)
new_df = df.shift(1)  # Shift 1 forward
df['Previous Day Price'] = df['price actual'].shift(1)  # Calc prev day price
df['1 day change'] = df['price actual'] - df['Previous Day Price']

# We can create custom data indexes (i.e business day index) for stock price
# data - this will be useful in the future. If we have time indexed data,
# we can shift the date
