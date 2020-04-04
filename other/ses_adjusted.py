from statsmodels.tsa.holtwinters import SimpleExpSmoothing


# Pass in seasonally adjusted data if required, then reseasonalise after.
# Train_index needs to be the index of the very final training_value.
def forecast(data, train_index, forecast_length):
    fitted_model = SimpleExpSmoothing(data).fit()
    return fitted_model.predict(0, train_index + forecast_length)


# Concatenating the fitted_values of the model with the forecast_length
# forecasted_values  gives the exact same results as just predicting from 0
# to the train_index + forecast length
