import statsmodels.api as sm
import pmdarima as pm
import numpy as np
import pandas as pd


def arima(data, train_index, forecast_length, order):
    fitted_model = sm.tsa.ARIMA(data[:train_index + 1], order=order).fit()
    prediction = fitted_model.predict(0, train_index + forecast_length)
    return prediction, fitted_model.arparams, fitted_model.maparams, \
        fitted_model.aic

    # May need to change 0 to 1 in predict (if decide differencing is needed,
    # and also may need to do typ="levels" instead of "linear" as it is
    # currently

    # By default, if we include differencing of order d in our model then we
    # probably  shouldn't use a constant, as this constant is equivalent to
    # adding a polynomial trend of order d.
    # If we don't have any differencing (i.e our model is stationary to
    # begin with), then we probably do want to include a constant which is
    # the mean of the series. Failing to use a constant will mean that our
    # model will go to zero in long term forecasts, which is not what we want.
    # See https://robjhyndman.com/hyndsight/arimaconstants/ and
    # https://otexts.com/fpp2/arima-r.html for more information, include how
    # the automated process works. Include a constant is the AICc improves,
    # basically.

    # When selecting the ARIMA model, select one model for each season. To
    # do this, on the (middle month?) of the training data, consult the
    # ACF/PACF and use that to select p,d,q etc. Then systematically
    # vary the parameters, noting the AIC of the best 5 models. Give each a
    # score, then add the scores across the years and choose the one with
    # the best score.

    # use fit(disp=False) to hide convergence information. Possibly play
    # around with the transparams, method (log likelihood etc), solver,
    # and maximum number of iterations as well as the constant


def sarima(data, train_index, forecast_length, order, seasonal_order):
    fitted_model = sm.tsa.statespace.SARIMAX(
        data[:train_index + 1], order=(0, 1, 1), seasonal_order=(0, 1, 1, 24)
    ).fit()
    prediction = fitted_model.predict(0, train_index + forecast_length)
    return prediction, fitted_model.params, fitted_model.aic


def auto(data, train_index, forecast_length, seasonality):
    fitted_model = pm.auto_arima(
        data[:train_index + 1],
        start_p=0, start_q=0, max_p=2, max_q=2,
        start_P=0, start_Q=0, max_Q=2, max_P=2,
        m=seasonality, max_d=2, max_D=2,
        trace=True, suppress_warnings=True, stepwise=True,
        information_criterion='aicc'
    )
    fitted = fitted_model.predict_in_sample(start=0, end=train_index)
    prediction = fitted_model.predict(forecast_length)
    return pd.Series(np.concatenate((fitted, prediction)))

    # Play with the stepwise. Stepwise=True enforces the H-K algo. Possibly
    # also increase the max values or the params?
