import statsmodels.api as sm
import pmdarima as pm
import numpy as np
import pandas as pd


def arima(data, forecast_length, order):
    fitted_model = sm.tsa.ARIMA(data, order=order).fit(disp=-1)
    prediction = fitted_model.predict(0, len(data) + forecast_length - 1)
    return prediction, np.concatenate(
        (fitted_model.arparams, fitted_model.maparams)
    ).tolist()

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


def sarima(data, forecast_length, order, seasonal_order):
    fitted_model = sm.tsa.statespace.SARIMAX(
        data, order=order, seasonal_order=seasonal_order
    ).fit(disp=-1)
    prediction = fitted_model.predict(0, len(data) + forecast_length - 1)
    return prediction, fitted_model.params.to_dict()


def auto(data, forecast_length, seasonality):
    fitted_model = pm.auto_arima(
        data,
        start_p=0, start_q=0, max_p=2, max_q=2,
        start_P=0, start_Q=0, max_Q=2, max_P=2,
        m=seasonality, max_d=2, max_D=2,
        trace=False, suppress_warnings=True, stepwise=True,
        information_criterion='aicc'
    )
    fitted = fitted_model.predict_in_sample(start=0, end=(len(data) - 1))
    prediction = fitted_model.predict(forecast_length)
    params = fitted_model.params().tolist()
    hyper_params = fitted_model.get_params()
    params.append(hyper_params["order"])
    params.append(hyper_params["seasonal_order"])

    return pd.Series(np.concatenate((fitted, prediction))), params
    # Play with the stepwise. Stepwise=True enforces the H-K algo. Possibly
    # also increase the max values or the params?
