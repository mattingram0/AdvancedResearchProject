# **A Comparison of Hybrid and Statistical Forecasting Techniques for Short-Term Energy Demand Forecasting**
## Abstract

Following the latest iteration of the M Time Series Forecasting Competition, this paper compares the performance of the winning (hybrid) model to classical statistical models, when applied specifically to short-term energy demand forecasting. The hybrid model combines an exponential smoothing model and a dilated Long Short-Term Memory recurrent neural network into one hybrid architecture. We extend this model through the inclusion of exogeneous, contemporaneous weather data. The statistical forecasting methods included for comparison are: the Auto-Regressive Integrated Moving Average (ARIMA) model, Holt-Winters’ exponential smoothing, and the Theta method, among others. We find that for forecast horizons of up to 6 hours, the ARIMA model performs optimally. When forecasting at longer horizons, the naive seasonal method is dominant. For consistently strong forecasts across all lead times from 1 to 48 hours, the novel hybrid models introduced in this paper are shown to be preferable. Therefore, we conclude that the hybrid model is indeed suitable for short-term energy demand forecasting.

-----

Please see **Report.pdf** for the full paper
