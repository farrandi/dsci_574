# Lecture 2: Forecasting Methods

## Forecasting

- **Forecasting**: Predicting future values of a time series

### Baseline Forecasting Methods

1. **Average**: Use average of all past observations

<img src="images/2_ave.png" width="300">

2. **Naive**: Use the last observation as the forecast
3. **Seasonally Adjusted Naive**: Same as Naive but with seasonally adjusted data (classical decomposition)

<img src="images/2_naive_san.png" width="350">

4. **Seasonally Naive**: Use the last observation from the same season (only one with seasonality)

```python
df["month"] = df.index.month
last_season = (df.drop_duplicates("month", keep="last")
                 .sort_values(by="month")
                 .set_index("month")["value"]
              )
df = df.drop(columns="month")
last_season
```

<img src="images/2_sn.png" width="300">

5. **Drift**: Linearly extrapolate the trend (only one that is not a straight horizontal line)

<img src="images/2_drift.png" width="300">

### Exponential Models

#### Simple Exponential Smoothing

- Forecast is a weighted average of all past observations
- Recursively defined: $\hat{y}_{t+1|t} = \alpha y_t + (1 - \alpha) \hat{y}_{t|t-1}$
- **$\alpha$: Smoothing parameter**
  - Close to 0: More weight to past observations
  - Close to 1: More weight to current observation (closer to Naive forecast)
- **Initial Forecast**:
  - $\hat{y}_{1|0} = y_1$
  - Heuristic: linear interpolation of the first few observations
  - Learn it by optimizing SSE
- Forecasts are flat
-

```python
 from statsmodels.tsa.holtwinters import SimpleExpSmoothing

 SES = SimpleExpSmoothing(data, initialization_method='heuristic')=

 # Fit the model
 model = SES.fit(smoothing_level=0.2, optimized=False)

 # Forecast
 forecast = model.forecast(steps=5)
```

#### Holt's Method

- Extend SES to include a trend component
  $$\hat{y}_{t+h|t} = \ell_t + h b_t$$

  $$\ell_t = \alpha y_t + (1 - \alpha)(\ell_{t-1} + b_{t-1})$$

  $$b_t = \beta(\ell_t - \ell_{t-1}) + (1 - \beta)b_{t-1}$$

- $\ell_t$: Level
- $b_t$: Smoothness of the trend
  - Close to 0: Trend is more linear
  - Close to 1: Trend changes with each observation
- $\alpha$: Smoothing parameter for level

#### Holt's Winter Method

- Extend Holt's method to include a seasonal component
  $$\hat{y}_{t+h|t} = \ell_t + h b_t + s_{t-m+h_m}$$

  $$b_t = \beta(\ell_t - \ell_{t-1}) + (1 - \beta)b_{t-1}$$

- For Additive Seasonal:
  $$\ell_t = \alpha(y_t - s_{t-m}) + (1 - \alpha)(\ell_{t-1} + b_{t-1})$$

  $$s_t = \gamma(y_t - \ell_{t-1} - b_{t-1}) + (1 - \gamma)s_{t-m}$$

- For Multiplicative Seasonal:
  $$\ell_t = \alpha\frac{y_t}{s_{t-m}} + (1 - \alpha)(\ell_{t-1} + b_{t-1})$$

  $$s_t = \gamma\frac{y_t}{\ell_{t-1} + b_{t-1}} + (1 - \gamma)s_{t-m}$$

| Trend component        | Seasonal Component   |
| ---------------------- | -------------------- |
| None `(N)`             | None `(N)`           |
| Additive `(A)`         | Additive `(A)`       |
| Additive Damped `(Ad)` | Multiplicative `(M)` |

- Simple Exponential Smoothing `(N,N)`
- Holt's Method `(A,N)`
- Holt's Winter Method `(A,A)`

```python
from statsmodels.tsa.holtwinters import ExponentialSmoothing

 model = ExponentialSmoothing(data,
     trend='add',
     damped_trend=True,
     seasonal='mul',
     seasonal_periods=12,
     initialization_method='estimated'
 ).fit(method="least_squares")
```

#### ETS (Error, Trend, Seasonal) Models

- Components:
  - Error: `{A, M}`
  - Trend: `{N, A, Ad}`
  - Seasonal: `{N, A, M}`

```python
from statsmodels.tsa.holtwinters import ETSModel

 model = ETSModel(data,
      error='add',
      trend='add',
      damped_trend=True,
      seasonal='add',
      seasonal_periods=12
 ).fit()

 # Forecast
 model.forecast(steps=5)

 # Summary
  model.summary()
```

- Can generate prediction intervals (confidence intervals):
  1. `model.get_prediction()` (analytical)
  2. `model.simulate()`

```python
pred = model.get_prediction(start=df.index[-1] + pd.DateOffset(months=1), end="2020").summary_frame()

# or
q = 0.975 # 95% CI
sim = model.simulate(anchor="end", nsimulations=348, repetitions=100, random_errors="bootstrap")
simu = pd.DataFrame({"median": simulations.median(axis=1),
                     "pi_lower": simulations.quantile((1 - q), axis=1),
                     "pi_upper": simulations.quantile(q, axis=1)},
                    index=simulations.index)
```

### Selecting a Model

- **Metrics**, Commonly used:
  - AIC, BIC
  - SSE/ MSE/ RMSE
  ```python
  # using ets model from above
  model.aic
  model.bic
  model.mse
  ```
- **Residuals**:
  - Visual inspection (should be uncorrelated, zero mean, normally distributed)
  - Running diagnostic Portmanteau tests:
    - **Ljung-Box Test**: $H_0$: Residuals are uncorrelated (white noise)
      - p-value < 0.05: Reject $H_0$ (bad)
    - **Jarque-Bera Test**: $H_0$: Residuals are normally distributed
      - p-value < 0.05: Reject $H_0$ (bad)

```python
# using ets model from above
model.summary().tables[-1]

# Ljung-Box Test
p = model.test_serial_correlation(method="ljungbox", lags=10)[0,1,-1]
# Jarque-Bera Test
p = model.test_normality(method="jarquebera")[0,1]
```

- **Out-of-sample Forecasting**:
  - Split data into training and testing
  - Fit model on training data
  - Forecast on testing data
  - Compare forecast with actuals
