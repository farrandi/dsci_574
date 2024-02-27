# Lecture 3: ARIMA Models

## ARIMA Models

- **ARIMA**: AutoRegressive Integrated Moving Average
- Commonly used for time series forecasting (other than exponential smoothing)
- Based on autocorrelation of data
- Do not model trend nor seasonality, so it is typically constrained to **stationary** data

### Stationarity

- Statistical properties of a time series do not change over time
  - Mean, variance is constant
  - Is roughly horizontal (no strong trend)
  - Does not show predictable patterns (no seasonality)
- DOES not mean that the time series is constant, just that the way it changes is constant
- It is one way of modelling dependence structure
  - Can only be independent in one way but dependent in many ways

#### Strong vs Weak Stationarity

| Property                                  | Strong Stationarity | Weak Stationarity        |
| ----------------------------------------- | ------------------- | ------------------------ |
| Mean, Variance, Autocovariance            | Constant            | Constant                 |
| Higher order moments (skewness, kurtosis) | Constant            | Not necessarily constant |

- Weak stationarity is often sufficient for time series analysis

### Checking for Stationarity

1. **Visual Inspection**: Plot the time series
   - Look for trends, seasonality, and variance (none of these should be present)
   - Make a correlogram plot (ACF plot should rapidly decay to 0)
2. **Summary Statistics**: Calculate mean, variance, and autocovariance
   - Mean and variance should be roughly constant over time
3. **Hypothesis Testing**: Use statistical tests
   - **Augmented Dickey-Fuller (ADF) test**
     - Null hypothesis: Time series is non-stationary
     - small p: it is stationary (reject null)
     - Use `statsmodels.tsa.stattools.adfuller`
   - **Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test**
     - Null hypothesis: Time series is stationary
     - small p: it is non-stationary (reject null)

```python
from statsmodels.tsa.stattools import adfuller

# ADF test
result = adfuller(data)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
```

### Making a Time Series Stationary

- **Stabilizing the variance using transformations**

  - Log or box-cox transformation

  $$w_t = \begin{cases} \frac{y_t^\lambda - 1}{\lambda} & \text{if } \lambda \neq 0 \\ \ln(y_t) & \text{if } \lambda = 0 \end{cases}$$

  ```python
  from scipy.stats import boxcox
  import numpy as np

  data = boxcox(data, lmbda=0)

  # log transformation
  data = np.log(data)
  ```

- **Stabilize the mean using differencing**

  - First difference: $y' = y_t - y_{t-1}$
  - Second difference: $y'' = y' - y'_{t-1} = y_t - 2y_{t-1} + y_{t-2}$
  - Seasonal difference: $y' = y_t - y_{t-m}$, where $m$ is the seasonal period

  ```python
  # First difference
  data1 = data.diff().dropna()
  # Second difference
  data2 = data.diff().diff().dropna()
  # Seasonal difference
  data_m = data.diff(m).dropna()
  ```

### AR and MA Models

| AR (AutoRegressive) Model                                                      | MA (Moving Average) Model                                                                                 |
| ------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------- |
| $y_t = \phi_1 y_{t-1} + \phi_2 y_{t-2} + \ldots + \phi_p y_{t-p} + \epsilon_t$ | $y_t = \epsilon_t + \theta_1 \epsilon_{t-1} + \theta_2 \epsilon_{t-2} + \ldots + \theta_q \epsilon_{t-q}$ |
| $p$: order of the AR model                                                     | $q$: order of the MA model                                                                                |
| $\phi$: AR coefficients                                                        | $\theta$: MA coefficients                                                                                 |
| $\epsilon_t$: white noise                                                      | $\epsilon_t$: white noise                                                                                 |
| Long memory model: $y_1$ has a direct effect on $y_t$ for all $t$              | Short memory model: $y_t$ is only affected by recent values of $\epsilon$                                 |
| Captures long-term trends and patterns                                         | Captures short-term fluctuations/ noise                                                                   |
| Good for modeling time-series with dependency on past values                   | Good for modeling time-series with a lot of volatility and noise                                          |
| Less sensitive to choice of lag or window size                                 | More sensitive to choice of lag or window size                                                            |

### ARMA Model

- **ARMA**: AutoRegressive Moving Average
- Combines AR and MA models
- Key Idea: Parsimony
  - fit a simpler, mixed model with fewer parameters, than either a pure AR or a pure MA model

$$y_t = c + \phi_1 y_{t-1} + \phi_2 y_{t-2} + \ldots + \phi_p y_{t-p} + \epsilon_t + \theta_1 \epsilon_{t-1} + \theta_2 \epsilon_{t-2} + \ldots + \theta_q \epsilon_{t-q}$$

- $c$: constant
- $\phi$: AR coefficients
- $\theta$: MA coefficients
- Usually write it as `ARMA(p, q)`

### ARIMA Model

- **ARIMA**: AutoRegressive Integrated Moving Average
- Combines ARMA with differencing
- `ARIMA(p, d, q)`
  - `p`: order of the AR model
  - `d`: degree of differencing
  - `q`: order of the MA model
- Use `statsmodels.tsa.arima.model.ARIMA`

```python
from statsmodels.tsa.arima.model import ARIMA

# All with first order differencing
model_ar = ARIMA(data["col"], order=(3, 1, 0)).fit() # AR(3)
model_ma = ARIMA(data["col"], order=(0, 1, 1)).fit() # MA(1)
model_arma = ARIMA(data["col"], order=(3, 1, 3)).fit() # ARMA(3, 3)
```

#### ARIMA hyperparameter tuning

```python
import pmdarima as pm

autoarima = pm.auto_arima(data.col,
                          start_q=0, star_d=1, start_q=0,
                          max_p=5, max_d=3, max_q=5,
                          seasonal=False)

autoarima.summary()
```
