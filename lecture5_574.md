# Lecture 5: Forecast Uncertainty, Anomaly Detection, and Imputation

## Probabilistic Forecasting

- We have been dealing with **point forecasts** (modelling averages)
- Want to estimate the **uncertainty** of our forecasts
  - or the extreme (e.g. 90% or 95% quantiles)
    - example: find upper quantile of electricity demand so that we can plan for the maximum demand
  - or predict the variance of the forecast (how volatile a metric will be in the future)

### Analytical

- Assume distribution of forecasts are normal

$$
\hat{y}_{T+h|T} \pm c \times \hat{\sigma}_{h}
$$

- $\hat{\sigma}_{h}$ is the standard deviation of the forecast
- $c$: coverage factor (e.g. 1.96 for 95% confidence interval)

$$
\hat{\sigma}_{h} = \sqrt{\frac{1}{T-K}\sum_{t=1}^{T} e_{t}^{2}}
$$

- Focus is finding $\hat{\sigma}_{h}$
  - $K$: number of parameters
  - $T$: total length of time series
  - $e_{t} = y_{t} - \hat{y}_{t|t-1}$
- Methods that have been derived mathematically:
  | Method | Forecast sd |
  |--------|--------------|
  | Mean | $\hat{\sigma}_{h} = \hat{\sigma_1} \sqrt{1 + \frac{h}{T}}$ |
  |Naive | $\hat{\sigma}_{h} = \hat{\sigma_1} \sqrt{h}$ |
  | Seasonal Naive | $\hat{\sigma}\_{h} = \hat{\sigma_1} \sqrt{\frac{h-1}{m}+1} $ |
    | Drift | $\hat{\sigma}_{h} = \hat{\sigma_1} \sqrt{h(1+\frac{h}{T})}$ |
- _Recall: $h$ is the forecast horizon (steps ahead), $m$ is the seasonal period_

```python
from pandas import pd

c = 1.96 # 95% confidence interval

train['pred'] = train['y'].shift(1)
train['residuals'] = train['y'] - train['pred']
sigma = train['residuals'].std()

h = np.arange(1, len(forecast_index) + 1)
naive_forecast = train['y'].iloc[-1]

# create lower and upper bound
naive = pd.DataFrame({"y": naive_forecast,
                      "pi_lower": naive_forecast - c * sigma * np.sqrt(horizon),
                      "pi_upper": naive_forecast + c * sigma * np.sqrt(horizon),
                      "Label": "Naive"},
                     index=forecast_index)
plot_prediction_intervals(train["y"], naive, "y", valid=valid["y"])
```

```python
# ETS
model = ETSModel(train["y"], error="add", trend="add", seasonal="add").fit(disp=0)

ets = model.get_prediction(start=forecast_index[0], end=forecast_index[-1]).summary_frame()
plot_prediction_intervals(train["y"], ets, "mean", valid=valid["y"], width=800)

# ARIMA
model = ARIMA(train["y"], order=(3, 1, 0), seasonal_order=(2, 1, 0, 12)).fit()

arima = model.get_prediction(start=forecast_index[0], end=forecast_index[-1]).summary_frame()
plot_prediction_intervals(train["y"], arima, "mean", valid=valid["y"], width=800)
```

### Simulation and Bootstrapping

- Assume future errors will be similar to past errors
- Draw from the distribution of past errors to simulate future errors

```python
# Fit an ETS model
model = ETSModel(train["y"], error="add", trend="add").fit(disp=0)

# simulate predictions
ets = model.simulate(anchor="end", nsimulations=len(forecast_index),
                     repetitions=n_simulations,
                     random_errors="bootstrap")
# plot
ax = train["y"].plot.line()
ets.plot.line(ax=ax, legend=False, color="r", alpha=0.05,
              xlabel="Time", ylabel="y", figsize=(8,5));

# get quantiles
ets = pd.DataFrame({"median": ets.median(axis=1),
                    "pi_lower": ets.quantile(1-0.975, axis=1),
                    "pi_upper": ets.quantile(0.975, axis=1)},
                   index=forecast_index)
```

### Quantile Regression

- Wish to predict particular quantile instead of mean

| High Quantile                          | Low Quantile                            |
| -------------------------------------- | --------------------------------------- |
| Higher penalty for predicting **OVER** | Higher penalty for predicting **UNDER** |

- Can use pytorch for this, see [here](https://pages.github.ubc.ca/MDS-2023-24/DSCI_574_spat-temp-mod_instructors/lectures/lecture5_uncertainty.html#quantile-regression)

### Evaluating Distributional Forecast Accuracy

- There are 4 main sources of uncertainty:

  1. Random error term
  2. Uncertainty in model parameter estimates
  3. Uncertainty in model selection
  4. Uncertainty about consistency of data generating process in the future

- Most methods only consider the first source of uncertainty
- Simulation tries to consider 2 and 3
- 4 is practically impossible to consider

## Anomaly Detection

### Rolling Median

### STL Decomposition

### Model-based

### ML approaches

#### Isolation Forest

#### K-NN

## Imputation

```

```
