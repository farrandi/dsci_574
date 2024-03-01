# Lecture 4: Forecasting TimeSeries with ML

## Time Series Forecasting in ML

### Key Differences vs. Traditional ML

| Traditional ML          | Time Series ML                   |
| ----------------------- | -------------------------------- |
| Data is IID             | Data is ordered                  |
| CV is random            | Use sliding window CV            |
| Use feature engineering | Use lags, rolling windows, etc.  |
| Predict new data        | Predict future (specify horizon) |

### `sktime` Library

#### 1. Load Data

```python
import pandas as pd

df = pd.read_csv('data.csv', index_col=0, parse_dates=True)
```

#### 2. Feature Engineering

```python
from sktime.transformations.series.lag import Lag
# Make a new column with lag
df['col-1'] = df['col'].shift(1)
# or use sktime
t = Lag(lags=[1,2,3], index_out="original")
pd.concat([df, t.fit_transform(df)], axis=1)
```

#### 3. Train-Test Split

- Never use random shuffling
- Need to keep temporal order

```python
from sktime.split import temporal_train_test_split
from sklearn.model_selection import train_test_split

y_train, y_test = temporal_train_test_split(y, test_size=0.2)

# or use sklearn
df_train, df_test = train_test_split(df.dropna(), test_size=0.2, shuffle=False)
```

##### Cross-Validation for Time Series

1. **Expanding Window**: start with small training set and increase it
   <img src="images/4_expanding.png" width="300">
2. **Fixed/sliding Window**: use a fixed window size
   <img src="images/4_sliding.png" width="300">

#### 4. Model Fitting

```python
from sktime.forecasting.model_evaluation import evaluate
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.arima import AutoARIMA

forecaster = NaiveForecaster(strategy="last", sp=12) # seasonal naive
# forecaster = AutoARIMA(sp=12)

results = evaluate(forecaster=forecaster, y=y_train, cv=cv, strategy="refit", return_data=True)
```

### Forecasting

1. **One-step forecasting**
2. **Multi-step forecasting**
   a. **Recursive strategy**
   b. **Direct strategy**
   c. **Hybrid strategy**: is dumb and bad
   d. **Multi-output strategy**: 2 different series (e.g. temperature and humidity)

### Feature Preprocessing and Engineering
