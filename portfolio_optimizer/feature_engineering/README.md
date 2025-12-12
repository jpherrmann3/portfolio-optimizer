# Feature Engineering Module

## Overview

The feature_engineering module provides functions for creating ML-ready features from financial time series data. It includes lagged features, rolling statistics, technical indicators, momentum features, and regime detection.

## Features

- **Lagged Features**: Create time-lagged variables for ML models
- **Rolling Statistics**: Compute rolling mean, volatility, skewness, kurtosis
- **Technical Indicators**: RSI, SMA, EMA, Bollinger Bands, MACD
- **Momentum Features**: Multi-period momentum and price-to-moving-average ratios
- **Regime Detection**: Identify market regimes using K-means or HMM
- **Time Features**: Calendar and cyclical time-based features

## Quick Start

```python
from portfolio_optimizer.data_loader import load_prices, compute_returns
from portfolio_optimizer.feature_engineering import (
    create_lagged_features,
    compute_rolling_stats,
    compute_technical_indicators,
    detect_regimes
)

# Load data
prices = load_prices(['AAPL', 'MSFT'], start='2023-01-01')
returns = compute_returns(prices)

# Create features
lagged = create_lagged_features(returns, lags=5)
rolling = compute_rolling_stats(returns, window=21)
tech = compute_technical_indicators(prices)
regimes = detect_regimes(returns, n_regimes=3)
```

## Functions

### `create_lagged_features(data, lags=5, columns=None)`

Create lagged features for time series prediction.

**Parameters:**
- `data`: DataFrame of time series data
- `lags`: Number of lags (int) or list of specific lags
- `columns`: Specific columns to lag (optional)

**Returns:** DataFrame with original data plus lagged features

**Example:**
```python
# Create lags 1-5
lagged = create_lagged_features(returns, lags=5)

# Create specific lags
lagged = create_lagged_features(returns, lags=[1, 5, 21])

# Lag only specific columns
lagged = create_lagged_features(returns, lags=3, columns=['AAPL'])
```

### `compute_rolling_stats(data, window=21, stats=None, min_periods=None)`

Compute rolling window statistics.

**Parameters:**
- `data`: Time series DataFrame
- `window`: Rolling window size (e.g., 21 days ≈ 1 month)
- `stats`: List of statistics ['mean', 'std', 'skew', 'kurt', 'min', 'max']
- `min_periods`: Minimum observations required

**Returns:** DataFrame with rolling statistics

**Example:**
```python
# All statistics with 21-day window
rolling = compute_rolling_stats(returns, window=21)

# Specific statistics
rolling = compute_rolling_stats(
    returns, 
    window=21, 
    stats=['mean', 'std']
)
```

### `compute_rolling_volatility(returns, window=21, annualize=True, min_periods=None)`

Compute rolling volatility from returns.

**Parameters:**
- `returns`: Return series
- `window`: Rolling window size
- `annualize`: Whether to annualize (assumes 252 trading days)
- `min_periods`: Minimum observations

**Returns:** DataFrame of rolling volatility

**Example:**
```python
# Annualized 21-day volatility
vol = compute_rolling_volatility(returns, window=21, annualize=True)

# Multiple windows
vol_10d = compute_rolling_volatility(returns, window=10)
vol_21d = compute_rolling_volatility(returns, window=21)
vol_63d = compute_rolling_volatility(returns, window=63)
```

### `compute_technical_indicators(prices, indicators=None)`

Compute common technical indicators.

**Parameters:**
- `prices`: Price DataFrame
- `indicators`: List of indicators to compute
  - Available: ['sma', 'ema', 'rsi', 'momentum', 'bollinger']

**Returns:** DataFrame with technical indicators

**Example:**
```python
# All indicators
tech = compute_technical_indicators(prices)

# Specific indicators only
tech = compute_technical_indicators(
    prices, 
    indicators=['sma', 'rsi']
)
```

### Individual Technical Indicators

#### `compute_sma(series, window=20)`
Simple Moving Average

#### `compute_ema(series, span=20)`
Exponential Moving Average

#### `compute_rsi(series, window=14)`
Relative Strength Index (0-100)

#### `compute_bollinger_bands(series, window=20, num_std=2.0)`
Returns: (middle_band, upper_band, lower_band)

**Example:**
```python
# Individual indicators
sma_20 = compute_sma(prices['AAPL'], window=20)
ema_12 = compute_ema(prices['AAPL'], span=12)
rsi = compute_rsi(prices['AAPL'], window=14)
mid, upper, lower = compute_bollinger_bands(prices['AAPL'])

# RSI interpretation
if rsi.iloc[-1] > 70:
    print("Overbought")
elif rsi.iloc[-1] < 30:
    print("Oversold")
```

### `create_momentum_features(prices, periods=None)`

Create momentum features at multiple time horizons.

**Parameters:**
- `prices`: Price DataFrame
- `periods`: List of periods (default: [5, 10, 21, 63])

**Returns:** DataFrame with momentum features

**Example:**
```python
momentum = create_momentum_features(prices, periods=[10, 21, 63])

# Features include:
# - {ticker}_momentum_{period}: percentage change
# - {ticker}_price_to_sma_{period}: price relative to SMA
```

### `detect_regimes(returns, n_regimes=3, method='kmeans', features=None)`

Detect market regimes using clustering or HMM.

**Parameters:**
- `returns`: Return series
- `n_regimes`: Number of regimes (e.g., bull, bear, sideways)
- `method`: 'kmeans' or 'hmm'
- `features`: Features for detection ['return', 'volatility']

**Returns:** DataFrame with regime labels and probabilities

**Example:**
```python
# K-means clustering (3 regimes)
regimes = detect_regimes(returns, n_regimes=3, method='kmeans')

# Access regime labels
current_regime = regimes['regime'].iloc[-1]

# One-hot encoded regimes
regime_dummies = regimes[['regime_0', 'regime_1', 'regime_2']]

# Analyze regime characteristics
for regime_id in regimes['regime'].unique():
    regime_returns = returns[regimes['regime'] == regime_id]
    print(f"Regime {regime_id}: mean={regime_returns.mean()}, vol={regime_returns.std()}")
```

## Advanced Features

### `compute_cross_sectional_features(returns, window=21)`

Compute cross-sectional features across multiple assets (rank, z-score, relative performance).

### `create_interaction_features(data, interactions=None)`

Create interaction features between variables (pairwise products).

### `create_time_features(index)`

Create time-based features from datetime index (day of week, month, cyclical encoding).

## Complete Feature Pipeline Example

```python
# 1. Load data
prices = load_prices(['AAPL', 'MSFT', 'GOOG'], start='2023-01-01')
returns = compute_returns(prices)

# 2. Create comprehensive feature set
lagged = create_lagged_features(returns, lags=5)
rolling_stats = compute_rolling_stats(returns, window=21)
volatility = compute_rolling_volatility(returns, window=21)
tech_indicators = compute_technical_indicators(prices)
momentum = create_momentum_features(prices, periods=[5, 21, 63])
regimes = detect_regimes(returns, n_regimes=3)

# 3. Combine all features
feature_set = pd.concat([
    returns,
    lagged,
    rolling_stats,
    volatility,
    tech_indicators,
    momentum,
    regimes
], axis=1)

# 4. Remove duplicates and handle missing values
feature_set = feature_set.loc[:, ~feature_set.columns.duplicated()]
feature_set = feature_set.dropna()

print(f"Total features: {feature_set.shape[1]}")
print(f"Ready for ML training with {feature_set.shape[0]} observations")
```

## Best Practices

### Avoid Data Leakage
- All features use only past data (no lookahead bias)
- Lagged features shift data by specified periods
- Rolling windows only use historical data

### Feature Scaling
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
features_scaled = pd.DataFrame(
    scaler.fit_transform(feature_set),
    columns=feature_set.columns,
    index=feature_set.index
)
```

### Handling Missing Values
```python
# Drop rows with NaN (after rolling windows warm up)
features_clean = feature_set.dropna()

# Or forward fill (use cautiously)
features_filled = feature_set.fillna(method='ffill')
```

### Feature Selection
```python
# Remove highly correlated features
corr_matrix = feature_set.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
features_reduced = feature_set.drop(columns=to_drop)
```

## Testing

Run the test suite:
```bash
pytest tests/test_feature_engineering.py -v
```

## Examples

See `examples/feature_engineering_example.py` for comprehensive usage examples including:
- Basic feature creation
- Technical indicator analysis
- Momentum strategies
- Regime detection
- Complete ML pipeline

## Dependencies

- pandas
- numpy
- scipy
- scikit-learn
- hmmlearn (optional, for HMM-based regime detection)

## Performance Tips

1. **Vectorization**: All functions use vectorized pandas operations
2. **Memory**: Use smaller windows for large datasets
3. **Computation**: Create only features you need
4. **Parallelization**: Process different assets in parallel if needed

## Common Use Cases

### Forecasting
```python
# Create features for return prediction
features = create_lagged_features(returns, lags=5)
features = pd.concat([features, compute_rolling_stats(returns, 21)], axis=1)
```

### Technical Trading
```python
# Technical analysis features
tech = compute_technical_indicators(prices)
momentum = create_momentum_features(prices)
```

### Risk Management
```python
# Volatility and regime features
vol = compute_rolling_volatility(returns, window=21)
regimes = detect_regimes(returns, n_regimes=3)
```

### Factor Models
```python
# Features for factor analysis
rolling = compute_rolling_stats(returns, window=63)
momentum = create_momentum_features(prices, periods=[21, 63, 252])
```
