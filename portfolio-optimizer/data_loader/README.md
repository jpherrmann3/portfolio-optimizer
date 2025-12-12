# Data Loader Module

## Overview

The data_loader module provides functions for ingesting, cleaning, and preprocessing financial data for portfolio optimization.

## Features

- **Price Data Loading**: Download historical price data from Yahoo Finance
- **Return Computation**: Calculate simple or log returns
- **Factor Data**: Load Fama-French factor data for risk analysis
- **Data Merging**: Combine asset returns with factor data
- **Data Quality**: Handle missing values and align time series

## Quick Start

```python
from portfolio_optimizer.data_loader import load_prices, compute_returns, load_factors, merge_factors

# Load price data
prices = load_prices(['AAPL', 'MSFT', 'GOOG'], start='2020-01-01', end='2023-12-31')

# Compute returns
returns = compute_returns(prices, method='simple')

# Load Fama-French factors
factors = load_factors(start='2020-01-01', end='2023-12-31')

# Merge returns with factors
combined = merge_factors(returns, factors)
```

## Functions

### `load_prices(tickers, start, end, source='yahoo', column='Adj Close')`

Download historical price data for specified tickers.

**Parameters:**
- `tickers`: List of ticker symbols or single ticker string
- `start`: Start date (string or datetime)
- `end`: End date (string or datetime), defaults to today
- `source`: Data source, currently supports 'yahoo'
- `column`: Price column to extract (default: 'Adj Close')

**Returns:** DataFrame with dates as index and tickers as columns

**Example:**
```python
# Single ticker
prices = load_prices('AAPL', start='2023-01-01')

# Multiple tickers
prices = load_prices(['AAPL', 'MSFT', 'GOOG'], start='2023-01-01', end='2023-12-31')
```

### `compute_returns(prices, method='simple', periods=1, dropna=True)`

Compute returns from price data.

**Parameters:**
- `prices`: DataFrame of prices
- `method`: 'simple' or 'log'
  - simple: (P_t - P_{t-1}) / P_{t-1}
  - log: log(P_t / P_{t-1})
- `periods`: Number of periods (1=daily, 5=weekly, 21=monthly for daily data)
- `dropna`: Whether to drop NaN values

**Returns:** DataFrame of returns

**Example:**
```python
# Daily simple returns
returns = compute_returns(prices, method='simple')

# Daily log returns
log_returns = compute_returns(prices, method='log')

# Weekly returns
weekly_returns = compute_returns(prices, periods=5)
```

### `load_factors(start, end, frequency='daily', factors=None)`

Load Fama-French factor data.

**Parameters:**
- `start`: Start date
- `end`: End date (defaults to today)
- `frequency`: 'daily' or 'monthly'
- `factors`: List of specific factors to load (optional)
  - Available: ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF']

**Returns:** DataFrame of factor returns

**Example:**
```python
# All factors
factors = load_factors(start='2023-01-01', frequency='daily')

# Specific factors
factors = load_factors(start='2023-01-01', factors=['Mkt-RF', 'SMB', 'HML'])
```

**Note:** Factor returns are in decimal format (0.01 = 1%)

### `merge_factors(returns, factors, how='inner')`

Merge asset returns with factor data.

**Parameters:**
- `returns`: DataFrame of asset returns
- `factors`: DataFrame of factor returns
- `how`: Merge method ('inner', 'outer', 'left', 'right')

**Returns:** Combined DataFrame

**Example:**
```python
combined = merge_factors(returns, factors, how='inner')
print(combined.columns)  # ['AAPL', 'MSFT', 'Mkt-RF', 'SMB', 'HML', ...]
```

## Additional Utilities

### `align_data(*dataframes, method='inner')`

Align multiple DataFrames to have matching date indices.

### `handle_missing_data(data, method='ffill', limit=None, threshold=0.5)`

Handle missing data with various strategies:
- `ffill`: Forward fill
- `bfill`: Backward fill
- `interpolate`: Linear interpolation
- `drop`: Drop rows with missing values
- `drop_cols`: Drop columns with too many missing values

## Data Quality

The module automatically handles:
- Missing values (forward/backward fill)
- Market holidays (no trading days)
- Data alignment across different time series
- Invalid or delisted tickers

## Testing

Run the test suite:
```bash
pytest tests/test_data_loader.py -v
```

## Examples

See `examples/data_loader_example.py` for comprehensive usage examples.

## Dependencies

- pandas
- numpy
- yfinance
- pandas-datareader (optional, for Fama-French factors)

## Notes

- All dates are handled as pandas DatetimeIndex
- Price data is adjusted for splits and dividends using 'Adj Close'
- Factor data comes from Kenneth French's Data Library
- Returns are in decimal format (0.01 = 1%)
