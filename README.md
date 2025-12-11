# Portfolio Optimizer with ML Forecasting and Risk Modeling

[![Python Version](https://img.shields.io/badge/python-3.13%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## Project Overview

The **Portfolio Optimizer with ML Forecasting and Risk Modeling** is a Python package designed to build, evaluate, and backtest quantitatively optimized investment portfolios. It combines **machine learning forecasting**, **advanced risk modeling**, and **portfolio optimization techniques** to produce actionable strategies for both research and practical investment purposes.

The project demonstrates:

- End-to-end **financial data pipelines**
- ML-based **forecasting of returns and volatility**
- **Covariance and risk estimation** with advanced techniques
- Portfolio optimization using **Mean-Variance, Hierarchical Risk Parity, and Black-Litterman approaches**
- Backtesting with **performance metrics, transaction costs, and turnover analysis**
- Unit-tested, modular, and production-ready code

This repository is perfect for showcasing your skills in **Python, ML, quantitative finance, and MLOps-style project structure**.

---

## Features

### 1. Data Ingestion & Processing
- Download historical price data from Yahoo Finance or custom CSVs
- Compute daily, weekly, or monthly returns
- Handle missing data and alignment
- Integrate factor datasets (Fama-French, macroeconomic indicators)

### 2. Feature Engineering
- Rolling statistics (mean, volatility, skew, kurtosis)
- Technical indicators (momentum, RSI, SMA)
- Lagged features for ML forecasting
- Optional regime detection using clustering or HMM

### 3. ML Forecasting
- Forecast returns using ARIMA, XGBoost, Random Forest, or LSTM/GRU
- Forecast volatility using GARCH/EGARCH
- Generate prediction intervals for risk-aware optimization

### 4. Risk Modeling
- Sample covariance, Ledoit-Wolf shrinkage, or exponential weighting
- Compute portfolio risk metrics (volatility, VaR, CVaR, drawdowns)

### 5. Portfolio Optimization
- Mean-Variance Optimization (MVO)
- Hierarchical Risk Parity (HRP)
- Black-Litterman model
- Optional robust optimization to handle estimation errors

### 6. Backtesting & Evaluation
- Walk-forward backtesting with rebalancing frequency
- Include transaction costs and turnover constraints
- Evaluate Sharpe ratio, Sortino ratio, max drawdown, and other performance metrics
- Generate visualizations and summary reports

### 7. CLI & Interactive Examples (Optional)
- Run optimization strategies from the command line
- Example Jupyter notebooks demonstrate workflows and best practices

---

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/portfolio-optimizer.git
cd portfolio-optimizer

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # Linux / Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

## Quick Start
```python
from portfolio_optimizer.data_loader import load_prices
from portfolio_optimizer.models import ReturnForecaster
from portfolio_optimizer.risk_model import compute_covariance
from portfolio_optimizer.optimizer import optimize_mvo
from portfolio_optimizer.backtesting import run_backtest

# 1. Load price data
prices = load_prices(tickers=["AAPL", "MSFT", "GOOG"], start="2020-01-01", end="2025-01-01")

# 2. Compute returns
returns = prices.pct_change().dropna()

# 3. Fit return forecast
forecaster = ReturnForecaster(model_type="xgboost")
forecaster.fit(returns)
mu_forecast = forecaster.predict(horizon=21)

# 4. Compute risk
cov_matrix = compute_covariance(returns, method="ledoit_wolf")

# 5. Optimize portfolio
weights = optimize_mvo(mu_forecast, cov_matrix, target="sharpe")

# 6. Run backtest
backtest_results = run_backtest(weights, returns, rebalance_freq="monthly")
```

## Folder Structure
```
portfolio-optimizer/
├── portfolio_optimizer/
│   ├── data_loader/
│   ├── feature_engineering/
│   ├── models/
│   ├── risk_model/
│   ├── optimizer/
│   ├── backtesting/
│   ├── cli/
│   └── utils.py
├── tests/
├── examples/
├── config/
│   └── example.yaml
├── README.md
├── setup.py
├── requirements.txt
└── LICENSE
```

## Back Testing Metrics
- Sharpe Ratio
- Sortino Ratio
- Max Drawdown
- Calmar Ratio
- Portfolio turnover and transaction cost analysis
- Weight stability and diversification analysis

## Optional Extensions
- Interactive Streamlit dashboard for live portfolio adjustments
- Docker container for deployment
- Integration with real-time market APIs
- Synthetic data generation for testing


## Dependencies
- Python 3.13+
- Pandas, NumPy
- Scikit-learn, XGBoost, lightGBM
- arch
- yfinance
- cvxpy
- matplotlib, seaborn, plotly
- pytest

## Contributing
Contributions are welcome! Please open an issue or submit a pull request. Make sure to follow the coding standards and add unit tests for new features.
