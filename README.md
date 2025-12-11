# portfolio-optimizer

## Project Overview

The **Portfolio Optimizer with ML Forecasting and Risk Modeling** is a Python package designed to build, evaluate, and backtest quantitatively optimized investment portfolios. It combines **machine learning forecasting**, **advanced risk modeling**, and **portfolio optimization techniques** to produce actionable strategies for both research and practical investment purposes.

The project demonstrates:

- End-to-end financial data pipelines
- ML-based forecasting of returns and volatility
- Covariance and risk estimation with advanced techniques
- Portfolio optimization using Mean-Variance, Hierarchical Risk Parity, and Black-Litterman approaches
- Backtesting with performance metrics, transaction costs, and turnover analysis
- Unit-tested, modular, and production-ready code

## Key Features

1. Data Ingestion & Processing
   - Download historical price data from Yahoo Finance or custom CSVs
   - Compute daily, weekly, or monthly returns
   - Handle missing data and alignment
   - Integrate factor datasets (Fama-French, macroeconomic indicators)
2. Feature Engineering
   - Rolling statistics (mean, volatility, skew, kurtosis)
   - Technical indicators (momentum, RSI, SMA)
   - Lagged features for ML forecasting
   - Optional regime detection using clustering or HMM
3. ML Forecasting
   - Forecast returns using ARIMA, XGBoost, Random Forest, or LSTM/GRU
   - Forecast volatility using GARCH/EGARCH
   - Generate prediction intervals for risk-aware optimization
4. Risk Modeling
   - Sample covariance, Ledoit-Wolf shrinkage, or exponential weighting
   - Compute portfolio risk metrics (volatility, VaR, CVaR, drawdowns)
5. Portfolio Optimization
   - Mean-Variance Optimization (MVO)
   - Hierarchical Risk Parity (HRP)
   - Black-Litterman model
   - Optional robust optimization to handle estimation errors
6. Backtesting & Evaluation
   - Walk-forward backtesting with rebalancing frequency
   - Include transaction costs and turnover constraints
   - Evaluate Sharpe ratio, Sortino ratio, max drawdown, and other performance metrics
   - Generate visualizations and summary reports
7. CLI & Interactive Examples (Optional)
   - Run optimization strategies from the command line
   - Example Jupyter notebooks demonstrate workflows and best practices