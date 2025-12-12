"""
Data Loader Usage Examples
===========================

This script demonstrates how to use the data_loader module.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from portfolio_optimizer.data_loader import (
    load_prices,
    compute_returns,
    load_factors,
    merge_factors,
)
import pandas as pd


def example_1_basic_data_loading():
    """Example 1: Basic price data loading"""
    print("\n" + "="*60)
    print("Example 1: Basic Price Data Loading")
    print("="*60)

    # Load single ticker
    print("\n1. Loading single ticker (AAPL)...")
    prices_single = load_prices('AAPL', start='2023-01-01', end='2023-12-31')
    print(f"Shape: {prices_single.shape}")
    print(f"Date range: {prices_single.index[0]} to {prices_single.index[-1]}")
    print(f"\nFirst 5 rows:\n{prices_single.head()}")

    # Load multiple tickers
    print("\n2. Loading multiple tickers...")
    tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN']
    prices_multi = load_prices(tickers, start='2023-01-01', end='2023-12-31')
    print(f"Shape: {prices_multi.shape}")
    print(f"Columns: {list(prices_multi.columns)}")
    print(f"\nFirst 5 rows:\n{prices_multi.head()}")

    # Summary statistics
    print("\n3. Summary statistics:")
    print(prices_multi.describe())

    return prices_multi


def example_2_compute_returns():
    """Example 2: Computing returns"""
    print("\n" + "="*60)
    print("Example 2: Computing Returns")
    print("="*60)

    # Load prices
    tickers = ['AAPL', 'MSFT']
    prices = load_prices(tickers, start='2023-01-01', end='2023-12-31')

    # Simple returns
    print("\n1. Simple (arithmetic) returns:")
    simple_returns = compute_returns(prices, method='simple')
    print(f"Shape: {simple_returns.shape}")
    print(f"\nFirst 5 rows:\n{simple_returns.head()}")
    print(f"\nMean daily returns:\n{simple_returns.mean()}")
    print(f"\nDaily volatility:\n{simple_returns.std()}")

    # Log returns
    print("\n2. Log returns:")
    log_returns = compute_returns(prices, method='log')
    print(f"\nFirst 5 rows:\n{log_returns.head()}")

    # Multi-period returns
    print("\n3. Weekly returns (5-day periods):")
    weekly_returns = compute_returns(prices, method='simple', periods=5)
    print(f"Shape: {weekly_returns.shape}")
    print(f"\nFirst 5 rows:\n{weekly_returns.head()}")

    # Annualized metrics (assuming 252 trading days)
    print("\n4. Annualized metrics:")
    ann_return = simple_returns.mean() * 252
    ann_vol = simple_returns.std() * (252 ** 0.5)
    sharpe = ann_return / ann_vol

    print(f"Annualized returns:\n{ann_return}")
    print(f"\nAnnualized volatility:\n{ann_vol}")
    print(f"\nSharpe ratio (assuming Rf=0):\n{sharpe}")

    return simple_returns


def example_3_load_factors():
    """Example 3: Loading factor data"""
    print("\n" + "="*60)
    print("Example 3: Loading Factor Data")
    print("="*60)

    # Load daily factors
    print("\n1. Loading daily Fama-French factors...")
    factors_daily = load_factors(start='2023-01-01', end='2023-12-31', frequency='daily')
    print(f"Shape: {factors_daily.shape}")
    print(f"Columns: {list(factors_daily.columns)}")
    print(f"\nFirst 5 rows:\n{factors_daily.head()}")

    # Summary statistics
    print("\n2. Factor summary statistics:")
    print(factors_daily.describe())

    # Correlation between factors
    print("\n3. Factor correlations:")
    print(factors_daily.corr())

    # Load specific factors
    print("\n4. Loading specific factors only...")
    specific_factors = load_factors(
        start='2023-01-01',
        end='2023-12-31',
        factors=['Mkt-RF', 'SMB', 'HML']
    )
    print(f"Columns: {list(specific_factors.columns)}")

    return factors_daily


def example_4_merge_data():
    """Example 4: Merging returns with factors"""
    print("\n" + "="*60)
    print("Example 4: Merging Returns with Factors")
    print("="*60)

    # Load and compute returns
    print("\n1. Loading asset data...")
    tickers = ['AAPL', 'MSFT', 'GOOG']
    prices = load_prices(tickers, start='2023-01-01', end='2023-12-31')
    returns = compute_returns(prices)
    print(f"Returns shape: {returns.shape}")

    # Load factors
    print("\n2. Loading factor data...")
    factors = load_factors(start='2023-01-01', end='2023-12-31')
    print(f"Factors shape: {factors.shape}")

    # Merge
    print("\n3. Merging returns with factors...")
    combined = merge_factors(returns, factors, how='inner')
    print(f"Combined shape: {combined.shape}")
    print(f"Columns: {list(combined.columns)}")
    print(f"\nFirst 5 rows:\n{combined.head()}")

    # Calculate excess returns
    if 'RF' in combined.columns:
        print("\n4. Computing excess returns (returns - risk-free rate)...")
        for ticker in tickers:
            combined[f'{ticker}_excess'] = combined[ticker] - combined['RF']

        print(f"New columns: {[c for c in combined.columns if 'excess' in c]}")
        print(f"\nMean excess returns:\n{combined[[c for c in combined.columns if 'excess' in c]].mean()}")

    return combined


def example_5_full_pipeline():
    """Example 5: Complete data pipeline"""
    print("\n" + "="*60)
    print("Example 5: Complete Data Pipeline")
    print("="*60)

    # Define portfolio
    tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'META']
    start_date = '2022-01-01'
    end_date = '2023-12-31'

    print(f"\nBuilding dataset for portfolio: {tickers}")
    print(f"Period: {start_date} to {end_date}")

    # Step 1: Load prices
    print("\n1. Loading price data...")
    prices = load_prices(tickers, start=start_date, end=end_date)
    print(f"   ✓ Loaded {len(prices)} days of price data")

    # Step 2: Compute returns
    print("\n2. Computing returns...")
    returns = compute_returns(prices, method='simple')
    print(f"   ✓ Computed {len(returns)} days of returns")

    # Step 3: Load factors
    print("\n3. Loading factor data...")
    factors = load_factors(start=start_date, end=end_date)
    print(f"   ✓ Loaded {len(factors)} days of factor data")

    # Step 4: Merge
    print("\n4. Merging returns with factors...")
    dataset = merge_factors(returns, factors, how='inner')
    print(f"   ✓ Final dataset has {len(dataset)} observations")
    print(f"   ✓ Total features: {len(dataset.columns)}")

    # Step 5: Data quality checks
    print("\n5. Data quality checks...")
    print(f"   • Missing values: {dataset.isna().sum().sum()}")
    print(f"   • Infinite values: {np.isinf(dataset).sum().sum()}")
    print(f"   • Date range: {dataset.index[0]} to {dataset.index[-1]}")

    # Step 6: Summary statistics
    print("\n6. Portfolio summary statistics:")
    summary = pd.DataFrame({
        'Mean Daily Return': dataset[tickers].mean(),
        'Daily Volatility': dataset[tickers].std(),
        'Annualized Return': dataset[tickers].mean() * 252,
        'Annualized Vol': dataset[tickers].std() * (252 ** 0.5),
        'Sharpe Ratio': (dataset[tickers].mean() / dataset[tickers].std()) * (252 ** 0.5)
    })
    print(summary)

    # Step 7: Correlation matrix
    print("\n7. Asset correlation matrix:")
    print(dataset[tickers].corr())

    return dataset


def main():
    """Run all examples"""
    print("\n" + "="*60)
    print("Portfolio Optimizer - Data Loader Examples")
    print("="*60)

    try:
        # Run examples
        prices = example_1_basic_data_loading()
        returns = example_2_compute_returns()
        factors = example_3_load_factors()
        combined = example_4_merge_data()
        dataset = example_5_full_pipeline()

        print("\n" + "="*60)
        print("All examples completed successfully!")
        print("="*60)

    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    import numpy as np
    main()
