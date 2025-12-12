"""
Feature Engineering Usage Examples
===================================

This script demonstrates how to use the feature_engineering module.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from portfolio_optimizer.data_loader import load_prices, compute_returns
from portfolio_optimizer.feature_engineering import (
    create_lagged_features,
    compute_rolling_stats,
    compute_rolling_volatility,
    compute_technical_indicators,
    detect_regimes,
    create_momentum_features,
    compute_rsi,
    compute_sma,
)
import pandas as pd
import numpy as np


def example_1_lagged_features():
    """Example 1: Creating lagged features"""
    print("\n" + "="*60)
    print("Example 1: Creating Lagged Features")
    print("="*60)
    
    # Load data
    print("\n1. Loading data...")
    prices = load_prices(['AAPL', 'MSFT'], start='2023-01-01', end='2023-12-31')
    returns = compute_returns(prices)
    print(f"Returns shape: {returns.shape}")
    
    # Create lagged features
    print("\n2. Creating lagged features (1-5 periods)...")
    lagged = create_lagged_features(returns, lags=5)
    print(f"Lagged features shape: {lagged.shape}")
    print(f"New columns created: {lagged.shape[1] - returns.shape[1]}")
    print(f"\nColumn names (first 10):")
    print(lagged.columns.tolist()[:10])
    
    # Specific lags
    print("\n3. Creating specific lags [1, 5, 21]...")
    specific_lagged = create_lagged_features(returns, lags=[1, 5, 21])
    print(f"Columns: {specific_lagged.columns.tolist()[:8]}")
    
    # Show data
    print("\n4. Sample data:")
    print(lagged[['AAPL', 'AAPL_lag1', 'AAPL_lag2', 'AAPL_lag5']].head(10))
    
    return lagged


def example_2_rolling_statistics():
    """Example 2: Computing rolling statistics"""
    print("\n" + "="*60)
    print("Example 2: Rolling Statistics")
    print("="*60)
    
    # Load data
    prices = load_prices(['AAPL'], start='2023-01-01', end='2023-12-31')
    returns = compute_returns(prices)
    
    # Rolling statistics
    print("\n1. Computing 21-day rolling statistics...")
    rolling = compute_rolling_stats(
        returns, 
        window=21,
        stats=['mean', 'std', 'skew', 'kurt']
    )
    print(f"Shape: {rolling.shape}")
    print(f"Columns: {rolling.columns.tolist()}")
    
    print("\n2. Sample rolling statistics:")
    print(rolling.tail())
    
    # Rolling volatility
    print("\n3. Computing rolling volatility...")
    vol = compute_rolling_volatility(returns, window=21, annualize=True)
    print(f"Annualized volatility (last 5 days):")
    print(vol.tail())
    
    # Multiple windows
    print("\n4. Multiple rolling windows...")
    vol_10d = compute_rolling_volatility(returns, window=10, annualize=True)
    vol_21d = compute_rolling_volatility(returns, window=21, annualize=True)
    vol_63d = compute_rolling_volatility(returns, window=63, annualize=True)
    
    vol_comparison = pd.DataFrame({
        '10d_vol': vol_10d['AAPL'],
        '21d_vol': vol_21d['AAPL'],
        '63d_vol': vol_63d['AAPL']
    })
    print("\nVolatility at different windows:")
    print(vol_comparison.tail())
    
    return rolling, vol_comparison


def example_3_technical_indicators():
    """Example 3: Technical indicators"""
    print("\n" + "="*60)
    print("Example 3: Technical Indicators")
    print("="*60)
    
    # Load data
    print("\n1. Loading data...")
    prices = load_prices(['AAPL'], start='2023-01-01', end='2023-12-31')
    
    # Compute all technical indicators
    print("\n2. Computing technical indicators...")
    tech_indicators = compute_technical_indicators(prices)
    print(f"Shape: {tech_indicators.shape}")
    print(f"Columns: {tech_indicators.columns.tolist()}")
    
    # Show sample data
    print("\n3. Sample technical indicators:")
    sample_cols = [col for col in tech_indicators.columns if 'AAPL' in col][:6]
    print(tech_indicators[sample_cols].tail(10))
    
    # Individual indicators
    print("\n4. Individual indicators:")
    
    # SMA
    sma_20 = compute_sma(prices['AAPL'], window=20)
    sma_50 = compute_sma(prices['AAPL'], window=50)
    print(f"\nSMA 20 (last 5): {sma_20.tail().values}")
    print(f"SMA 50 (last 5): {sma_50.tail().values}")
    
    # RSI
    rsi = compute_rsi(prices['AAPL'], window=14)
    print(f"\nRSI (last 5): {rsi.tail().values}")
    print(f"Current RSI: {rsi.iloc[-1]:.2f}")
    
    if rsi.iloc[-1] > 70:
        print("  → Overbought territory!")
    elif rsi.iloc[-1] < 30:
        print("  → Oversold territory!")
    else:
        print("  → Neutral zone")
    
    # Price comparison with SMAs
    print("\n5. Price vs Moving Averages:")
    comparison = pd.DataFrame({
        'Price': prices['AAPL'],
        'SMA_20': sma_20,
        'SMA_50': sma_50
    })
    print(comparison.tail())
    
    return tech_indicators


def example_4_momentum_features():
    """Example 4: Momentum features"""
    print("\n" + "="*60)
    print("Example 4: Momentum Features")
    print("="*60)
    
    # Load data
    prices = load_prices(['AAPL', 'MSFT', 'GOOG'], start='2023-01-01', end='2023-12-31')
    
    # Create momentum features
    print("\n1. Creating momentum features...")
    momentum = create_momentum_features(prices, periods=[5, 10, 21, 63])
    print(f"Shape: {momentum.shape}")
    print(f"Number of features per stock: {momentum.shape[1] // len(prices.columns)}")
    
    # Show momentum at different horizons
    print("\n2. Momentum at different time horizons (AAPL):")
    aapl_momentum = momentum[[col for col in momentum.columns if 'AAPL_momentum' in col]]
    print(aapl_momentum.tail())
    
    # Identify momentum leaders
    print("\n3. Recent momentum performance (last day):")
    latest_momentum = {}
    for ticker in prices.columns:
        mom_21 = momentum[f'{ticker}_momentum_21'].iloc[-1]
        latest_momentum[ticker] = mom_21
    
    momentum_df = pd.DataFrame.from_dict(
        latest_momentum, 
        orient='index', 
        columns=['21-day momentum']
    ).sort_values('21-day momentum', ascending=False)
    
    print(momentum_df)
    print(f"\nMomentum leader: {momentum_df.index[0]}")
    
    # Price relative to SMA
    print("\n4. Price relative to SMA (AAPL):")
    price_to_sma = momentum[[col for col in momentum.columns if 'AAPL_price_to_sma' in col]]
    print(price_to_sma.tail())
    
    return momentum


def example_5_regime_detection():
    """Example 5: Market regime detection"""
    print("\n" + "="*60)
    print("Example 5: Market Regime Detection")
    print("="*60)
    
    # Load data (need more data for regime detection)
    print("\n1. Loading extended dataset...")
    prices = load_prices(['AAPL'], start='2022-01-01', end='2023-12-31')
    returns = compute_returns(prices)
    print(f"Data points: {len(returns)}")
    
    # Detect regimes
    print("\n2. Detecting market regimes (3 regimes: bull, bear, sideways)...")
    regimes = detect_regimes(returns, n_regimes=3, method='kmeans')
    print(f"Shape: {regimes.shape}")
    print(f"Columns: {regimes.columns.tolist()}")
    
    # Analyze regimes
    print("\n3. Regime distribution:")
    regime_counts = regimes['regime'].value_counts()
    print(regime_counts)
    
    # Regime characteristics
    print("\n4. Characteristics of each regime:")
    for regime_id in sorted(regimes['regime'].dropna().unique()):
        regime_mask = regimes['regime'] == regime_id
        regime_returns = returns[regime_mask]
        
        mean_return = regime_returns.mean().values[0]
        volatility = regime_returns.std().values[0]
        
        print(f"\nRegime {int(regime_id)}:")
        print(f"  Observations: {regime_mask.sum()}")
        print(f"  Mean return: {mean_return:.4f} ({mean_return*252*100:.2f}% annualized)")
        print(f"  Volatility: {volatility:.4f} ({volatility*np.sqrt(252)*100:.2f}% annualized)")
        
        if mean_return > 0.001 and volatility < 0.015:
            regime_name = "Bull Market (high return, low vol)"
        elif mean_return < -0.001:
            regime_name = "Bear Market (negative return)"
        else:
            regime_name = "Sideways Market"
        print(f"  Interpretation: {regime_name}")
    
    # Recent regime
    print("\n5. Recent regime transitions:")
    recent_regimes = regimes['regime'].tail(20)
    print(recent_regimes)
    print(f"\nCurrent regime: {int(regimes['regime'].iloc[-1])}")
    
    return regimes


def example_6_full_feature_pipeline():
    """Example 6: Complete feature engineering pipeline"""
    print("\n" + "="*60)
    print("Example 6: Complete Feature Engineering Pipeline")
    print("="*60)
    
    # Load data
    print("\n1. Loading data...")
    tickers = ['AAPL', 'MSFT']
    prices = load_prices(tickers, start='2023-01-01', end='2023-12-31')
    returns = compute_returns(prices)
    print(f"   ✓ {len(prices)} days of price data")
    print(f"   ✓ {len(tickers)} assets")
    
    # Create comprehensive feature set
    print("\n2. Creating feature set...")
    
    # a) Lagged features
    print("   • Lagged features (1-5 days)...")
    lagged = create_lagged_features(returns, lags=5)
    
    # b) Rolling statistics
    print("   • Rolling statistics (21-day window)...")
    rolling_stats = compute_rolling_stats(returns, window=21, stats=['mean', 'std'])
    
    # c) Rolling volatility
    print("   • Rolling volatility (multiple windows)...")
    vol_10 = compute_rolling_volatility(returns, window=10)
    vol_21 = compute_rolling_volatility(returns, window=21)
    vol_10.columns = [f'{col}_vol10' for col in vol_10.columns]
    vol_21.columns = [f'{col}_vol21' for col in vol_21.columns]
    
    # d) Technical indicators
    print("   • Technical indicators...")
    tech_indicators = compute_technical_indicators(prices, indicators=['sma', 'rsi', 'momentum'])
    
    # e) Momentum features
    print("   • Momentum features...")
    momentum = create_momentum_features(prices, periods=[5, 21])
    
    # f) Regime detection
    print("   • Market regimes...")
    regimes = detect_regimes(returns, n_regimes=3)
    
    # Combine all features
    print("\n3. Combining features...")
    feature_set = pd.concat([
        returns,
        lagged,
        rolling_stats,
        vol_10,
        vol_21,
        tech_indicators,
        momentum,
        regimes
    ], axis=1)
    
    # Remove duplicate columns
    feature_set = feature_set.loc[:, ~feature_set.columns.duplicated()]
    
    print(f"   ✓ Total features: {feature_set.shape[1]}")
    print(f"   ✓ Total observations: {feature_set.shape[0]}")
    
    # Feature summary
    print("\n4. Feature summary:")
    print(f"   • Original returns: {len(tickers)}")
    print(f"   • Lagged features: {sum('lag' in col for col in feature_set.columns)}")
    print(f"   • Rolling stats: {sum('roll' in col for col in feature_set.columns)}")
    print(f"   • Volatility: {sum('vol' in col for col in feature_set.columns)}")
    print(f"   • Technical: {sum(any(x in col for x in ['sma', 'rsi', 'ema']) for col in feature_set.columns)}")
    print(f"   • Momentum: {sum('momentum' in col for col in feature_set.columns)}")
    print(f"   • Regime: {sum('regime' in col for col in feature_set.columns)}")
    
    # Data quality
    print("\n5. Data quality:")
    print(f"   • Missing values: {feature_set.isna().sum().sum()}")
    print(f"   • Complete cases: {feature_set.dropna().shape[0]}")
    print(f"   • Completion rate: {feature_set.dropna().shape[0]/len(feature_set)*100:.1f}%")
    
    # Sample features
    print("\n6. Sample features (last 5 rows):")
    sample_cols = ['AAPL', 'AAPL_lag1', 'AAPL_roll_mean_21', 'AAPL_rsi_14', 'regime']
    sample_cols = [col for col in sample_cols if col in feature_set.columns]
    print(feature_set[sample_cols].tail())
    
    print("\n7. Feature set ready for ML model training!")
    
    return feature_set


def main():
    """Run all examples"""
    print("\n" + "="*60)
    print("Portfolio Optimizer - Feature Engineering Examples")
    print("="*60)
    
    try:
        # Run examples
        lagged = example_1_lagged_features()
        rolling, vol = example_2_rolling_statistics()
        tech = example_3_technical_indicators()
        momentum = example_4_momentum_features()
        regimes = example_5_regime_detection()
        feature_set = example_6_full_feature_pipeline()
        
        print("\n" + "="*60)
        print("All examples completed successfully!")
        print("="*60)
        print(f"\nFinal feature set shape: {feature_set.shape}")
        print(f"Features created: {feature_set.shape[1]} columns")
        print(f"Observations: {feature_set.shape[0]} rows")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
