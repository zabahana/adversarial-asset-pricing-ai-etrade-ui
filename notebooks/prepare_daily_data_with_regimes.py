"""
Prepare daily stock data and identify market regimes
"""
import pandas as pd
import numpy as np
import os
import glob

def identify_market_regime(df, window=252):
    """
    Identify market regime: BULL, BEAR, or SIDEWAYS
    Based on 1-year rolling return
    """
    df['rolling_return'] = df['Close'].pct_change(window)
    
    # Calculate trend strength
    df['trend'] = df['Close'].rolling(window=50).mean()
    df['volatility'] = df['Close'].pct_change().rolling(window=20).std()
    
    # Classify regimes
    conditions = [
        (df['rolling_return'] > 0.15),  # Bull: >15% annual return
        (df['rolling_return'] < -0.10),  # Bear: <-10% annual return
    ]
    choices = ['BULL', 'BEAR']
    df['regime'] = np.select(conditions, choices, default='SIDEWAYS')
    
    return df

def analyze_stock_regimes(symbol):
    """Analyze market regimes for a stock"""
    filepath = f'stock_data/{symbol}.csv'
    
    if not os.path.exists(filepath):
        return None
    
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'], utc=True)
    df = df.sort_values('Date')
    
    # Identify regimes
    df = identify_market_regime(df)
    
    # Count regime periods
    regime_counts = df['regime'].value_counts()
    total_days = len(df)
    
    return {
        'symbol': symbol,
        'total_days': total_days,
        'bull_days': regime_counts.get('BULL', 0),
        'bear_days': regime_counts.get('BEAR', 0),
        'sideways_days': regime_counts.get('SIDEWAYS', 0),
        'bull_pct': regime_counts.get('BULL', 0) / total_days * 100,
        'bear_pct': regime_counts.get('BEAR', 0) / total_days * 100,
        'sideways_pct': regime_counts.get('SIDEWAYS', 0) / total_days * 100,
    }

print("="*80)
print("ANALYZING DAILY DATA - MARKET REGIMES")
print("="*80)

csv_files = glob.glob('stock_data/*.csv')
stocks = sorted([os.path.basename(f).replace('.csv', '') for f in csv_files])

print(f"\n📊 Analyzing {len(stocks)} stocks...")
print("="*80)

all_results = []

for symbol in stocks:
    result = analyze_stock_regimes(symbol)
    if result:
        all_results.append(result)
        print(f"{result['symbol']:6s}: {result['total_days']:4d} days | "
              f"Bull:{result['bull_pct']:5.1f}% | "
              f"Bear:{result['bear_pct']:5.1f}% | "
              f"Sideways:{result['sideways_pct']:5.1f}%")

# Find best stocks for each regime
results_df = pd.DataFrame(all_results)

print("\n" + "="*80)
print("BEST STOCKS FOR TESTING BY REGIME")
print("="*80)

print("\n🟢 MOST BULL MARKET DAYS:")
bull_stocks = results_df.nlargest(5, 'bull_pct')[['symbol', 'bull_pct']]
for _, row in bull_stocks.iterrows():
    print(f"   {row['symbol']}: {row['bull_pct']:.1f}%")

print("\n🔴 MOST BEAR MARKET DAYS:")
bear_stocks = results_df.nlargest(5, 'bear_pct')[['symbol', 'bear_pct']]
for _, row in bear_stocks.iterrows():
    print(f"   {row['symbol']}: {row['bear_pct']:.1f}%")

print("\n🟡 MOST SIDEWAYS MARKET DAYS:")
sideways_stocks = results_df.nlargest(5, 'sideways_pct')[['symbol', 'sideways_pct']]
for _, row in sideways_stocks.iterrows():
    print(f"   {row['symbol']}: {row['sideways_pct']:.1f}%")

print("\n" + "="*80)
print("✓ DAILY DATA ANALYSIS COMPLETE")
print("="*80)
