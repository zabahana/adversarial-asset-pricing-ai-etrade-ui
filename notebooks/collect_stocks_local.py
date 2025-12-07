"""
Collect 30 stocks and save locally (no BigQuery)
"""
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os

# 30 stocks
ALL_STOCKS = [
    'AAPL', 'MSFT', 'JPM', 'JNJ', 'WMT', 'NVDA', 'AMZN', 'V', 'UNH', 'XOM',
    'GOOGL', 'META', 'TSLA', 'BAC', 'GS', 'MA', 'PFE', 'LLY', 'HD', 'PG',
    'ORCL', 'NKE', 'DIS', 'NFLX', 'CVX', 'CAT', 'BA', 'NEE', 'PLD', 'NEM'
]

# Create data directory
os.makedirs('stock_data', exist_ok=True)

print("="*80)
print("DOWNLOADING 30 STOCKS TO LOCAL CSV FILES")
print("="*80)

end_date = datetime.now()
start_date = end_date - timedelta(days=3650)

collected = []

for i, symbol in enumerate(ALL_STOCKS, 1):
    try:
        print(f"[{i}/30] {symbol}...", end=' ', flush=True)
        
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date, auto_adjust=False)
        
        if len(df) > 0:
            df.to_csv(f'stock_data/{symbol}.csv')
            print(f"✓ {len(df)} days")
            collected.append(symbol)
        else:
            print("❌ No data")
            
    except Exception as e:
        print(f"❌ {str(e)[:40]}")

print("\n" + "="*80)
print(f"✓ Downloaded: {len(collected)}/30 stocks")
print(f"📁 Saved in: ./stock_data/")
print("="*80)
