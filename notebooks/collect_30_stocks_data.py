"""
Collect data for 30 stocks across 3 phases
"""
from google.cloud import bigquery
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import time

PROJECT_ID = "ambient-isotope-463716-u6"

# 30 stocks organized by phase
PHASE_1 = ['AAPL', 'MSFT', 'JPM', 'JNJ', 'WMT', 'NVDA', 'AMZN', 'V', 'UNH', 'XOM']
PHASE_2 = ['GOOGL', 'META', 'TSLA', 'BAC', 'GS', 'MA', 'PFE', 'LLY', 'HD', 'PG']
PHASE_3 = ['ORCL', 'NKE', 'DIS', 'NFLX', 'CVX', 'CAT', 'BA', 'NEE', 'PLD', 'NEM']

ALL_STOCKS = PHASE_1 + PHASE_2 + PHASE_3

print("="*80)
print("COLLECTING DATA FOR 30 STOCKS")
print("="*80)
print(f"\nPhase 1 (10 stocks): {', '.join(PHASE_1)}")
print(f"Phase 2 (10 stocks): {', '.join(PHASE_2)}")
print(f"Phase 3 (10 stocks): {', '.join(PHASE_3)}")
print(f"\nTotal: {len(ALL_STOCKS)} stocks")
print("="*80)

client = bigquery.Client(project=PROJECT_ID)

# Check which stocks already exist
query = """
SELECT DISTINCT symbol 
FROM `ambient-isotope-463716-u6.processed_market_data.technical_indicators`
"""
existing_df = client.query(query).to_dataframe()
existing_stocks = set(existing_df['symbol'].tolist()) if len(existing_df) > 0 else set()

print(f"\n✓ Already have data for: {', '.join(sorted(existing_stocks))}")

# Stocks to collect
to_collect = [s for s in ALL_STOCKS if s not in existing_stocks]
print(f"\n📥 Need to collect: {len(to_collect)} stocks")
print(f"   {', '.join(to_collect)}")

if len(to_collect) == 0:
    print("\n✓ All 30 stocks already in database!")
else:
    print(f"\n⏱️  Estimated time: ~{len(to_collect) * 2} minutes")
    
    response = input("\nProceed with data collection? (yes/no): ")
    if response.lower() != 'yes':
        print("Cancelled.")
        exit()
    
    # Collect data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=3650)  # 10 years
    
    collected = 0
    failed = []
    
    for symbol in to_collect:
        try:
            print(f"\n[{collected+1}/{len(to_collect)}] Downloading {symbol}...")
            
            # Download data
            df = yf.download(symbol, start=start_date, end=end_date, progress=False)
            
            if len(df) == 0:
                print(f"  ⚠️  No data returned for {symbol}")
                failed.append(symbol)
                continue
            
            # Prepare data
            df = df.reset_index()
            df['symbol'] = symbol
            df['timestamp'] = pd.to_datetime(df['Date'])
            
            # Calculate technical indicators
            df['close_price'] = df['Close']
            df['volume'] = df['Volume']
            
            # SMAs
            df['sma_5'] = df['close_price'].rolling(window=5).mean()
            df['sma_10'] = df['close_price'].rolling(window=10).mean()
            df['sma_20'] = df['close_price'].rolling(window=20).mean()
            df['sma_50'] = df['close_price'].rolling(window=50).mean()
            
            # EMAs
            df['ema_12'] = df['close_price'].ewm(span=12).mean()
            df['ema_26'] = df['close_price'].ewm(span=26).mean()
            
            # MACD
            df['macd'] = df['ema_12'] - df['ema_26']
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            # Bollinger Bands
            df['bb_middle'] = df['sma_20']
            bb_std = df['close_price'].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + (2 * bb_std)
            df['bb_lower'] = df['bb_middle'] - (2 * bb_std)
            
            # RSI
            delta = df['close_price'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi_14'] = 100 - (100 / (1 + rs))
            
            gain_30 = delta.where(delta > 0, 0).rolling(window=30).mean()
            loss_30 = (-delta.where(delta < 0, 0)).rolling(window=30).mean()
            rs_30 = gain_30 / loss_30
            df['rsi_30'] = 100 - (100 / (1 + rs_30))
            
            # Price changes
            df['price_change_1d'] = df['close_price'].pct_change() * 100
            df['price_change_5d'] = df['close_price'].pct_change(5) * 100
            
            # Volatility
            df['volatility_10d'] = df['close_price'].rolling(window=10).std()
            df['volatility_20d'] = df['close_price'].rolling(window=20).std()
            
            df['processing_timestamp'] = datetime.now()
            
            # Select columns for upload
            upload_cols = [
                'symbol', 'timestamp', 'close_price', 'volume',
                'rsi_14', 'rsi_30', 'macd', 'macd_signal', 'macd_histogram',
                'bb_upper', 'bb_middle', 'bb_lower',
                'sma_20', 'sma_50', 'ema_12', 'ema_26',
                'price_change_1d', 'price_change_5d',
                'volatility_20d', 'processing_timestamp',
                'sma_5', 'sma_10', 'volatility_10d'
            ]
            
            upload_df = df[upload_cols].copy()
            upload_df = upload_df.dropna(subset=['close_price'])
            
            # Upload to BigQuery
            table_id = f"{PROJECT_ID}.processed_market_data.technical_indicators"
            job_config = bigquery.LoadJobConfig(
                write_disposition="WRITE_APPEND",
            )
            
            job = client.load_table_from_dataframe(
                upload_df, table_id, job_config=job_config
            )
            job.result()
            
            print(f"  ✓ {symbol}: {len(upload_df)} rows uploaded")
            collected += 1
            
            # Rate limiting
            time.sleep(1)
            
        except Exception as e:
            print(f"  ❌ {symbol} failed: {str(e)}")
            failed.append(symbol)
    
    print("\n" + "="*80)
    print("DATA COLLECTION COMPLETE")
    print("="*80)
    print(f"✓ Successfully collected: {collected}/{len(to_collect)} stocks")
    if failed:
        print(f"❌ Failed: {len(failed)} stocks - {', '.join(failed)}")
    print("="*80)

# Final verification
query = """
SELECT symbol, COUNT(*) as row_count
FROM `ambient-isotope-463716-u6.processed_market_data.technical_indicators`
GROUP BY symbol
ORDER BY symbol
"""
final_df = client.query(query).to_dataframe()

print("\n📊 CURRENT DATABASE STATUS:")
print("="*80)
print(final_df.to_string(index=False))
print("="*80)
print(f"\nTotal stocks in database: {len(final_df)}/{len(ALL_STOCKS)}")

if len(final_df) == len(ALL_STOCKS):
    print("\n✅ All 30 stocks ready for comparison!")
else:
    missing = set(ALL_STOCKS) - set(final_df['symbol'].tolist())
    print(f"\n⚠️  Missing {len(missing)} stocks: {', '.join(sorted(missing))}")
