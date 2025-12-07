"""
Collect comprehensive 30-stock portfolio data with correct schema
"""
import yfinance as yf
import pandas as pd
import numpy as np
from google.cloud import bigquery
from datetime import datetime, timezone
import time

# 30-Stock Comprehensive Portfolio
PORTFOLIO_30 = {
    'Technology': ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META', 'TSLA', 'ORCL'],
    'Financial': ['JPM', 'BAC', 'GS', 'V', 'MA'],
    'Healthcare': ['JNJ', 'UNH', 'PFE', 'LLY'],
    'Consumer Discretionary': ['AMZN', 'HD', 'NKE'],
    'Consumer Staples': ['WMT', 'PG'],
    'Energy': ['XOM', 'CVX'],
    'Industrial': ['CAT', 'BA'],
    'Communication': ['DIS', 'NFLX'],
    'Utilities': ['NEE'],
    'Real Estate & Materials': ['PLD', 'NEM']
}

class Portfolio30Collector:
    def __init__(self, project_id):
        self.project_id = project_id
        self.client = bigquery.Client(project=project_id)
    
    def collect_stock_data(self, symbol, period="max"):
        """Collect data for a single stock"""
        try:
            print(f"\n📊 Collecting {symbol}...", end=" ")
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period)
            
            if df.empty:
                print(f"❌ No data")
                return None
            
            # Prepare data
            df = df.reset_index()
            df['symbol'] = symbol
            df = df.rename(columns={'Date': 'timestamp'})
            
            # Keep original column names that match BigQuery schema
            df['close_price'] = df['Close']
            df['open_price'] = df['Open']
            df['high_price'] = df['High']
            df['low_price'] = df['Low']
            df['volume'] = df['Volume'].astype('Int64')
            
            # Calculate features
            df['price_change_1d'] = df['close_price'].pct_change() * 100
            df['price_change_5d'] = df['close_price'].pct_change(5) * 100
            df['volatility_10d'] = df['close_price'].pct_change().rolling(10).std() * 100
            df['sma_5'] = df['close_price'].rolling(5).mean()
            df['sma_10'] = df['close_price'].rolling(10).mean()
            df['sma_20'] = df['close_price'].rolling(20).mean()
            
            # Data source
            df['data_source'] = 'yfinance_30stock_collection'
            
            print(f"✓ {len(df)} records ({df['timestamp'].min().strftime('%Y-%m-%d')} to {df['timestamp'].max().strftime('%Y-%m-%d')})")
            return df
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return None
    
    def save_to_bigquery_batch(self, df):
        """Save using insert_rows (more compatible)"""
        if df is None or len(df) == 0:
            return 0
        
        table_id = f"{self.project_id}.processed_market_data.technical_indicators"
        
        # Prepare rows for insertion
        rows_to_insert = []
        for _, row in df.iterrows():
            row_dict = {
                'symbol': row['symbol'],
                'timestamp': row['timestamp'].isoformat() if pd.notna(row['timestamp']) else None,
                'close_price': float(row['close_price']) if pd.notna(row['close_price']) else None,
                'volume': int(row['volume']) if pd.notna(row['volume']) else None,
                'sma_5': float(row['sma_5']) if pd.notna(row['sma_5']) else None,
                'sma_10': float(row['sma_10']) if pd.notna(row['sma_10']) else None,
                'sma_20': float(row['sma_20']) if pd.notna(row['sma_20']) else None,
                'price_change_1d': float(row['price_change_1d']) if pd.notna(row['price_change_1d']) else None,
                'price_change_5d': float(row['price_change_5d']) if pd.notna(row['price_change_5d']) else None,
                'volatility_10d': float(row['volatility_10d']) if pd.notna(row['volatility_10d']) else None,
                'data_source': row.get('data_source', 'yfinance')
            }
            rows_to_insert.append(row_dict)
        
        # Insert in batches of 1000
        table = self.client.get_table(table_id)
        total_inserted = 0
        
        for i in range(0, len(rows_to_insert), 1000):
            batch = rows_to_insert[i:i+1000]
            errors = self.client.insert_rows_json(table, batch)
            
            if errors:
                print(f"\n   ⚠️  Insertion errors: {errors[:3]}")  # Show first 3 errors
            else:
                total_inserted += len(batch)
        
        return total_inserted
    
    def collect_all_stocks(self):
        """Collect all 30 stocks"""
        print("=" * 80)
        print("COLLECTING 30-STOCK COMPREHENSIVE PORTFOLIO")
        print("=" * 80)
        
        all_stocks = [stock for sector in PORTFOLIO_30.values() for stock in sector]
        
        print(f"\nTotal stocks to collect: {len(all_stocks)}")
        print(f"Estimated time: ~{len(all_stocks) * 0.5:.1f} minutes\n")
        
        results = {}
        success_count = 0
        total_records = 0
        
        for sector, stocks in PORTFOLIO_30.items():
            print(f"\n{'='*80}")
            print(f"SECTOR: {sector.upper()}")
            print('='*80)
            
            for symbol in stocks:
                df = self.collect_stock_data(symbol, period="max")
                
                if df is not None:
                    # Save to BigQuery
                    saved = self.save_to_bigquery_batch(df)
                    print(f"   → Saved {saved} records to BigQuery")
                    
                    results[symbol] = {
                        'sector': sector,
                        'records': len(df),
                        'saved': saved,
                        'date_range': (df['timestamp'].min(), df['timestamp'].max()),
                        'price_range': (df['close_price'].min(), df['close_price'].max())
                    }
                    success_count += 1
                    total_records += saved
                
                time.sleep(1)  # Rate limiting
        
        # Summary
        print("\n" + "=" * 80)
        print("COLLECTION SUMMARY")
        print("=" * 80)
        print(f"Successful: {success_count}/{len(all_stocks)} stocks")
        print(f"Total records saved: {total_records:,}")
        print(f"Average per stock: {total_records/success_count:.0f}")
        
        print("\n" + "=" * 80)
        print("BY SECTOR:")
        print("=" * 80)
        for sector in PORTFOLIO_30.keys():
            sector_stocks = [s for s in results if results[s]['sector'] == sector]
            sector_records = sum(results[s]['saved'] for s in sector_stocks)
            print(f"{sector:25s}: {len(sector_stocks):2d} stocks | {sector_records:6,} records")
        
        print("\n" + "=" * 80)
        print("✓ DATA COLLECTION COMPLETE!")
        print("=" * 80)
        
        return results

if __name__ == "__main__":
    PROJECT_ID = "ambient-isotope-463716-u6"
    
    collector = Portfolio30Collector(PROJECT_ID)
    results = collector.collect_all_stocks()
    
    print("\n📊 Ready for DQN training on 30 stocks!")
    print("   Next: Train attention-enhanced DQN on all sectors")
