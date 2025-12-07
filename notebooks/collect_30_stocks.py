"""
Collect comprehensive 30-stock portfolio data
"""
import yfinance as yf
import pandas as pd
import numpy as np
from google.cloud import bigquery
from datetime import datetime
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
    
    def collect_stock_data(self, symbol, period="10y"):
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
            df = df.rename(columns={
                'Date': 'timestamp',
                'Open': 'open_price',
                'High': 'high_price',
                'Low': 'low_price',
                'Close': 'close_price',
                'Volume': 'volume'
            })
            
            # Add basic features
            df['daily_return'] = df['close_price'].pct_change()
            df['volatility_20d'] = df['daily_return'].rolling(20).std()
            df['sma_5'] = df['close_price'].rolling(5).mean()
            df['sma_10'] = df['close_price'].rolling(10).mean()
            df['sma_20'] = df['close_price'].rolling(20).mean()
            df['sma_50'] = df['close_price'].rolling(50).mean()
            df['price_change_1d'] = df['close_price'].pct_change() * 100
            df['price_change_5d'] = df['close_price'].pct_change(5) * 100
            
            # Advanced features
            df['momentum_10d'] = df['close_price'] - df['close_price'].shift(10)
            df['rsi_14'] = self._calculate_rsi(df['close_price'], 14)
            df['volatility_10d'] = df['daily_return'].rolling(10).std()
            
            # Sharpe ratio (rolling 20-day)
            rf_daily = 0.02 / 252
            excess_returns = df['daily_return'] - rf_daily
            df['sharpe_20d'] = (
                excess_returns.rolling(20).mean() / 
                df['daily_return'].rolling(20).std()
            ) * np.sqrt(252)
            
            df['ingestion_timestamp'] = datetime.now(datetime.timezone.utc)
            
            print(f"✓ {len(df)} records ({df['timestamp'].min().strftime('%Y-%m-%d')} to {df['timestamp'].max().strftime('%Y-%m-%d')})")
            return df
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return None
    
    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def save_to_bigquery(self, df, table_name='technical_indicators'):
        """Save to BigQuery"""
        if df is None or len(df) == 0:
            return 0
        
        dataset_id = f"{self.project_id}.processed_market_data"
        table_id = f"{dataset_id}.{table_name}"
        
        # Ensure dataset exists
        dataset = bigquery.Dataset(dataset_id)
        dataset.location = "US"
        self.client.create_dataset(dataset, exists_ok=True)
        
        # Select columns for BigQuery
        cols_to_save = [
            'symbol', 'timestamp', 'open_price', 'high_price', 'low_price',
            'close_price', 'volume', 'daily_return', 'volatility_20d',
            'sma_5', 'sma_10', 'sma_20', 'sma_50', 'price_change_1d',
            'price_change_5d', 'momentum_10d', 'rsi_14', 'volatility_10d',
            'sharpe_20d', 'ingestion_timestamp'
        ]
        
        # Add missing field that BigQuery table expects
        df_to_save = df[cols_to_save].copy()
        df_to_save['processing_timestamp'] = df_to_save['ingestion_timestamp']
        
        
        # Load to BigQuery
        job_config = bigquery.LoadJobConfig(
            write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
        )
        
        job = self.client.load_table_from_dataframe(
            df_to_save, table_id, job_config=job_config
        )
        job.result()
        
        return len(df_to_save)
    
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
                    saved = self.save_to_bigquery(df)
                    print(f"   → Saved {saved} records to BigQuery")
                    
                    results[symbol] = {
                        'sector': sector,
                        'records': len(df),
                        'date_range': (df['timestamp'].min(), df['timestamp'].max()),
                        'price_range': (df['close_price'].min(), df['close_price'].max())
                    }
                    success_count += 1
                    total_records += len(df)
                
                time.sleep(1)  # Rate limiting
        
        # Summary
        print("\n" + "=" * 80)
        print("COLLECTION SUMMARY")
        print("=" * 80)
        print(f"Successful: {success_count}/{len(all_stocks)} stocks")
        print(f"Total records: {total_records:,}")
        print(f"Average per stock: {total_records/success_count:.0f}")
        
        print("\n" + "=" * 80)
        print("BY SECTOR:")
        print("=" * 80)
        for sector in PORTFOLIO_30.keys():
            sector_stocks = [s for s in results if results[s]['sector'] == sector]
            sector_records = sum(results[s]['records'] for s in sector_stocks)
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
    print("   Next: python train_attention_30stocks.py")
