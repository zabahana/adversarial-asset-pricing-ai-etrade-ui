import yfinance as yf
import pandas as pd
import numpy as np
import json
from datetime import datetime
from google.cloud import pubsub_v1, bigquery
import time

class EnhancedMarketDataCollector:
    """
    Collect monthly aggregated data for 10 stocks with advanced financial metrics
    """
    
    def __init__(self, project_id):
        self.project_id = project_id
        self.publisher = pubsub_v1.PublisherClient()
        self.bq_client = bigquery.Client(project=project_id)
        self.topic_path = self.publisher.topic_path(project_id, "market-data-raw")
        
        # 10 diverse stocks across sectors
        self.symbols = [
            "AAPL",   # Technology
            "GOOGL",  # Technology
            "MSFT",   # Technology
            "TSLA",   # Automotive
            "NVDA",   # Semiconductors
            "JPM",    # Financial
            "JNJ",    # Healthcare
            "XOM",    # Energy
            "WMT",    # Retail
            "DIS"     # Entertainment
        ]
    
    def fetch_monthly_data(self, symbol, period="10y"):
        """
        Fetch daily data and aggregate to monthly
        """
        print(f"\nFetching data for {symbol}...")
        
        try:
            ticker = yf.Ticker(symbol)
            # Get daily data first
            daily_data = ticker.history(period=period)
            
            if daily_data.empty:
                print(f"  ✗ No data for {symbol}")
                return None
            
            # Resample to monthly (end of month)
            monthly_data = pd.DataFrame({
                'open': daily_data['Open'].resample('ME').first(),
                'high': daily_data['High'].resample('ME').max(),
                'low': daily_data['Low'].resample('ME').min(),
                'close': daily_data['Close'].resample('ME').last(),
                'volume': daily_data['Volume'].resample('ME').sum(),
                'adj_close': daily_data['Close'].resample('ME').last()
            }).dropna()
            
            # Calculate monthly returns
            monthly_data['monthly_return'] = monthly_data['close'].pct_change()
            
            # Calculate volatility (std of daily returns in that month)
            daily_data['daily_return'] = daily_data['Close'].pct_change()
            monthly_volatility = daily_data['daily_return'].resample('ME').std()
            monthly_data['volatility'] = monthly_volatility
            
            monthly_data['symbol'] = symbol
            monthly_data.reset_index(inplace=True)
            monthly_data.rename(columns={'Date': 'timestamp'}, inplace=True)
            
            print(f"  ✓ Collected {len(monthly_data)} monthly records")
            print(f"    Range: {monthly_data['timestamp'].min().strftime('%Y-%m')} to "
                  f"{monthly_data['timestamp'].max().strftime('%Y-%m')}")
            
            return monthly_data
            
        except Exception as e:
            print(f"  ✗ Error fetching {symbol}: {e}")
            return None
    
    def calculate_advanced_metrics(self, df, lookback_periods=[3, 6, 12]):
        """
        Calculate advanced financial metrics WITHOUT data leakage
        Uses only past data (no future information)
        """
        df = df.sort_values('timestamp').copy()
        
        # Simple moving averages (lagged to avoid leakage)
        for period in lookback_periods:
            df[f'sma_{period}'] = df['close'].shift(1).rolling(window=period).mean()
        
        # Momentum indicators (lagged)
        df['momentum_3m'] = (df['close'].shift(1) - df['close'].shift(4)) / df['close'].shift(4)
        df['momentum_6m'] = (df['close'].shift(1) - df['close'].shift(7)) / df['close'].shift(7)
        df['momentum_12m'] = (df['close'].shift(1) - df['close'].shift(13)) / df['close'].shift(13)
        
        # Rolling Sharpe Ratio (12-month, risk-free rate = 2% annual)
        rf_monthly = 0.02 / 12
        excess_returns = df['monthly_return'] - rf_monthly
        df['sharpe_12m'] = (
            excess_returns.shift(1).rolling(window=12).mean() / 
            excess_returns.shift(1).rolling(window=12).std()
        )
        
        # Maximum Drawdown (12-month rolling)
        def calculate_max_drawdown(prices):
            cumulative = (1 + prices).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            return drawdown.min()
        
        df['max_drawdown_12m'] = df['monthly_return'].shift(1).rolling(
            window=12
        ).apply(calculate_max_drawdown, raw=False)
        
        # Volatility ratios
        df['vol_3m'] = df['monthly_return'].shift(1).rolling(window=3).std()
        df['vol_12m'] = df['monthly_return'].shift(1).rolling(window=12).std()
        df['vol_ratio'] = df['vol_3m'] / df['vol_12m']
        
        # Risk-adjusted return
        df['return_to_vol'] = df['monthly_return'].shift(1) / (df['volatility'].shift(1) + 1e-6)
        
        # Sortino Ratio (downside deviation)
        downside_returns = df['monthly_return'].shift(1).copy()
        downside_returns[downside_returns > 0] = 0
        df['sortino_12m'] = (
            excess_returns.shift(1).rolling(window=12).mean() /
            downside_returns.rolling(window=12).std()
        )
        
        return df
    
    def create_monthly_dataset_table(self):
        """Create BigQuery table for monthly data"""
        dataset_id = f"{self.project_id}.monthly_market_data"
        
        # Create dataset if not exists
        dataset = bigquery.Dataset(dataset_id)
        dataset.location = "US"
        try:
            self.bq_client.create_dataset(dataset, exists_ok=True)
            print(f"✓ Dataset {dataset_id} ready")
        except Exception as e:
            print(f"Dataset creation: {e}")
        
        # Define schema for monthly data
        schema = [
            bigquery.SchemaField("symbol", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("timestamp", "TIMESTAMP", mode="REQUIRED"),
            bigquery.SchemaField("open_price", "FLOAT64"),
            bigquery.SchemaField("high_price", "FLOAT64"),
            bigquery.SchemaField("low_price", "FLOAT64"),
            bigquery.SchemaField("close_price", "FLOAT64"),
            bigquery.SchemaField("volume", "INTEGER"),
            bigquery.SchemaField("monthly_return", "FLOAT64"),
            bigquery.SchemaField("volatility", "FLOAT64"),
            bigquery.SchemaField("sma_3", "FLOAT64"),
            bigquery.SchemaField("sma_6", "FLOAT64"),
            bigquery.SchemaField("sma_12", "FLOAT64"),
            bigquery.SchemaField("momentum_3m", "FLOAT64"),
            bigquery.SchemaField("momentum_6m", "FLOAT64"),
            bigquery.SchemaField("momentum_12m", "FLOAT64"),
            bigquery.SchemaField("sharpe_12m", "FLOAT64"),
            bigquery.SchemaField("max_drawdown_12m", "FLOAT64"),
            bigquery.SchemaField("vol_ratio", "FLOAT64"),
            bigquery.SchemaField("return_to_vol", "FLOAT64"),
            bigquery.SchemaField("sortino_12m", "FLOAT64"),
            bigquery.SchemaField("ingestion_timestamp", "TIMESTAMP"),
        ]
        
        table_id = f"{dataset_id}.monthly_features"
        table = bigquery.Table(table_id, schema=schema)
        
        try:
            self.bq_client.create_table(table, exists_ok=True)
            print(f"✓ Table {table_id} ready")
        except Exception as e:
            print(f"Table creation: {e}")
        
        return table_id
    
    def save_to_bigquery(self, df, table_id):
        """Save processed data directly to BigQuery"""
        
        # Prepare data for BigQuery
        df_to_insert = df.copy()
        df_to_insert['ingestion_timestamp'] = datetime.utcnow()
        
        # Rename columns to match schema
        df_to_insert = df_to_insert.rename(columns={
            'open': 'open_price',
            'high': 'high_price',
            'low': 'low_price',
            'close': 'close_price'
        })
        
        # Select only columns that exist in schema
        columns_to_insert = [
            'symbol', 'timestamp', 'open_price', 'high_price', 'low_price',
            'close_price', 'volume', 'monthly_return', 'volatility',
            'sma_3', 'sma_6', 'sma_12', 'momentum_3m', 'momentum_6m', 
            'momentum_12m', 'sharpe_12m', 'max_drawdown_12m', 'vol_ratio',
            'return_to_vol', 'sortino_12m', 'ingestion_timestamp'
        ]
        
        df_to_insert = df_to_insert[columns_to_insert].copy()
        
        # Drop rows with NaN in critical columns
        df_to_insert = df_to_insert.dropna(subset=['close_price', 'volume'])
        
        # Insert into BigQuery
        job_config = bigquery.LoadJobConfig(
            write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
        )
        
        job = self.bq_client.load_table_from_dataframe(
            df_to_insert, table_id, job_config=job_config
        )
        job.result()  # Wait for completion
        
        return len(df_to_insert)
    
    def collect_all_stocks(self):
        """Collect and process data for all stocks"""
        
        print("=" * 80)
        print("ENHANCED MONTHLY DATA COLLECTION - 10 STOCKS")
        print("=" * 80)
        print(f"Stocks: {', '.join(self.symbols)}")
        print("Period: Last 10 years")
        print("Frequency: Monthly")
        print("=" * 80)
        
        # Create table
        table_id = self.create_monthly_dataset_table()
        
        all_data = []
        
        for symbol in self.symbols:
            # Fetch monthly data
            monthly_df = self.fetch_monthly_data(symbol, period="10y")
            
            if monthly_df is not None:
                # Calculate advanced metrics
                print(f"  → Calculating financial metrics for {symbol}...")
                enhanced_df = self.calculate_advanced_metrics(monthly_df)
                
                # Save to BigQuery
                print(f"  → Saving to BigQuery...")
                saved_count = self.save_to_bigquery(enhanced_df, table_id)
                print(f"  ✓ Saved {saved_count} records for {symbol}")
                
                all_data.append(enhanced_df)
                
                # Rate limiting
                time.sleep(2)
        
        # Summary
        print("\n" + "=" * 80)
        print("COLLECTION COMPLETE")
        print("=" * 80)
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            print(f"Total records collected: {len(combined_df):,}")
            print(f"Stocks: {combined_df['symbol'].nunique()}")
            print(f"Date range: {combined_df['timestamp'].min().strftime('%Y-%m')} to "
                  f"{combined_df['timestamp'].max().strftime('%Y-%m')}")
            print(f"\nFeatures calculated:")
            print("  • Simple Moving Averages (3, 6, 12 months)")
            print("  • Momentum (3, 6, 12 months)")
            print("  • Rolling Sharpe Ratio (12 months)")
            print("  • Maximum Drawdown (12 months)")
            print("  • Sortino Ratio (12 months)")
            print("  • Volatility metrics")
            print("  • Risk-adjusted returns")
            print("\n⚠️  All features are properly lagged to avoid data leakage!")
            print("=" * 80)
            
            return combined_df
        else:
            print("No data collected")
            return None

if __name__ == "__main__":
    PROJECT_ID = "ambient-isotope-463716-u6"
    
    collector = EnhancedMarketDataCollector(PROJECT_ID)
    data = collector.collect_all_stocks()
    
    if data is not None:
        print("\n✓ Data ready for time-series cross-validation!")
        print("  Next step: Create train/validation/test splits with walk-forward validation")

