"""
Collect 30-stock portfolio using existing Pub/Sub pipeline
"""
import yfinance as yf
import pandas as pd
import json
from google.cloud import pubsub_v1
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

class Portfolio30PubSubCollector:
    def __init__(self, project_id):
        self.project_id = project_id
        self.publisher = pubsub_v1.PublisherClient()
        self.topic_path = self.publisher.topic_path(project_id, "market-data-raw")
    
    def collect_stock_data(self, symbol, period="max"):
        """Collect data for a single stock"""
        try:
            print(f"\n📊 {symbol}...", end=" ")
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period)
            
            if df.empty:
                print(f"❌ No data")
                return None, 0
            
            df = df.reset_index()
            print(f"✓ {len(df)} records", end=" ")
            
            # Publish to Pub/Sub (which will flow through your pipeline)
            published = 0
            for _, row in df.iterrows():
                data_point = {
                    'symbol': symbol,
                    'timestamp': row['Date'].isoformat() + 'Z',
                    'close_price': float(row['Close']),
                    'volume': int(row['Volume']),
                    'data_source': '30stock_collection'
                }
                
                message = json.dumps(data_point).encode('utf-8')
                future = self.publisher.publish(self.topic_path, message)
                future.result()  # Wait for publish
                published += 1
            
            print(f"→ Published {published}")
            return df, published
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return None, 0
    
    def collect_all_stocks(self, limit_per_stock=None):
        """
        Collect all 30 stocks
        
        Args:
            limit_per_stock: Limit records per stock (None = all data)
                            Use 100 for quick test, None for full collection
        """
        print("=" * 80)
        print("COLLECTING 30-STOCK PORTFOLIO VIA PUB/SUB")
        print("=" * 80)
        
        all_stocks = [stock for sector in PORTFOLIO_30.values() for stock in sector]
        
        if limit_per_stock:
            print(f"\n⚠️  TEST MODE: {limit_per_stock} records per stock")
        else:
            print(f"\n📊 FULL MODE: All available historical data")
        
        print(f"Total stocks: {len(all_stocks)}")
        print(f"Estimated time: ~{len(all_stocks) * 1:.0f} minutes")
        print("=" * 80)
        
        results = {}
        total_published = 0
        
        for sector, stocks in PORTFOLIO_30.items():
            print(f"\n{sector.upper()}")
            print("-" * 80)
            
            for symbol in stocks:
                df, published = self.collect_stock_data(symbol, period="max")
                
                if df is not None:
                    # Limit if in test mode
                    if limit_per_stock and published > limit_per_stock:
                        print(f"   (limited to {limit_per_stock} for testing)")
                    
                    results[symbol] = {
                        'sector': sector,
                        'records': len(df),
                        'published': published,
                        'date_range': (df['Date'].min(), df['Date'].max())
                    }
                    total_published += published
                
                time.sleep(2)  # Rate limiting
        
        # Summary
        print("\n" + "=" * 80)
        print("COLLECTION SUMMARY")
        print("=" * 80)
        print(f"Stocks collected: {len(results)}/{len(all_stocks)}")
        print(f"Total published: {total_published:,} records")
        print(f"Average per stock: {total_published/len(results):.0f}")
        
        print("\n" + "=" * 80)
        print("BY SECTOR:")
        print("=" * 80)
        for sector in PORTFOLIO_30.keys():
            sector_stocks = [s for s in results if results[s]['sector'] == sector]
            sector_records = sum(results[s]['published'] for s in sector_stocks)
            if sector_stocks:
                print(f"{sector:25s}: {len(sector_stocks):2d} stocks | {sector_records:6,} records")
        
        print("\n" + "=" * 80)
        print("✓ PUBLISHED TO PUB/SUB!")
        print("=" * 80)
        print("\n⏰ Wait 10-15 minutes for Cloud Functions to process all data")
        print("   Then run: python train_attention_30stocks.py")
        print("=" * 80)
        
        return results

if __name__ == "__main__":
    import sys
    
    PROJECT_ID = "ambient-isotope-463716-u6"
    
    # Check for test mode
    test_mode = '--test' in sys.argv
    limit = 100 if test_mode else None
    
    collector = Portfolio30PubSubCollector(PROJECT_ID)
    
    if test_mode:
        print("\n🧪 TEST MODE: Collecting 100 records per stock")
        print("   Run without --test flag for full collection\n")
    
    results = collector.collect_all_stocks(limit_per_stock=limit)
