import yfinance as yf
import json
import time
from datetime import datetime
from google.cloud import pubsub_v1

class YFinanceCollector:
    def __init__(self, project_id):
        self.project_id = project_id
        self.publisher = pubsub_v1.PublisherClient()
        self.topic_path = self.publisher.topic_path(project_id, "market-data-raw")
    
    def fetch_historical_data(self, symbol, period="max"):
        """Fetch historical data using yfinance"""
        print(f"\nFetching historical data for {symbol}...")
        
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)
            
            if hist.empty:
                print(f"✗ No data returned for {symbol}")
                return []
            
            # Convert to our format
            market_data = []
            for date, row in hist.iterrows():
                data_point = {
                    "symbol": symbol,
                    "timestamp": date.strftime("%Y-%m-%dT16:00:00Z"),
                    "open_price": float(row['Open']),
                    "high_price": float(row['High']),
                    "low_price": float(row['Low']),
                    "close_price": float(row['Close']),
                    "volume": int(row['Volume']),
                    "data_source": "yfinance_historical"
                }
                market_data.append(data_point)
            
            print(f"✓ Fetched {len(market_data)} records for {symbol}")
            print(f"  Date range: {market_data[0]['timestamp'][:10]} to {market_data[-1]['timestamp'][:10]}")
            return market_data
            
        except Exception as e:
            print(f"✗ Error fetching data for {symbol}: {e}")
            return []
    
    def publish_batch(self, data, batch_size=10):
        """Publish data in batches to avoid overwhelming the pipeline"""
        total = len(data)
        published = 0
        
        for i in range(0, total, batch_size):
            batch = data[i:i+batch_size]
            
            for data_point in batch:
                try:
                    message = json.dumps(data_point).encode('utf-8')
                    future = self.publisher.publish(self.topic_path, message)
                    future.result()
                    published += 1
                except Exception as e:
                    print(f"✗ Failed to publish: {e}")
            
            # Progress update
            print(f"  Published {published}/{total} records...", end='\r')
            time.sleep(1)  # Small delay between batches
        
        print(f"\n✓ Successfully published {published} records")
        return published
    
    def collect_full_history(self, symbols):
        """Collect full historical data for multiple symbols"""
        all_data = {}
        
        print("=" * 80)
        print("HISTORICAL DATA COLLECTION (yfinance)")
        print("=" * 80)
        print(f"Collecting data for: {', '.join(symbols)}")
        print("=" * 80)
        
        for symbol in symbols:
            data = self.fetch_historical_data(symbol, period="max")
            
            if data:
                all_data[symbol] = data
                
                # Publish to pipeline
                print(f"Publishing {symbol} to pipeline...")
                self.publish_batch(data, batch_size=20)
                
                # Small delay between symbols
                print("Waiting 5 seconds before next symbol...\n")
                time.sleep(5)
            else:
                print(f"✗ Skipping {symbol} due to fetch error\n")
        
        return all_data

if __name__ == "__main__":
    PROJECT_ID = "ambient-isotope-463716-u6"
    SYMBOLS = ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA"]
    
    collector = YFinanceCollector(PROJECT_ID)
    historical_data = collector.collect_full_history(SYMBOLS)
    
    # Summary
    print("\n" + "=" * 80)
    print("COLLECTION COMPLETE")
    print("=" * 80)
    total_records = sum(len(data) for data in historical_data.values())
    print(f"Total records collected: {total_records}")
    
    for symbol, data in historical_data.items():
        if data:
            print(f"{symbol}: {len(data)} records ({data[0]['timestamp'][:10]} to {data[-1]['timestamp'][:10]})")
    
    print("\n📊 Data is being processed through your pipeline...")
    print("⏰ Wait ~5-10 minutes for feature engineering to complete")
    print("=" * 80)
