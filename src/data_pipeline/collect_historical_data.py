"""
Collect extended historical market data for model training
"""
import requests
import json
import time
import os
from datetime import datetime, timedelta
from google.cloud import pubsub_v1

class HistoricalDataCollector:
    def __init__(self, api_key, project_id):
        self.api_key = api_key
        self.project_id = project_id
        self.publisher = pubsub_v1.PublisherClient()
        self.topic_path = self.publisher.topic_path(project_id, "market-data-raw")
        self.base_url = "https://www.alphavantage.co/query"
    
    def fetch_daily_adjusted(self, symbol, outputsize='full'):
        """
        Fetch full historical data (20+ years)
        outputsize='full' gives 20+ years of data
        outputsize='compact' gives last 100 days
        """
        params = {
            "function": "TIME_SERIES_DAILY_ADJUSTED",
            "symbol": symbol,
            "apikey": self.api_key,
            "outputsize": outputsize,
            "datatype": "json"
        }
        
        print(f"Fetching historical data for {symbol}...")
        response = requests.get(self.base_url, params=params)
        
        if response.status_code != 200:
            print(f"Error: HTTP {response.status_code}")
            return []
        
        data = response.json()
        
        if "Time Series (Daily)" not in data:
            print(f"Error response: {data.get('Note', data.get('Error Message', 'Unknown error'))}")
            return []
        
        time_series = data["Time Series (Daily)"]
        print(f"Retrieved {len(time_series)} days of data for {symbol}")
        
        market_data = []
        for date_str, values in time_series.items():
            try:
                market_data.append({
                    "symbol": symbol,
                    "timestamp": f"{date_str}T16:00:00Z",  # Market close
                    "open_price": float(values["1. open"]),
                    "high_price": float(values["2. high"]),
                    "low_price": float(values["3. low"]),
                    "close_price": float(values["4. close"]),
                    "adjusted_close": float(values["5. adjusted close"]),
                    "volume": int(values["6. volume"]),
                    "data_source": "alpha_vantage_historical"
                })
            except (KeyError, ValueError) as e:
                print(f"Error parsing data for {date_str}: {e}")
                continue
        
        # Sort by date (oldest first)
        market_data.sort(key=lambda x: x['timestamp'])
        return market_data
    
    def publish_batch(self, market_data, batch_size=50):
        """Publish data in batches to avoid overwhelming the system"""
        total = len(market_data)
        published = 0
        
        print(f"Publishing {total} records in batches of {batch_size}...")
        
        for i in range(0, total, batch_size):
            batch = market_data[i:i+batch_size]
            
            for data_point in batch:
                try:
                    message = json.dumps(data_point).encode('utf-8')
                    future = self.publisher.publish(self.topic_path, message)
                    future.result(timeout=10)
                    published += 1
                    
                    if published % 100 == 0:
                        print(f"Published {published}/{total} records...")
                        
                except Exception as e:
                    print(f"Error publishing record: {e}")
            
            # Small delay between batches
            time.sleep(2)
        
        print(f"✓ Successfully published {published} records")
        return published
    
    def collect_full_history(self, symbols):
        """Collect full historical data for multiple symbols"""
        all_data = {}
        
        for symbol in symbols:
            data = self.fetch_daily_adjusted(symbol, outputsize='full')
            
            if data:
                all_data[symbol] = data
                print(f"✓ Collected {len(data)} records for {symbol}")
                
                # Publish to pipeline
                self.publish_batch(data)
                
                # Rate limiting - Alpha Vantage free tier: 5 calls/minute
                print("Waiting 15 seconds for rate limit...")
                time.sleep(15)
            else:
                print(f"✗ Failed to collect data for {symbol}")
        
        return all_data

if __name__ == "__main__":
    # Configuration
    API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
    PROJECT_ID = "ambient-isotope-463716-u6"
    
    if not API_KEY:
        print("Error: ALPHA_VANTAGE_API_KEY environment variable not set")
        exit(1)
    
    # Symbols to collect
    SYMBOLS = ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA"]
    
    print("=" * 80)
    print("HISTORICAL DATA COLLECTION")
    print("=" * 80)
    print(f"Collecting data for: {', '.join(SYMBOLS)}")
    print(f"This will take approximately {len(SYMBOLS) * 1} minutes")
    print("=" * 80)
    
    collector = HistoricalDataCollector(API_KEY, PROJECT_ID)
    
    # Collect data
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
    
    print("\nData is being processed through your pipeline...")
    print("Wait ~5 minutes for feature engineering to complete")
    print("=" * 80)
