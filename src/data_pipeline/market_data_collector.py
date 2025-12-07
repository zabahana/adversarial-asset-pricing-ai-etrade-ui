import requests
import json
import time
import os
from datetime import datetime
from google.cloud import pubsub_v1

class MarketDataCollector:
    def __init__(self, api_key, project_id):
        self.api_key = api_key
        self.project_id = project_id
        self.publisher = pubsub_v1.PublisherClient()
        self.topic_path = self.publisher.topic_path(project_id, "market-data-raw")
    
    def fetch_daily_data(self, symbol):
        """Fetch daily data from Alpha Vantage"""
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "TIME_SERIES_DAILY_ADJUSTED",
            "symbol": symbol,
            "apikey": self.api_key,
            "outputsize": "compact"  # Last 100 data points
        }
        
        response = requests.get(url, params=params)
        data = response.json()
        
        if "Time Series (Daily)" not in data:
            print(f"Error fetching data for {symbol}: {data}")
            return []
        
        time_series = data["Time Series (Daily)"]
        market_data = []
        
        for date, values in time_series.items():
            market_data.append({
                "symbol": symbol,
                "timestamp": f"{date}T16:00:00Z",  # Market close time
                "open_price": float(values["1. open"]),
                "high_price": float(values["2. high"]),
                "low_price": float(values["3. low"]),
                "close_price": float(values["4. close"]),
                "adjusted_close": float(values["5. adjusted close"]),
                "volume": int(values["6. volume"]),
                "data_source": "alpha_vantage"
            })
        
        return market_data
    
    def publish_data(self, market_data):
        """Publish data to Pub/Sub"""
        for data_point in market_data:
            message = json.dumps(data_point).encode('utf-8')
            future = self.publisher.publish(self.topic_path, message)
            print(f"Published {data_point['symbol']} {data_point['timestamp'][:10]}: {future.result()}")
            time.sleep(1)  # Rate limiting
    
    def collect_and_publish(self, symbols):
        """Collect and publish data for multiple symbols"""
        for symbol in symbols:
            print(f"Fetching data for {symbol}...")
            data = self.fetch_daily_data(symbol)
            if data:
                print(f"Publishing {len(data)} records for {symbol}")
                self.publish_data(data[:5])  # Publish last 5 days
                time.sleep(12)  # Alpha Vantage rate limit: 5 calls/minute

if __name__ == "__main__":
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    project_id = "ambient-isotope-463716-u6"
    
    if not api_key:
        print("Please set ALPHA_VANTAGE_API_KEY environment variable")
        exit(1)
    
    collector = MarketDataCollector(api_key, project_id)
    symbols = ["AAPL", "GOOGL", "MSFT", "TSLA"]
    
    collector.collect_and_publish(symbols)
