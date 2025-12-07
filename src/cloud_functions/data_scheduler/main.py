import os
import requests
import json
import time
from datetime import datetime
from google.cloud import pubsub_v1
import functions_framework

@functions_framework.http
def scheduled_data_collection(request):
    """HTTP function triggered by Cloud Scheduler"""
    
    api_key = os.environ.get('ALPHA_VANTAGE_API_KEY')
    project_id = os.environ.get('GCP_PROJECT')
    
    if not api_key:
        return "API key not configured", 500
    
    publisher = pubsub_v1.PublisherClient()
    topic_path = publisher.topic_path(project_id, "market-data-raw")
    
    symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA"]
    collected_count = 0
    
    for symbol in symbols:
        try:
            # Fetch latest data from Alpha Vantage
            url = "https://www.alphavantage.co/query"
            params = {
                "function": "TIME_SERIES_INTRADAY",
                "symbol": symbol,
                "interval": "5min",
                "apikey": api_key,
                "outputsize": "compact"
            }
            
            response = requests.get(url, params=params)
            data = response.json()
            
            if "Time Series (5min)" in data:
                time_series = data["Time Series (5min)"]
                latest_time = max(time_series.keys())
                latest_data = time_series[latest_time]
                
                market_data = {
                    "symbol": symbol,
                    "timestamp": latest_time.replace(" ", "T") + "Z",
                    "open_price": float(latest_data["1. open"]),
                    "high_price": float(latest_data["2. high"]),
                    "low_price": float(latest_data["3. low"]),
                    "close_price": float(latest_data["4. close"]),
                    "volume": int(latest_data["5. volume"]),
                    "data_source": "alpha_vantage_scheduled"
                }
                
                message = json.dumps(market_data).encode('utf-8')
                future = publisher.publish(topic_path, message)
                future.result()  # Wait for publish
                collected_count += 1
                
            time.sleep(12)  # Rate limiting
            
        except Exception as e:
            print(f"Error collecting data for {symbol}: {e}")
    
    return f"Successfully collected data for {collected_count} symbols"
