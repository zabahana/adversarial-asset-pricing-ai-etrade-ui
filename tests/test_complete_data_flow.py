#!/usr/bin/env python3
"""
Test the complete data flow from Pub/Sub to BigQuery with Cloud Functions
"""

import json
import time
import os
from datetime import datetime, timedelta
from google.cloud import pubsub_v1
from google.cloud import bigquery

def create_sample_market_data():
    """Create sample market data for testing"""
    
    symbols = ["AAPL", "GOOGL", "MSFT", "TSLA"]
    sample_data = []
    
    base_time = datetime.utcnow() - timedelta(days=5)  # Start with recent data
    
    for symbol in symbols:
        # Generate 5 days of sample data (need historical data for features)
        for i in range(5):
            timestamp = base_time + timedelta(days=i)
            
            # Simulate realistic price movements
            base_prices = {"AAPL": 175, "GOOGL": 140, "MSFT": 420, "TSLA": 250}
            base_price = base_prices[symbol]
            
            # Add some price variation based on day
            price_variation = base_price * 0.02 * ((i % 3) - 1)  # -2%, 0%, +2% variation
            
            data_point = {
                "symbol": symbol,
                "timestamp": timestamp.isoformat() + "Z",
                "open_price": base_price + price_variation * 0.8,
                "high_price": base_price + abs(price_variation),
                "low_price": base_price - abs(price_variation),
                "close_price": base_price + price_variation,
                "volume": 1000000 + (i * 100000),  # Vary volume
                "adjusted_close": base_price + price_variation,
                "data_source": "test_pipeline"
            }
            
            sample_data.append(data_point)
    
    return sample_data

def publish_sample_data(project_id, sample_data, delay_seconds=2):
    """Publish sample data to Pub/Sub with delays"""
    
    publisher = pubsub_v1.PublisherClient()
    topic_path = publisher.topic_path(project_id, "market-data-raw")
    
    published_count = 0
    
    print(f"Publishing {len(sample_data)} messages to market-data-raw topic...")
    
    for i, data_point in enumerate(sample_data):
        message_data = json.dumps(data_point).encode('utf-8')
        
        try:
            future = publisher.publish(topic_path, message_data)
            message_id = future.result(timeout=10)
            published_count += 1
            
            print(f"Published {data_point['symbol']} - {data_point['timestamp'][:10]} (ID: {message_id})")
            
            # Add delay between messages to allow processing
            if i < len(sample_data) - 1:  # Don't delay after the last message
                time.sleep(delay_seconds)
                
        except Exception as e:
            print(f"Failed to publish message: {e}")
    
    print(f"Published {published_count} messages successfully!")
    return published_count

def check_data_processing(project_id, wait_time=30):
    """Check if data has been processed through both functions"""
    
    print(f"Waiting {wait_time} seconds for complete data processing...")
    time.sleep(wait_time)
    
    bq_client = bigquery.Client(project=project_id)
    
    # Check raw data table
    print("\nChecking raw market data...")
    raw_query = f"""
    SELECT 
        symbol,
        COUNT(*) as record_count,
        MIN(timestamp) as earliest_data,
        MAX(timestamp) as latest_data,
        MAX(ingestion_timestamp) as latest_ingestion
    FROM `{project_id}.raw_market_data.daily_prices`
    WHERE data_source = 'test_pipeline'
    GROUP BY symbol
    ORDER BY symbol
    """
    
    raw_count = 0
    try:
        raw_results = bq_client.query(raw_query).result()
        
        for row in raw_results:
            print(f"  {row.symbol}: {row.record_count} records")
            print(f"      Range: {row.earliest_data} to {row.latest_data}")
            raw_count += row.record_count
        
        print(f"Total raw records: {raw_count}")
        
    except Exception as e:
        print(f"Error checking raw data: {e}")
        raw_count = 0
    
    # Check processed data table
    print("\nChecking processed market data...")
    processed_query = f"""
    SELECT 
        symbol,
        COUNT(*) as record_count,
        AVG(sma_5) as avg_sma_5,
        AVG(price_change_1d) as avg_price_change,
        MAX(processing_timestamp) as latest_processing
    FROM `{project_id}.processed_market_data.technical_indicators`
    WHERE symbol IN ('AAPL', 'GOOGL', 'MSFT', 'TSLA')
    GROUP BY symbol
    ORDER BY symbol
    """
    
    processed_count = 0
    try:
        processed_results = bq_client.query(processed_query).result()
        
        for row in processed_results:
            sma_5 = f"{row.avg_sma_5:.2f}" if row.avg_sma_5 else "N/A"
            price_change = f"{row.avg_price_change:.2f}%" if row.avg_price_change else "N/A"
            
            print(f"  {row.symbol}: {row.record_count} processed records")
            print(f"      Avg SMA(5): ${sma_5}, Avg Price Change: {price_change}")
            processed_count += row.record_count
        
        print(f"Total processed records: {processed_count}")
        
    except Exception as e:
        print(f"Error checking processed data: {e}")
        processed_count = 0
    
    return raw_count, processed_count

def main():
    project_id = os.getenv('GOOGLE_CLOUD_PROJECT', "ambient-isotope-463716-u6")
    
    print("Testing Complete Data Pipeline...")
    print("=" * 60)
    print(f"Project: {project_id}")
    print("Pipeline: Pub/Sub -> Data Validator -> Feature Engineer -> BigQuery")
    
    # Step 1: Create sample data
    print("\nStep 1: Creating sample market data...")
    sample_data = create_sample_market_data()
    print(f"Created {len(sample_data)} data points for {len(set(d['symbol'] for d in sample_data))} symbols")
    
    # Step 2: Publish data
    print("\nStep 2: Publishing data to Pub/Sub...")
    published_count = publish_sample_data(project_id, sample_data, delay_seconds=3)
    
    # Step 3: Wait and check processing
    print("\nStep 3: Waiting for Cloud Functions to process data...")
    raw_count, processed_count = check_data_processing(project_id, wait_time=45)
    
    # Summary
    print("\n" + "=" * 60)
    print("PIPELINE TEST SUMMARY:")
    print(f"  Messages published: {published_count}")
    print(f"  Raw data stored: {raw_count}")
    print(f"  Features computed: {processed_count}")
    
    if published_count > 0:
        raw_success_rate = (raw_count / published_count * 100)
        feature_success_rate = (processed_count / raw_count * 100) if raw_count > 0 else 0
        
        print(f"  Raw data success rate: {raw_success_rate:.1f}%")
        print(f"  Feature processing rate: {feature_success_rate:.1f}%")
        
        if processed_count > 0:
            print("PIPELINE TEST PASSED! Complete data flow is working correctly.")
        else:
            print("PIPELINE TEST PARTIAL. Raw data working, check feature engineering function.")
    else:
        print("PIPELINE TEST FAILED. Check Cloud Function logs.")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
