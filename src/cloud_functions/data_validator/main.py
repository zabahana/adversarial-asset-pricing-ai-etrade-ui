import json
import base64
from datetime import datetime
from google.cloud import bigquery
from google.cloud import pubsub_v1
import functions_framework

bq_client = bigquery.Client()
publisher = pubsub_v1.PublisherClient()

def validate_market_data(data):
    required_fields = ['symbol', 'timestamp', 'close_price', 'data_source']
    
    for field in required_fields:
        if field not in data:
            return False, f"Missing required field: {field}"
    
    try:
        if not isinstance(data['symbol'], str) or len(data['symbol']) < 1:
            return False, "Invalid symbol"
        
        if not isinstance(data['close_price'], (int, float)) or data['close_price'] <= 0:
            return False, "Invalid close_price"
        
        return True, "Valid"
        
    except Exception as e:
        return False, f"Validation error: {str(e)}"

@functions_framework.cloud_event
def process_market_data(cloud_event):
    try:
        message_data = base64.b64decode(cloud_event.data['message']['data']).decode('utf-8')
        data = json.loads(message_data)
        
        print(f"Processing message for symbol: {data.get('symbol', 'unknown')}")
        
        is_valid, validation_message = validate_market_data(data)
        
        if not is_valid:
            print(f"Validation failed: {validation_message}")
            return
        
        data['ingestion_timestamp'] = datetime.utcnow().isoformat()
        
        # Insert into BigQuery
        table_id = f"{bq_client.project}.raw_market_data.daily_prices"
        
        row_to_insert = {
            'symbol': data['symbol'],
            'timestamp': data['timestamp'],
            'close_price': data['close_price'],
            'data_source': data['data_source'],
            'ingestion_timestamp': data['ingestion_timestamp']
        }
        
        table = bq_client.get_table(table_id)
        errors = bq_client.insert_rows_json(table, [row_to_insert])
        
        if errors:
            print(f"BigQuery insertion errors: {errors}")
        else:
            print(f"Successfully inserted data for {data['symbol']}")
            
            # Publish to processed topic for feature engineering
            processed_topic = f"projects/{bq_client.project}/topics/market-data-processed"
            message_bytes = json.dumps(data).encode('utf-8')
            
            future = publisher.publish(processed_topic, message_bytes)
            message_id = future.result()
            print(f"Published to processed topic: {message_id}")
    
    except Exception as e:
        print(f"Error processing message: {str(e)}")
        raise
