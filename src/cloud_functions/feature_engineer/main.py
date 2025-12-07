"""
Cloud Function for computing basic technical indicators
"""

import json
import base64
from datetime import datetime, timedelta
from google.cloud import bigquery
import functions_framework

bq_client = bigquery.Client()

def calculate_simple_moving_average(prices, period):
    """Calculate Simple Moving Average"""
    if len(prices) < period:
        return None
    return sum(prices[-period:]) / period

def calculate_price_change(prices):
    """Calculate price change percentage"""
    if len(prices) < 2:
        return None
    return ((prices[-1] - prices[-2]) / prices[-2]) * 100

def get_historical_prices(symbol, days=30):
    """Get historical price data from BigQuery"""
    
    query = f"""
    SELECT 
        close_price,
        timestamp
    FROM `{bq_client.project}.raw_market_data.daily_prices`
    WHERE symbol = @symbol
    AND timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL @days DAY)
    ORDER BY timestamp ASC
    """
    
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("symbol", "STRING", symbol),
            bigquery.ScalarQueryParameter("days", "INT64", days),
        ]
    )
    
    try:
        query_job = bq_client.query(query, job_config=job_config)
        results = query_job.result()
        
        prices = []
        for row in results:
            if row.close_price is not None:
                prices.append(float(row.close_price))
        
        return prices
    except Exception as e:
        print(f"Error fetching historical data: {e}")
        return []

def compute_basic_features(symbol, current_data):
    """Compute basic technical indicators"""
    
    # Get historical data
    historical_prices = get_historical_prices(symbol, days=30)
    
    if len(historical_prices) < 5:  # Need minimum data
        print(f"Insufficient historical data for {symbol}: {len(historical_prices)} points")
        return None
    
    # Add current price
    current_price = float(current_data['close_price'])
    all_prices = historical_prices + [current_price]
    
    # Calculate basic indicators
    features = {
        'symbol': symbol,
        'timestamp': current_data['timestamp'],
        'close_price': current_price,
        'volume': current_data.get('volume'),
    }
    
    # Simple Moving Averages
    features['sma_5'] = calculate_simple_moving_average(all_prices, 5)
    features['sma_10'] = calculate_simple_moving_average(all_prices, 10)
    features['sma_20'] = calculate_simple_moving_average(all_prices, 20)
    
    # Price changes
    features['price_change_1d'] = calculate_price_change(all_prices)
    if len(all_prices) >= 6:
        features['price_change_5d'] = ((all_prices[-1] - all_prices[-6]) / all_prices[-6]) * 100
    
    # Basic volatility (standard deviation of recent prices)
    if len(all_prices) >= 10:
        recent_prices = all_prices[-10:]
        mean_price = sum(recent_prices) / len(recent_prices)
        variance = sum((price - mean_price) ** 2 for price in recent_prices) / len(recent_prices)
        features['volatility_10d'] = (variance ** 0.5) / mean_price * 100  # Coefficient of variation
    
    features['processing_timestamp'] = datetime.utcnow().isoformat()
    
    return features

@functions_framework.cloud_event
def process_features(cloud_event):
    """Main function to process basic technical indicators"""
    
    try:
        # Decode message
        message_data = base64.b64decode(cloud_event.data['message']['data']).decode('utf-8')
        data = json.loads(message_data)
        
        symbol = data.get('symbol')
        print(f"Computing features for {symbol}")
        
        # Compute features
        features = compute_basic_features(symbol, data)
        
        if features is None:
            print(f"Could not compute features for {symbol}")
            return
        
        # Insert into BigQuery
        table_id = f"{bq_client.project}.processed_market_data.technical_indicators"
        
        # Prepare row for insertion (remove None values)
        row_to_insert = {}
        for key, value in features.items():
            if value is not None:
                row_to_insert[key] = value
        
        try:
            table = bq_client.get_table(table_id)
            errors = bq_client.insert_rows_json(table, [row_to_insert])
            
            if errors:
                print(f"BigQuery insertion errors: {errors}")
            else:
                print(f"Successfully computed and stored features for {symbol}")
                
        except Exception as e:
            print(f"Error inserting into BigQuery: {e}")
    
    except Exception as e:
        print(f"Error processing features: {str(e)}")
        raise
