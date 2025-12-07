#!/usr/bin/env python3
"""
BigQuery table schemas for the asset pricing AI system
"""

from google.cloud import bigquery

# Schema for raw market data
RAW_MARKET_DATA_SCHEMA = [
    bigquery.SchemaField("symbol", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("timestamp", "TIMESTAMP", mode="REQUIRED"), 
    bigquery.SchemaField("open_price", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("high_price", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("low_price", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("close_price", "FLOAT64", mode="REQUIRED"),
    bigquery.SchemaField("volume", "INT64", mode="NULLABLE"),
    bigquery.SchemaField("adjusted_close", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("data_source", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("ingestion_timestamp", "TIMESTAMP", mode="REQUIRED"),
]

# Schema for processed market data with technical indicators
PROCESSED_MARKET_DATA_SCHEMA = [
    bigquery.SchemaField("symbol", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("timestamp", "TIMESTAMP", mode="REQUIRED"),
    bigquery.SchemaField("close_price", "FLOAT64", mode="REQUIRED"),
    bigquery.SchemaField("volume", "INT64", mode="NULLABLE"),
    # Technical indicators
    bigquery.SchemaField("rsi_14", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("rsi_30", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("macd", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("macd_signal", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("macd_histogram", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("bb_upper", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("bb_middle", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("bb_lower", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("sma_20", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("sma_50", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("ema_12", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("ema_26", "FLOAT64", mode="NULLABLE"),
    # Price-based features
    bigquery.SchemaField("price_change_1d", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("price_change_5d", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("volatility_20d", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("processing_timestamp", "TIMESTAMP", mode="REQUIRED"),
]

# Schema for training data
ML_TRAINING_DATA_SCHEMA = [
    bigquery.SchemaField("symbol", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("timestamp", "TIMESTAMP", mode="REQUIRED"),
    bigquery.SchemaField("state_vector", "REPEATED", mode="REPEATED", 
                         fields=[bigquery.SchemaField("feature_value", "FLOAT64")]),
    bigquery.SchemaField("action_space", "JSON", mode="NULLABLE"),
    bigquery.SchemaField("reward", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("next_state_vector", "REPEATED", mode="REPEATED",
                         fields=[bigquery.SchemaField("feature_value", "FLOAT64")]),
    bigquery.SchemaField("done", "BOOLEAN", mode="REQUIRED"),
    bigquery.SchemaField("episode_id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("step_number", "INT64", mode="REQUIRED"),
]

def create_tables(project_id):
    """Create all required BigQuery tables"""
    
    client = bigquery.Client(project=project_id)
    
    tables_config = [
        {
            "dataset_id": "raw_market_data",
            "table_id": "daily_prices", 
            "schema": RAW_MARKET_DATA_SCHEMA,
            "description": "Daily OHLCV data for all assets"
        },
        {
            "dataset_id": "raw_market_data", 
            "table_id": "intraday_prices",
            "schema": RAW_MARKET_DATA_SCHEMA,
            "description": "Intraday price data (1min, 5min, 15min intervals)"
        },
        {
            "dataset_id": "processed_market_data",
            "table_id": "technical_indicators",
            "schema": PROCESSED_MARKET_DATA_SCHEMA, 
            "description": "Market data with computed technical indicators"
        },
        {
            "dataset_id": "ml_training_data",
            "table_id": "rl_episodes",
            "schema": ML_TRAINING_DATA_SCHEMA,
            "description": "Training episodes for reinforcement learning"
        }
    ]
    
    for table_config in tables_config:
        dataset_ref = client.dataset(table_config["dataset_id"])
        table_ref = dataset_ref.table(table_config["table_id"])
        
        table = bigquery.Table(table_ref, schema=table_config["schema"])
        table.description = table_config["description"]
        
        # Set partitioning by date for better performance
        if "prices" in table_config["table_id"]:
            table.time_partitioning = bigquery.TimePartitioning(
                type_=bigquery.TimePartitioningType.DAY,
                field="timestamp"
            )
        
        try:
            table = client.create_table(table)
            print(f" Created table {table.project}.{table.dataset_id}.{table.table_id}")
        except Exception as e:
            if "Already Exists" in str(e):
                print(f"  Table {table_config['dataset_id']}.{table_config['table_id']} already exists")
            else:
                print(f" Error creating table {table_config['table_id']}: {e}")

if __name__ == "__main__":
    import os
    
    project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
    if not project_id:
        print(" Please set GOOGLE_CLOUD_PROJECT environment variable")
        exit(1)
    
    print("  Creating BigQuery tables...")
    create_tables(project_id)
    print(" BigQuery table creation completed!")