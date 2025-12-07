"""
Fixed version with correct schema
"""
with open('collect_30_stocks.py', 'r') as f:
    content = f.read()

# Fix 1: Replace utcnow() with UTC-aware datetime
content = content.replace(
    "df['ingestion_timestamp'] = datetime.utcnow()",
    "df['ingestion_timestamp'] = datetime.now(datetime.timezone.utc)"
)

# Fix 2: Add the missing field name that BigQuery expects
old_cols = """        cols_to_save = [
            'symbol', 'timestamp', 'open_price', 'high_price', 'low_price',
            'close_price', 'volume', 'daily_return', 'volatility_20d',
            'sma_5', 'sma_10', 'sma_20', 'sma_50', 'price_change_1d',
            'price_change_5d', 'momentum_10d', 'rsi_14', 'volatility_10d',
            'sharpe_20d', 'ingestion_timestamp'
        ]"""

new_cols = """        cols_to_save = [
            'symbol', 'timestamp', 'open_price', 'high_price', 'low_price',
            'close_price', 'volume', 'daily_return', 'volatility_20d',
            'sma_5', 'sma_10', 'sma_20', 'sma_50', 'price_change_1d',
            'price_change_5d', 'momentum_10d', 'rsi_14', 'volatility_10d',
            'sharpe_20d', 'ingestion_timestamp'
        ]
        
        # Add missing field that BigQuery table expects
        df_to_save = df[cols_to_save].copy()
        df_to_save['processing_timestamp'] = df_to_save['ingestion_timestamp']"""

content = content.replace(old_cols, new_cols)

# Fix 3: Update the df_to_save line
content = content.replace(
    "\n        df_to_save = df[cols_to_save].copy()\n        \n        # Load",
    "\n        \n        # Load"
)

with open('collect_30_stocks.py', 'w') as f:
    f.write(content)

print("✓ Fixed schema mismatch issues")
