from google.cloud import bigquery

PROJECT_ID = "ambient-isotope-463716-u6"
client = bigquery.Client(project=PROJECT_ID)

# Check what tables exist
print("Checking datasets and tables...")
query = """
SELECT table_name 
FROM `ambient-isotope-463716-u6.processed_market_data.INFORMATION_SCHEMA.TABLES`
"""
tables = client.query(query).to_dataframe()
print("\nAvailable tables:")
print(tables)

# Check schema of technical_indicators
print("\n" + "="*80)
print("Schema of technical_indicators table:")
print("="*80)
query = """
SELECT column_name, data_type
FROM `ambient-isotope-463716-u6.processed_market_data.INFORMATION_SCHEMA.COLUMNS`
WHERE table_name = 'technical_indicators'
ORDER BY ordinal_position
"""
schema = client.query(query).to_dataframe()
print(schema)

# Check sample data
print("\n" + "="*80)
print("Sample data (first 5 rows):")
print("="*80)
query = """
SELECT *
FROM `ambient-isotope-463716-u6.processed_market_data.technical_indicators`
LIMIT 5
"""
sample = client.query(query).to_dataframe()
print(sample)

# Check distinct symbols
print("\n" + "="*80)
print("Available symbols:")
print("="*80)
query = """
SELECT DISTINCT symbol, COUNT(*) as row_count
FROM `ambient-isotope-463716-u6.processed_market_data.technical_indicators`
GROUP BY symbol
ORDER BY symbol
"""
symbols = client.query(query).to_dataframe()
print(symbols)
