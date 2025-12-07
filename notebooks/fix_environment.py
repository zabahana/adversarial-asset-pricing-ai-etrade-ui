import sys

# Read the file
with open('protagonist_dqn.py', 'r') as f:
    content = f.read()

# Find and fix the __init__ method
old_init = """    def __init__(self, data, initial_capital=100000, transaction_cost=0.001,
                 shares_per_trade=100):
        \"\"\"
        Args:
            data: DataFrame with columns [timestamp, symbol, close_price, features...]
            initial_capital: Starting cash
            transaction_cost: Transaction cost as percentage (0.001 = 0.1%)
            shares_per_trade: Number of shares to buy/sell per action
        \"\"\"
        self.data = data.reset_index(drop=True)
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.shares_per_trade = shares_per_trade
        
        # State
        self.reset()
        
        # Feature columns (exclude timestamp, symbol)
        self.feature_cols = [col for col in data.columns 
                           if col not in ['timestamp', 'symbol', 'ingestion_timestamp']]"""

new_init = """    def __init__(self, data, initial_capital=100000, transaction_cost=0.001,
                 shares_per_trade=100):
        \"\"\"
        Args:
            data: DataFrame with columns [timestamp, symbol, close_price, features...]
            initial_capital: Starting cash
            transaction_cost: Transaction cost as percentage (0.001 = 0.1%)
            shares_per_trade: Number of shares to buy/sell per action
        \"\"\"
        self.data = data.reset_index(drop=True)
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.shares_per_trade = shares_per_trade
        
        # Feature columns (exclude timestamp, symbol) - MUST BE SET BEFORE reset()
        self.feature_cols = [col for col in data.columns 
                           if col not in ['timestamp', 'symbol', 'ingestion_timestamp']]
        
        # State - NOW we can reset
        self.reset()"""

content = content.replace(old_init, new_init)

# Write back
with open('protagonist_dqn.py', 'w') as f:
    f.write(content)

print("✓ Fixed TradingEnvironment initialization order")
