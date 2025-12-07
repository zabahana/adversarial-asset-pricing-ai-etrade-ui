"""
Compare Standard DQN vs Attention DQN on 30 stocks (LOCAL DATA)
"""
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from protagonist_dqn import ProtagonistDQN, TradingEnvironment
from attention_dqn import AttentionDQN
import time
from datetime import datetime
import os
import glob

class DQNComparison30Stocks:
    def __init__(self, data_dir='stock_data'):
        self.data_dir = data_dir
        self.results = {
            'standard': {},
            'attention': {}
        }
    
    def load_stock_data(self, symbol):
        """Load and resample stock data to monthly"""
        filepath = os.path.join(self.data_dir, f'{symbol}.csv')
        
        if not os.path.exists(filepath):
            print(f"  ⚠️  File not found: {filepath}")
            return None
        
        df = pd.read_csv(filepath)
        
        # Fix datetime parsing
        df['Date'] = pd.to_datetime(df['Date'], utc=True)
        df = df.sort_values('Date')
        df = df.set_index('Date')
        
        # Ensure index is DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # Resample to monthly
        df_monthly = df.resample('ME').agg({
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()
        
        df_monthly = df_monthly.reset_index()
        df_monthly.columns = ['timestamp', 'close_price', 'volume']
        
        # Add technical indicators
        df_monthly['sma_5'] = df_monthly['close_price'].rolling(window=5).mean()
        df_monthly['sma_10'] = df_monthly['close_price'].rolling(window=10).mean()
        df_monthly['sma_20'] = df_monthly['close_price'].rolling(window=20).mean()
        df_monthly['price_change_1d'] = df_monthly['close_price'].pct_change() * 100
        df_monthly['price_change_5d'] = df_monthly['close_price'].pct_change(5) * 100
        df_monthly['volatility_10d'] = df_monthly['close_price'].rolling(window=10).std()
        df_monthly['returns'] = df_monthly['close_price'].pct_change().fillna(0)
        df_monthly['momentum'] = df_monthly['close_price'] - df_monthly['close_price'].shift(3).fillna(df_monthly['close_price'])
        df_monthly['volatility_ratio'] = df_monthly['volatility_10d'] / (df_monthly['volatility_10d'].rolling(6).mean().fillna(1) + 1e-6)
        
        df_monthly = df_monthly.fillna(0)
        df_monthly['symbol'] = symbol
        
        print(f"  {symbol}: {len(df_monthly)} months")
        return df_monthly
    
    def train_single_stock(self, df, model_type='standard', num_episodes=20):
        """Train one model on one stock"""
        
        n = len(df)
        train_end = int(n * 0.7)
        val_end = int(n * 0.85)
        
        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy()
        test_df = df.iloc[val_end:].copy()
        
        state_size = 14
        
        if model_type == 'attention':
            agent = AttentionDQN(
                state_size=state_size, action_size=3,
                num_heads=4, learning_rate=0.001,
                gamma=0.95, epsilon_start=1.0,
                epsilon_end=0.01, epsilon_decay=0.995
            )
        else:
            agent = ProtagonistDQN(
                state_size=state_size, action_size=3,
                learning_rate=0.001, gamma=0.95,
                epsilon_start=1.0, epsilon_end=0.01,
                epsilon_decay=0.995
            )
        
        # Quick training (silent)
        for episode in range(num_episodes):
            train_env = TradingEnvironment(train_df, initial_capital=100000)
            state = train_env.reset()
            
            while True:
                action = agent.select_action(state, training=True)
                next_state, reward, done, info = train_env.step(action)
                agent.store_experience(state, action, reward, next_state, done)
                agent.train_step(batch_size=32)
                
                if done:
                    break
                state = next_state
            
            agent.decay_epsilon()
            
            if (episode + 1) % 10 == 0:
                agent.update_target_network()
        
        # Test evaluation
        test_env = TradingEnvironment(test_df, initial_capital=100000)
        state = test_env.reset()
        
        while True:
            action = agent.select_action(state, training=False)
            next_state, reward, done, info = test_env.step(action)
            
            if done:
                break
            state = next_state
        
        test_metrics = test_env.calculate_metrics()
        bh_return = (test_df['close_price'].iloc[-1] - test_df['close_price'].iloc[0]) / test_df['close_price'].iloc[0]
        
        return {
            'test_metrics': test_metrics,
            'buy_hold_return': bh_return
        }
    
    def compare_on_stock(self, symbol, num_episodes=20):
        """Compare both models on one stock"""
        
        print(f"[{symbol}] ", end='', flush=True)
        
        df = self.load_stock_data(symbol)
        
        if df is None or len(df) < 20:
            print(f"⚠️  Skip")
            return None
        
        # Train both models
        print(f"🤖", end='', flush=True)
        standard_result = self.train_single_stock(df, 'standard', num_episodes)
        
        print(f"🧠", end='', flush=True)
        attention_result = self.train_single_stock(df, 'attention', num_episodes)
        
        # Store results
        self.results['standard'][symbol] = standard_result
        self.results['attention'][symbol] = attention_result
        
        # Quick comparison
        std_return = standard_result['test_metrics']['total_return'] * 100
        att_return = attention_result['test_metrics']['total_return'] * 100
        
        winner = '🧠' if att_return > std_return else '🤖'
        print(f" {winner} Std:{std_return:5.1f}% Att:{att_return:5.1f}%")
        
        return {
            'standard': standard_result,
            'attention': attention_result
        }

if __name__ == "__main__":
    print("=" * 80)
    print("30-STOCK DQN COMPARISON (LOCAL DATA)")
    print("=" * 80)
    
    comparator = DQNComparison30Stocks()
    
    # Get all available stocks
    csv_files = glob.glob('stock_data/*.csv')
    available_stocks = sorted([os.path.basename(f).replace('.csv', '') for f in csv_files])
    
    print(f"\n✓ Found {len(available_stocks)} stocks")
    print(f"📊 Training: 20 episodes per model")
    print(f"⏱️  Est. time: ~{len(available_stocks) * 2} minutes")
    print("=" * 80)
    
    # Option to test on subset first
    response = input(f"\nTrain on all {len(available_stocks)} stocks? (yes/no/number): ")
    
    if response.lower() == 'no':
        print("Cancelled.")
        exit()
    elif response.isdigit():
        n = int(response)
        stocks_to_compare = available_stocks[:n]
        print(f"\n🎯 Training on first {n} stocks")
    else:
        stocks_to_compare = available_stocks
        print(f"\n🎯 Training on all {len(stocks_to_compare)} stocks")
    
    print("=" * 80)
    
    stocks_compared = {}
    start_time = time.time()
    
    for i, symbol in enumerate(stocks_to_compare, 1):
        print(f"{i:2d}/{len(stocks_to_compare)} ", end='')
        result = comparator.compare_on_stock(symbol, num_episodes=20)
        if result:
            stocks_compared[symbol] = result
    
    total_time = time.time() - start_time
    
    # Generate summary
    print("\n" + "=" * 80)
    print("COMPARISON COMPLETE!")
    print("=" * 80)
    print(f"⏱️  Total time: {total_time/60:.1f} minutes")
    print(f"✓ Stocks compared: {len(stocks_compared)}")
    
    # Quick stats
    symbols = list(stocks_compared.keys())
    std_returns = [comparator.results['standard'][s]['test_metrics']['total_return'] * 100 for s in symbols]
    att_returns = [comparator.results['attention'][s]['test_metrics']['total_return'] * 100 for s in symbols]
    
    att_wins = sum(1 for i in range(len(symbols)) if att_returns[i] > std_returns[i])
    
    print(f"\n📊 Average Standard DQN:  {np.mean(std_returns):6.2f}%")
    print(f"📊 Average Attention DQN: {np.mean(att_returns):6.2f}%")
    print(f"🏆 Attention wins: {att_wins}/{len(symbols)} stocks")
    print("=" * 80)
    
    # Save results
    results_dict = {
        'symbols': symbols,
        'standard': std_returns,
        'attention': att_returns,
        'buy_hold': [comparator.results['standard'][s]['buy_hold_return'] * 100 for s in symbols]
    }
    
    import json
    with open('30_stocks_results.json', 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print("\n✓ Results saved to: 30_stocks_results.json")
    print("\nNext: Run visualization script to generate charts!")
    print("=" * 80)
