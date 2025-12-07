"""
Daily DQN Comparison with Transaction Costs and 100 Episodes
"""
import numpy as np
import pandas as pd
import os
import glob
import json
import time
from protagonist_dqn import ProtagonistDQN
from attention_dqn import AttentionDQN
from enhanced_trading_env import EnhancedTradingEnvironment

class DailyDQNComparison:
    def __init__(self, data_dir='stock_data'):
        self.data_dir = data_dir
        self.results = {
            'standard': {},
            'attention': {}
        }
    
    def load_daily_data(self, symbol):
        """Load daily stock data with technical indicators"""
        filepath = os.path.join(self.data_dir, f'{symbol}.csv')
        
        if not os.path.exists(filepath):
            return None
        
        df = pd.read_csv(filepath)
        df['Date'] = pd.to_datetime(df['Date'], utc=True)
        df = df.sort_values('Date')
        
        # Rename columns
        df['close_price'] = df['Close']
        df['volume'] = df['Volume']
        
        # Technical indicators
        df['sma_5'] = df['close_price'].rolling(window=5).mean()
        df['sma_10'] = df['close_price'].rolling(window=10).mean()
        df['sma_20'] = df['close_price'].rolling(window=20).mean()
        df['price_change_1d'] = df['close_price'].pct_change() * 100
        df['price_change_5d'] = df['close_price'].pct_change(5) * 100
        df['volatility_10d'] = df['close_price'].rolling(window=10).std()
        df['returns'] = df['close_price'].pct_change().fillna(0)
        df['momentum'] = df['close_price'] - df['close_price'].shift(3).fillna(df['close_price'])
        df['volatility_ratio'] = df['volatility_10d'] / (df['volatility_10d'].rolling(6).mean().fillna(1) + 1e-6)
        
        df = df.fillna(0)
        df['symbol'] = symbol
        
        print(f"  {symbol}: {len(df)} trading days")
        return df
    
    def train_agent(self, df, model_type='standard', num_episodes=100):
        """Train agent with 100+ episodes"""
        
        # Split data (70% train, 15% val, 15% test)
        n = len(df)
        train_end = int(n * 0.70)
        val_end = int(n * 0.85)
        
        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy()
        test_df = df.iloc[val_end:].copy()
        
        state_size = 14
        
        # Create agent
        if model_type == 'attention':
            agent = AttentionDQN(
                state_size=state_size, action_size=3,
                num_heads=4, learning_rate=0.0005,
                gamma=0.95, epsilon_start=1.0,
                epsilon_end=0.01, epsilon_decay=0.995
            )
        else:
            agent = ProtagonistDQN(
                state_size=state_size, action_size=3,
                learning_rate=0.0005, gamma=0.95,
                epsilon_start=1.0, epsilon_end=0.01,
                epsilon_decay=0.995
            )
        
        # Training with transaction costs
        best_val_return = -np.inf
        
        for episode in range(num_episodes):
            # Train episode
            train_env = EnhancedTradingEnvironment(
                train_df, 
                initial_capital=100000,
                commission_pct=0.001,
                slippage_pct=0.0005
            )
            state = train_env.reset()
            
            while True:
                action = agent.select_action(state, training=True)
                next_state, reward, done, info = train_env.step(action)
                agent.store_experience(state, action, reward, next_state, done)
                agent.train_step(batch_size=64)
                
                if done:
                    break
                state = next_state
            
            agent.decay_epsilon()
            
            # Validate every 20 episodes
            if (episode + 1) % 20 == 0:
                val_env = EnhancedTradingEnvironment(val_df, initial_capital=100000)
                val_metrics = self._evaluate(agent, val_env)
                
                if val_metrics['total_return'] > best_val_return:
                    best_val_return = val_metrics['total_return']
                
                agent.update_target_network()
        
        # Test evaluation
        test_env = EnhancedTradingEnvironment(test_df, initial_capital=100000)
        test_metrics = self._evaluate(agent, test_env)
        
        # Buy & hold baseline (with transaction costs)
        bh_cost = test_df['close_price'].iloc[0] * (0.001 + 0.0005)  # Buy cost
        bh_sell_cost = test_df['close_price'].iloc[-1] * (0.001 + 0.0005)  # Sell cost
        bh_return = ((test_df['close_price'].iloc[-1] - test_df['close_price'].iloc[0] - bh_cost - bh_sell_cost) 
                     / test_df['close_price'].iloc[0])
        
        return {
            'test_metrics': test_metrics,
            'buy_hold_return': bh_return,
            'test_env': test_env
        }
    
    def _evaluate(self, agent, env):
        """Evaluate agent"""
        state = env.reset()
        
        while True:
            action = agent.select_action(state, training=False)
            next_state, reward, done, info = env.step(action)
            
            if done:
                break
            state = next_state
        
        return env.calculate_metrics()
    
    def compare_stock(self, symbol, num_episodes=100):
        """Compare both models on one stock"""
        
        print(f"\n{'='*70}")
        print(f"{symbol} - DAILY DATA, {num_episodes} EPISODES, WITH TRANSACTION COSTS")
        print('='*70)
        
        df = self.load_daily_data(symbol)
        
        if df is None or len(df) < 500:
            print(f"  ⚠️  Insufficient data (need 500+ days)")
            return None
        
        # Train Standard DQN
        print(f"\n  🤖 Training Standard DQN...")
        start = time.time()
        standard_result = self.train_agent(df, 'standard', num_episodes)
        std_time = time.time() - start
        
        # Train Attention DQN
        print(f"  🧠 Training Attention DQN...")
        start = time.time()
        attention_result = self.train_agent(df, 'attention', num_episodes)
        att_time = time.time() - start
        
        # Store results
        self.results['standard'][symbol] = standard_result
        self.results['attention'][symbol] = attention_result
        
        # Display results
        std_ret = standard_result['test_metrics']['total_return'] * 100
        att_ret = attention_result['test_metrics']['total_return'] * 100
        bh_ret = standard_result['buy_hold_return'] * 100
        
        std_costs = standard_result['test_metrics']['total_costs']
        att_costs = attention_result['test_metrics']['total_costs']
        
        print(f"\n  {'='*70}")
        print(f"  RESULTS:")
        print(f"  {'='*70}")
        print(f"  Standard DQN:   {std_ret:7.2f}% (Costs: ${std_costs:,.2f}, Trades: {standard_result['test_metrics']['total_trades']})")
        print(f"  Attention DQN:  {att_ret:7.2f}% (Costs: ${att_costs:,.2f}, Trades: {attention_result['test_metrics']['total_trades']})")
        print(f"  Buy & Hold:     {bh_ret:7.2f}% (Costs: minimal)")
        print(f"  ")
        print(f"  Winner: {'🧠 ATTENTION' if att_ret > std_ret else '🤖 STANDARD'}")
        print(f"  {'='*70}")
        
        return {'standard': standard_result, 'attention': attention_result}

if __name__ == "__main__":
    print("="*80)
    print("DAILY DQN COMPARISON - 100 EPISODES WITH TRANSACTION COSTS")
    print("="*80)
    print("\nFeatures:")
    print("  ✓ Daily trading data (2,500+ days per stock)")
    print("  ✓ 100 training episodes per model")
    print("  ✓ Transaction costs: 0.1% commission + 0.05% slippage")
    print("  ✓ Realistic position management")
    print("="*80)
    
    comparator = DailyDQNComparison()
    
    # Get available stocks
    csv_files = glob.glob('stock_data/*.csv')
    available_stocks = sorted([os.path.basename(f).replace('.csv', '') for f in csv_files])
    
    print(f"\n✓ Found {len(available_stocks)} stocks")
    
    # Select stocks to compare
    response = input(f"\nHow many stocks to compare? (1-{len(available_stocks)}): ")
    
    try:
        n = int(response)
        stocks_to_compare = available_stocks[:n]
    except:
        print("Invalid input. Using 3 stocks.")
        stocks_to_compare = available_stocks[:3]
    
    print(f"\n🎯 Comparing {len(stocks_to_compare)} stocks: {', '.join(stocks_to_compare)}")
    print(f"⏱️  Estimated time: ~{len(stocks_to_compare) * 10} minutes")
    print("="*80)
    
    stocks_compared = {}
    start_time = time.time()
    
    for i, symbol in enumerate(stocks_to_compare, 1):
        print(f"\n[{i}/{len(stocks_to_compare)}]")
        result = comparator.compare_stock(symbol, num_episodes=100)
        if result:
            stocks_compared[symbol] = result
    
    total_time = time.time() - start_time
    
    # Summary
    print("\n" + "="*80)
    print("COMPARISON COMPLETE!")
    print("="*80)
    print(f"⏱️  Total time: {total_time/60:.1f} minutes")
    print(f"✓ Stocks compared: {len(stocks_compared)}")
    
    # Stats
    symbols = list(stocks_compared.keys())
    std_returns = [comparator.results['standard'][s]['test_metrics']['total_return'] * 100 for s in symbols]
    att_returns = [comparator.results['attention'][s]['test_metrics']['total_return'] * 100 for s in symbols]
    bh_returns = [comparator.results['standard'][s]['buy_hold_return'] * 100 for s in symbols]
    
    att_wins = sum(1 for i in range(len(symbols)) if att_returns[i] > std_returns[i])
    
    print(f"\n📊 Average Standard DQN:  {np.mean(std_returns):6.2f}%")
    print(f"📊 Average Attention DQN: {np.mean(att_returns):6.2f}%")
    print(f"📊 Average Buy & Hold:    {np.mean(bh_returns):6.2f}%")
    print(f"🏆 Attention wins: {att_wins}/{len(symbols)} stocks")
    
    # Save results
    results_dict = {
        'config': {
            'data_frequency': 'DAILY',
            'num_episodes': 100,
            'commission_pct': 0.1,
            'slippage_pct': 0.05
        },
        'stocks': symbols,
        'standard_returns': std_returns,
        'attention_returns': att_returns,
        'buy_hold_returns': bh_returns
    }
    
    with open('daily_100ep_results.json', 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print("\n✓ Results saved to: daily_100ep_results.json")
    print("="*80)
