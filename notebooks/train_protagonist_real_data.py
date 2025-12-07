import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from google.cloud import bigquery
from protagonist_dqn import ProtagonistDQN, TradingEnvironment
import warnings
warnings.filterwarnings('ignore')

class RealDataTrainer:
    """Train protagonist DQN on real historical market data"""
    
    def __init__(self, project_id):
        self.project_id = project_id
        self.client = bigquery.Client(project=project_id)
    
    def load_real_data(self, symbol='AAPL', min_records=50):
        """Load real processed data from BigQuery"""
        
        query = f"""
        SELECT 
            symbol,
            timestamp,
            close_price,
            sma_5,
            sma_10,
            sma_20,
            price_change_1d,
            price_change_5d,
            volatility_10d
        FROM `{self.project_id}.processed_market_data.technical_indicators`
        WHERE symbol = '{symbol}'
        ORDER BY timestamp ASC
        """
        
        print(f"Loading real data for {symbol}...")
        df = self.client.query(query).to_dataframe()
        
        if len(df) < min_records:
            print(f"⚠️  Only {len(df)} records available for {symbol}")
            return None
        
        # Fill NaN values with 0 (or use forward fill)
        df = df.fillna(0)
        
        # Add derived features for state representation
        df['returns'] = df['close_price'].pct_change().fillna(0)
        df['momentum'] = df['close_price'] - df['close_price'].shift(5).fillna(df['close_price'])
        df['volatility_ratio'] = df['volatility_10d'] / (df['volatility_10d'].rolling(20).mean().fillna(1) + 1e-6)
        
        print(f"✓ Loaded {len(df)} real records for {symbol}")
        print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"  Price range: ${df['close_price'].min():.2f} to ${df['close_price'].max():.2f}")
        
        return df
    
    def prepare_data_splits(self, df, train_ratio=0.7, val_ratio=0.15):
        """Create chronological train/val/test splits"""
        
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy()
        test_df = df.iloc[val_end:].copy()
        
        print("\n" + "=" * 80)
        print("DATA SPLITS (CHRONOLOGICAL - NO LEAKAGE)")
        print("=" * 80)
        print(f"Training:   {len(train_df):4d} records | {train_df['timestamp'].min()} to {train_df['timestamp'].max()}")
        print(f"Validation: {len(val_df):4d} records | {val_df['timestamp'].min()} to {val_df['timestamp'].max()}")
        print(f"Test:       {len(test_df):4d} records | {test_df['timestamp'].min()} to {test_df['timestamp'].max()}")
        print("=" * 80)
        
        return train_df, val_df, test_df
    
    def train_episode(self, agent, env):
        """Train for one episode"""
        state = env.reset()
        total_reward = 0
        steps = 0
        
        while True:
            # Select action
            action = agent.select_action(state, training=True)
            
            # Take step
            next_state, reward, done, info = env.step(action)
            
            # Store experience
            agent.store_experience(state, action, reward, next_state, done)
            
            # Train
            loss = agent.train_step(batch_size=32)
            
            total_reward += reward
            steps += 1
            
            if done:
                break
            
            state = next_state
        
        # Decay epsilon
        agent.decay_epsilon()
        
        return total_reward, steps, env.calculate_metrics()
    
    def evaluate_episode(self, agent, env):
        """Evaluate agent (no training)"""
        state = env.reset()
        total_reward = 0
        
        while True:
            action = agent.select_action(state, training=False)  # Greedy
            next_state, reward, done, info = env.step(action)
            
            total_reward += reward
            
            if done:
                break
            
            state = next_state
        
        return total_reward, env.calculate_metrics()
    
    def train_agent(self, train_df, val_df, num_episodes=50, initial_capital=100000):
        """Complete training loop"""
        
        # Determine state size from features
        feature_cols = [col for col in train_df.columns 
                       if col not in ['timestamp', 'symbol']]
        state_size = len(feature_cols) + 3  # +3 for position features
        
        print(f"\nState size: {state_size} features")
        print(f"Features: {feature_cols}")
        
        # Create agent
        agent = ProtagonistDQN(
            state_size=state_size,
            action_size=3,
            learning_rate=0.0005,
            gamma=0.95,
            epsilon_start=1.0,
            epsilon_end=0.01,
            epsilon_decay=0.995,
            buffer_size=10000
        )
        
        # Training metrics
        train_returns = []
        val_returns = []
        train_sharpe = []
        val_sharpe = []
        
        print("\n" + "=" * 80)
        print("TRAINING PROTAGONIST DQN ON REAL DATA")
        print("=" * 80)
        
        for episode in range(num_episodes):
            # Train
            train_env = TradingEnvironment(train_df, initial_capital=initial_capital)
            train_reward, train_steps, train_metrics = self.train_episode(agent, train_env)
            train_returns.append(train_metrics['total_return'])
            train_sharpe.append(train_metrics['sharpe_ratio'])
            
            # Validate every 5 episodes
            if (episode + 1) % 5 == 0:
                val_env = TradingEnvironment(val_df, initial_capital=initial_capital)
                val_reward, val_metrics = self.evaluate_episode(agent, val_env)
                val_returns.append(val_metrics['total_return'])
                val_sharpe.append(val_metrics['sharpe_ratio'])
                
                print(f"Episode {episode+1:3d}/{num_episodes} | "
                      f"Train Return: {train_metrics['total_return']*100:6.2f}% | "
                      f"Val Return: {val_metrics['total_return']*100:6.2f}% | "
                      f"Sharpe: {val_metrics['sharpe_ratio']:5.2f} | "
                      f"ε: {agent.epsilon:.3f}")
            
            # Update target network every 10 episodes
            if (episode + 1) % 10 == 0:
                agent.update_target_network()
        
        print("=" * 80)
        print("✓ TRAINING COMPLETE")
        print("=" * 80)
        
        return agent, {
            'train_returns': train_returns,
            'val_returns': val_returns,
            'train_sharpe': train_sharpe,
            'val_sharpe': val_sharpe
        }
    
    def visualize_results(self, agent, test_df, history, initial_capital=100000):
        """Visualize training results and test performance"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Protagonist DQN: Real Market Data Performance', 
                    fontsize=16, fontweight='bold')
        
        # 1. Training returns
        ax = axes[0, 0]
        ax.plot(history['train_returns'], linewidth=2, label='Train', alpha=0.8)
        if history['val_returns']:
            val_episodes = [i*5 for i in range(len(history['val_returns']))]
            ax.plot(val_episodes, history['val_returns'], 
                   linewidth=2, label='Validation', marker='o', markersize=6)
        ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_title('Cumulative Returns', fontweight='bold')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Return (%)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Sharpe ratio
        ax = axes[0, 1]
        ax.plot(history['train_sharpe'], linewidth=2, label='Train', alpha=0.8)
        if history['val_sharpe']:
            val_episodes = [i*5 for i in range(len(history['val_sharpe']))]
            ax.plot(val_episodes, history['val_sharpe'], 
                   linewidth=2, label='Validation', marker='o', markersize=6)
        ax.axhline(1.0, color='green', linestyle='--', linewidth=2, label='Target')
        ax.set_title('Sharpe Ratio', fontweight='bold')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Sharpe Ratio')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Loss history
        ax = axes[0, 2]
        if agent.loss_history:
            ax.plot(agent.loss_history, linewidth=1, alpha=0.7)
            ax.set_title('Training Loss', fontweight='bold')
            ax.set_xlabel('Training Step')
            ax.set_ylabel('Loss')
            ax.grid(True, alpha=0.3)
        
        # 4. Test performance - portfolio value
        ax = axes[1, 0]
        test_env = TradingEnvironment(test_df, initial_capital=initial_capital)
        state = test_env.reset()
        
        while True:
            action = agent.select_action(state, training=False)
            next_state, reward, done, info = test_env.step(action)
            if done:
                break
            state = next_state
        
        ax.plot(test_env.portfolio_history, linewidth=2, color='green', alpha=0.8)
        ax.axhline(initial_capital, color='red', linestyle='--', linewidth=2, 
                  label=f'Initial: ${initial_capital:,.0f}')
        ax.fill_between(range(len(test_env.portfolio_history)), 
                        initial_capital, test_env.portfolio_history, 
                        alpha=0.3, color='green')
        ax.set_title('Test Set: Portfolio Value', fontweight='bold')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Portfolio Value ($)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 5. Test metrics
        ax = axes[1, 1]
        ax.axis('off')
        test_metrics = test_env.calculate_metrics()
        
        ax.text(0.5, 0.9, 'TEST PERFORMANCE', ha='center', 
               fontsize=14, fontweight='bold')
        
        metrics_text = [
            f"Total Return: {test_metrics['total_return']*100:.2f}%",
            f"Final Value: ${test_metrics['final_value']:,.2f}",
            f"Sharpe Ratio: {test_metrics['sharpe_ratio']:.2f}",
            f"Max Drawdown: {test_metrics['max_drawdown']*100:.2f}%",
            f"Total Trades: {test_metrics['total_trades']}"
        ]
        
        y_pos = 0.7
        for text in metrics_text:
            ax.text(0.5, y_pos, text, ha='center', fontsize=12, fontweight='bold')
            y_pos -= 0.12
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        # 6. Price vs Actions
        ax = axes[1, 2]
        ax.plot(test_df['close_price'].values, linewidth=2, alpha=0.7, label='Price')
        
        # Mark trades
        for trade in test_env.trade_history:
            color = 'green' if trade['action'] == 'BUY' else 'red'
            marker = '^' if trade['action'] == 'BUY' else 'v'
            ax.scatter(trade['step'], trade['price'], 
                      color=color, marker=marker, s=100, zorder=5)
        
        ax.set_title('Test Set: Trading Actions', fontweight='bold')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Price ($)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('protagonist_dqn_real_results.png', dpi=300, bbox_inches='tight')
        print("\n✓ Saved: protagonist_dqn_real_results.png")

if __name__ == "__main__":
    PROJECT_ID = "ambient-isotope-463716-u6"
    
    print("=" * 80)
    print("PROTAGONIST DQN: REAL HISTORICAL DATA TRAINING")
    print("=" * 80)
    
    trainer = RealDataTrainer(PROJECT_ID)
    
    # Load real data
    df = trainer.load_real_data(symbol='AAPL', min_records=100)
    
    if df is not None:
        # Split data
        train_df, val_df, test_df = trainer.prepare_data_splits(df)
        
        # Train agent
        agent, history = trainer.train_agent(
            train_df, val_df, 
            num_episodes=50,
            initial_capital=100000
        )
        
        # Visualize results
        print("\nCreating visualizations...")
        trainer.visualize_results(agent, test_df, history)
        
        # Save model
        agent.save_checkpoint('protagonist_dqn_aapl.pth')
        
        print("\n" + "=" * 80)
        print("✓ PROTAGONIST DQN TRAINED ON REAL DATA!")
        print("=" * 80)
    else:
        print("\n⚠️  Insufficient data. Need to collect more data first.")
