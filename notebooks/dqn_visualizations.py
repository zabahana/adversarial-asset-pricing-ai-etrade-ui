import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from google.cloud import bigquery
from datetime import datetime, timedelta

class DQNDataVisualizer:
    def __init__(self, project_id):
        self.client = bigquery.Client(project=project_id)
        self.project_id = project_id
        
    def load_processed_data(self, symbol="AAPL", days=365):
        """Load processed features from BigQuery"""
        query = f"""
        SELECT 
            symbol,
            timestamp,
            close_price,
            volume,
            sma_5,
            sma_10,
            sma_20,
            price_change_1d,
            price_change_5d,
            volatility_10d
        FROM `{self.project_id}.processed_market_data.technical_indicators`
        WHERE symbol = '{symbol}'
        AND timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {days} DAY)
        ORDER BY timestamp ASC
        """
        
        df = self.client.query(query).to_dataframe()
        print(f"✓ Loaded {len(df)} records for {symbol}")
        return df
    
    def check_all_symbols(self):
        """Check what data is available for all symbols"""
        query = f"""
        SELECT 
            symbol,
            COUNT(*) as record_count,
            MIN(timestamp) as first_date,
            MAX(timestamp) as last_date
        FROM `{self.project_id}.processed_market_data.technical_indicators`
        GROUP BY symbol
        ORDER BY symbol
        """
        
        df = self.client.query(query).to_dataframe()
        print("\n" + "=" * 80)
        print("AVAILABLE DATA BY SYMBOL")
        print("=" * 80)
        for _, row in df.iterrows():
            print(f"{row['symbol']:6s}: {row['record_count']:>6,} records | "
                  f"{row['first_date'].strftime('%Y-%m-%d')} to {row['last_date'].strftime('%Y-%m-%d')}")
        print("=" * 80)
        return df
    
    def create_state_representation(self, df, window_size=10):
        """Show how raw data becomes DQN state"""
        
        fig, axes = plt.subplots(4, 1, figsize=(16, 12))
        fig.suptitle('From Raw Data to DQN State Representation', 
                     fontsize=16, fontweight='bold', y=0.995)
        
        sample_data = df.tail(50).copy()
        
        # 1. Raw Price Data
        ax = axes[0]
        ax.plot(range(len(sample_data)), sample_data['close_price'].values, 
                linewidth=2, color='blue', label='Close Price')
        ax.fill_between(range(len(sample_data)), sample_data['close_price'].values, 
                        alpha=0.3, color='blue')
        ax.set_title('Step 1: Raw Price Time Series', fontweight='bold', fontsize=12)
        ax.set_ylabel('Price ($)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Technical Indicators
        ax = axes[1]
        ax.plot(range(len(sample_data)), sample_data['close_price'].values, 
                linewidth=2, label='Close Price', color='blue')
        if 'sma_5' in sample_data.columns:
            sma5_clean = sample_data['sma_5'].dropna()
            if len(sma5_clean) > 0:
                ax.plot(range(len(sample_data))[-len(sma5_clean):], sma5_clean.values, 
                        linewidth=1.5, label='SMA-5', linestyle='--', color='orange')
        if 'sma_20' in sample_data.columns:
            sma20_clean = sample_data['sma_20'].dropna()
            if len(sma20_clean) > 0:
                ax.plot(range(len(sample_data))[-len(sma20_clean):], sma20_clean.values, 
                        linewidth=1.5, label='SMA-20', linestyle='--', color='red')
        ax.set_title('Step 2: Technical Indicators Computed', fontweight='bold', fontsize=12)
        ax.set_ylabel('Price ($)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Normalized Features
        ax = axes[2]
        features = ['close_price', 'sma_20', 'volatility_10d']
        for feature in features:
            if feature in sample_data.columns:
                data = sample_data[feature].dropna()
                if len(data) > 0 and data.max() > data.min():
                    normalized = (data - data.min()) / (data.max() - data.min())
                    ax.plot(range(len(data)), normalized.values, 
                           linewidth=1.5, label=feature, alpha=0.8)
        ax.set_title('Step 3: Features Normalized [0, 1]', fontweight='bold', fontsize=12)
        ax.set_ylabel('Normalized Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. State Vector as Heatmap
        ax = axes[3]
        state_features = ['close_price', 'sma_20', 'price_change_1d', 
                         'price_change_5d', 'volatility_10d']
        
        recent = sample_data.tail(window_size)[state_features].dropna()
        
        if len(recent) > 0:
            state_matrix = (recent - recent.min()) / (recent.max() - recent.min() + 1e-8)
            sns.heatmap(state_matrix.T, ax=ax, cmap='RdYlGn', 
                       cbar_kws={'label': 'Normalized Value'},
                       xticklabels=[f't-{window_size-i-1}' for i in range(len(state_matrix))],
                       yticklabels=state_features, annot=False)
            ax.set_title(f'Step 4: DQN State Vector (Window={window_size} timesteps × {len(state_features)} features)', 
                        fontweight='bold', fontsize=12)
        
        plt.tight_layout()
        plt.savefig('dqn_state_representation.png', dpi=300, bbox_inches='tight')
        print("\n✓ Saved: dqn_state_representation.png")
    
    def create_trading_framework_viz(self, df):
        """Create trading framework visualization"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('DQN Trading Framework Overview', fontsize=16, fontweight='bold')
        
        sample = df.tail(200).copy()
        
        # 1. Price Chart
        ax = axes[0, 0]
        ax.plot(range(len(sample)), sample['close_price'].values, linewidth=2, color='blue')
        ax.fill_between(range(len(sample)), sample['close_price'].values, alpha=0.3, color='blue')
        ax.set_title('🌍 Market Environment', fontweight='bold')
        ax.set_ylabel('Price ($)')
        ax.grid(True, alpha=0.3)
        
        # 2. Technical Indicators
        ax = axes[0, 1]
        ax.plot(range(len(sample)), sample['close_price'].values, linewidth=2, label='Price', color='blue')
        if 'sma_20' in sample.columns:
            sma_clean = sample['sma_20'].dropna()
            if len(sma_clean) > 0:
                ax.plot(range(len(sample))[-len(sma_clean):], sma_clean.values, 
                       linewidth=2, label='SMA-20', linestyle='--', color='orange')
        ax.set_title('📊 State Features', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Q-Values (simulated)
        ax = axes[0, 2]
        actions = ['HOLD', 'BUY', 'SELL']
        q_values = [0.6, 0.85, 0.3]
        colors = ['gray', 'green', 'red']
        ax.barh(actions, q_values, color=colors, alpha=0.7)
        ax.axvline(max(q_values), color='gold', linestyle='--', linewidth=2, label='Best Action')
        ax.set_title('💎 Q-Values', fontweight='bold')
        ax.set_xlabel('Expected Return')
        ax.legend()
        
        # 4. Portfolio Value (simulated)
        ax = axes[1, 0]
        returns = sample['price_change_1d'].fillna(0) / 100
        portfolio = 10000 * (1 + returns).cumprod()
        ax.plot(range(len(portfolio)), portfolio.values, linewidth=2, color='green', alpha=0.8)
        ax.fill_between(range(len(portfolio)), 10000, portfolio.values, alpha=0.3, color='green')
        ax.axhline(10000, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_title('📈 Portfolio Value', fontweight='bold')
        ax.set_ylabel('Value ($)')
        ax.grid(True, alpha=0.3)
        
        # 5. Rewards (simulated)
        ax = axes[1, 1]
        rewards = np.random.randn(len(sample)) * 2 + 0.5
        colors_reward = ['green' if r > 0 else 'red' for r in rewards]
        ax.bar(range(len(rewards)), rewards, color=colors_reward, alpha=0.6, width=1)
        ax.axhline(0, color='black', linestyle='-', linewidth=1)
        ax.set_title('🎁 Reward Signal', fontweight='bold')
        ax.set_ylabel('Reward')
        ax.grid(True, alpha=0.3)
        
        # 6. Training Progress (simulated)
        ax = axes[1, 2]
        episodes = range(1, 51)
        cumulative_return = 100 + np.cumsum(np.random.randn(50) * 5 + 2)
        ax.plot(episodes, cumulative_return, linewidth=2, color='purple', marker='o', markersize=4)
        ax.fill_between(episodes, 100, cumulative_return, alpha=0.3, color='purple')
        ax.set_title('📊 Training Progress', fontweight='bold')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Cumulative Return (%)')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('dqn_trading_framework.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: dqn_trading_framework.png")

if __name__ == "__main__":
    print("=" * 80)
    print("DQN DATA VISUALIZATION")
    print("=" * 80)
    
    visualizer = DQNDataVisualizer("ambient-isotope-463716-u6")
    
    # Check all available data
    visualizer.check_all_symbols()
    
    print("\n1. Loading processed data from BigQuery...")
    df = visualizer.load_processed_data(symbol="AAPL", days=365)
    
    if len(df) > 0:
        print("\n2. Creating state representation visualization...")
        visualizer.create_state_representation(df, window_size=10)
        
        print("\n3. Creating trading framework visualization...")
        visualizer.create_trading_framework_viz(df)
        
        print("\n" + "=" * 80)
        print("VISUALIZATION COMPLETE!")
        print("=" * 80)
        print("\nGenerated files:")
        print("  📊 dqn_state_representation.png")
        print("  🧠 dqn_trading_framework.png")
        print("=" * 80)
    else:
        print("\n❌ No data available for AAPL. Check other symbols.")
