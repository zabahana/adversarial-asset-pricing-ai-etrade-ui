import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from google.cloud import bigquery
from protagonist_dqn import ProtagonistDQN, TradingEnvironment
import json
from datetime import datetime

class ModelAnalyzer:
    """Comprehensive model analysis and visualization"""
    
    def __init__(self, project_id, checkpoint_path):
        self.project_id = project_id
        self.checkpoint_path = checkpoint_path
        self.client = bigquery.Client(project=project_id)
    
    def load_trained_model(self):
        """Load the trained model"""
        print("Loading trained model...")
        agent = ProtagonistDQN(state_size=13, action_size=3)
        agent.load_checkpoint(self.checkpoint_path)
        print(f"✓ Model loaded (epsilon={agent.epsilon:.3f})")
        return agent
    
    def load_test_data(self, symbol='AAPL'):
        """Load test data"""
        query = f"""
        SELECT 
            symbol, timestamp, close_price, sma_5, sma_10, sma_20,
            price_change_1d, price_change_5d, volatility_10d
        FROM `{self.project_id}.processed_market_data.technical_indicators`
        WHERE symbol = '{symbol}'
        ORDER BY timestamp ASC
        """
        
        df = self.client.query(query).to_dataframe()
        df = df.fillna(0)
        df['returns'] = df['close_price'].pct_change().fillna(0)
        df['momentum'] = df['close_price'] - df['close_price'].shift(5).fillna(df['close_price'])
        df['volatility_ratio'] = df['volatility_10d'] / (df['volatility_10d'].rolling(20).mean().fillna(1) + 1e-6)
        
        # Use last 15% as test set
        test_start = int(len(df) * 0.85)
        test_df = df.iloc[test_start:].copy()
        
        print(f"✓ Test data: {len(test_df)} records")
        return test_df
    
    def evaluate_on_test(self, agent, test_df, initial_capital=100000):
        """Detailed test evaluation"""
        env = TradingEnvironment(test_df, initial_capital=initial_capital)
        state = env.reset()
        
        actions_taken = []
        q_values_history = []
        
        while True:
            # Get Q-values
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_vals = agent.q_network(state_tensor).numpy()[0]
                q_values_history.append(q_vals)
            
            action = agent.select_action(state, training=False)
            actions_taken.append(action)
            
            next_state, reward, done, info = env.step(action)
            
            if done:
                break
            state = next_state
        
        metrics = env.calculate_metrics()
        
        # Calculate buy-and-hold baseline
        buy_hold_return = (test_df['close_price'].iloc[-1] - test_df['close_price'].iloc[0]) / test_df['close_price'].iloc[0]
        
        return {
            'metrics': metrics,
            'portfolio_history': env.portfolio_history,
            'trade_history': env.trade_history,
            'actions': actions_taken,
            'q_values': q_values_history,
            'buy_hold_return': buy_hold_return,
            'env': env
        }
    
    def create_comprehensive_report(self, agent, test_df, results):
        """Create comprehensive analysis report"""
        
        fig = plt.figure(figsize=(20, 14))
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
        fig.suptitle('Protagonist DQN: Comprehensive Performance Analysis', 
                    fontsize=18, fontweight='bold')
        
        metrics = results['metrics']
        
        # 1. Portfolio Value Over Time
        ax1 = fig.add_subplot(gs[0, :])
        portfolio = results['portfolio_history']
        ax1.plot(portfolio, linewidth=2.5, color='green', label='DQN Agent', alpha=0.8)
        ax1.axhline(100000, color='red', linestyle='--', linewidth=2, label='Initial Capital')
        
        # Buy-and-hold comparison
        buy_hold = 100000 * (1 + test_df['close_price'].pct_change().fillna(0)).cumprod()
        ax1.plot(buy_hold.values, linewidth=2.5, color='blue', linestyle='--', 
                label='Buy & Hold', alpha=0.6)
        
        ax1.fill_between(range(len(portfolio)), 100000, portfolio, 
                        alpha=0.3, color='green')
        ax1.set_title('Portfolio Value: DQN vs Buy & Hold', fontweight='bold', fontsize=14)
        ax1.set_xlabel('Trading Days')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend(fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Add performance annotations
        final_value = portfolio[-1]
        max_value = max(portfolio)
        ax1.text(len(portfolio)*0.7, max_value*0.95, 
                f'Final: ${final_value:,.0f}\nReturn: {metrics["total_return"]*100:.1f}%',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
                fontsize=11, fontweight='bold')
        
        # 2. Performance Metrics Comparison
        ax2 = fig.add_subplot(gs[1, 0])
        
        dqn_metrics = [
            metrics['total_return'] * 100,
            metrics['sharpe_ratio'],
            abs(metrics['max_drawdown']) * 100,
            metrics['total_trades'] / 10  # Normalize for display
        ]
        
        bh_return = results['buy_hold_return'] * 100
        bh_sharpe = (test_df['returns'].mean() / test_df['returns'].std()) * np.sqrt(252)
        bh_dd = abs(self._calc_drawdown(buy_hold.values)) * 100
        
        baseline_metrics = [bh_return, bh_sharpe, bh_dd, 0.2]  # B&H has minimal trades
        
        x = np.arange(4)
        width = 0.35
        
        bars1 = ax2.bar(x - width/2, dqn_metrics, width, label='DQN Agent', 
                       color='green', alpha=0.7)
        bars2 = ax2.bar(x + width/2, baseline_metrics, width, label='Buy & Hold',
                       color='blue', alpha=0.7)
        
        ax2.set_title('Performance Metrics Comparison', fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(['Return\n(%)', 'Sharpe\nRatio', 'Max DD\n(%)', 'Trades\n(×10)'])
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}', ha='center', va='bottom', fontsize=9)
        
        # 3. Action Distribution
        ax3 = fig.add_subplot(gs[1, 1])
        actions = results['actions']
        action_counts = [actions.count(i) for i in range(3)]
        action_names = ['HOLD', 'BUY', 'SELL']
        colors = ['gray', 'green', 'red']
        
        wedges, texts, autotexts = ax3.pie(action_counts, labels=action_names, colors=colors,
                                           autopct='%1.1f%%', startangle=90)
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        ax3.set_title('Action Distribution', fontweight='bold')
        
        # 4. Q-Values Evolution
        ax4 = fig.add_subplot(gs[1, 2])
        q_vals = np.array(results['q_values'])
        for i, name in enumerate(action_names):
            ax4.plot(q_vals[:, i], label=name, linewidth=2, color=colors[i], alpha=0.7)
        
        ax4.set_title('Q-Values Over Time', fontweight='bold')
        ax4.set_xlabel('Time Step')
        ax4.set_ylabel('Q-Value')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Price Chart with Trades
        ax5 = fig.add_subplot(gs[2, :])
        ax5.plot(test_df['close_price'].values, linewidth=2, color='blue', 
                alpha=0.7, label='AAPL Price')
        
        # Mark trades
        for trade in results['trade_history']:
            if trade['action'] == 'BUY':
                ax5.scatter(trade['step'], trade['price'], color='green', 
                           marker='^', s=150, zorder=5, edgecolors='black', linewidths=1.5)
            else:
                ax5.scatter(trade['step'], trade['price'], color='red', 
                           marker='v', s=150, zorder=5, edgecolors='black', linewidths=1.5)
        
        ax5.set_title('Trading Actions on Test Data', fontweight='bold', fontsize=14)
        ax5.set_xlabel('Trading Days')
        ax5.set_ylabel('AAPL Price ($)')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Drawdown Analysis
        ax6 = fig.add_subplot(gs[3, 0])
        drawdowns = self._calc_drawdown_series(portfolio)
        ax6.fill_between(range(len(drawdowns)), 0, drawdowns*100, 
                        alpha=0.5, color='red', label='Drawdown')
        ax6.plot(drawdowns*100, linewidth=2, color='darkred')
        ax6.set_title('Drawdown Over Time', fontweight='bold')
        ax6.set_xlabel('Trading Days')
        ax6.set_ylabel('Drawdown (%)')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # 7. Daily Returns Distribution
        ax7 = fig.add_subplot(gs[3, 1])
        daily_returns = np.diff(portfolio) / portfolio[:-1] * 100
        ax7.hist(daily_returns, bins=30, alpha=0.7, color='purple', edgecolor='black')
        ax7.axvline(daily_returns.mean(), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {daily_returns.mean():.2f}%')
        ax7.set_title('Daily Returns Distribution', fontweight='bold')
        ax7.set_xlabel('Return (%)')
        ax7.set_ylabel('Frequency')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        
        # 8. Performance Summary Table
        ax8 = fig.add_subplot(gs[3, 2])
        ax8.axis('off')
        
        summary_data = [
            ['Metric', 'DQN Agent', 'Buy & Hold'],
            ['Total Return', f"{metrics['total_return']*100:.2f}%", f"{bh_return:.2f}%"],
            ['Final Value', f"${metrics['final_value']:,.0f}", f"${buy_hold.iloc[-1]:,.0f}"],
            ['Sharpe Ratio', f"{metrics['sharpe_ratio']:.2f}", f"{bh_sharpe:.2f}"],
            ['Max Drawdown', f"{metrics['max_drawdown']*100:.2f}%", f"{bh_dd:.2f}%"],
            ['Total Trades', str(metrics['total_trades']), '2'],
            ['Win Rate', f"{self._calc_win_rate(results['trade_history']):.1f}%", 'N/A']
        ]
        
        table = ax8.table(cellText=summary_data, cellLoc='center', loc='center',
                         bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style header row
        for i in range(3):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Style alternating rows
        for i in range(1, len(summary_data)):
            for j in range(3):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')
        
        ax8.set_title('Performance Summary', fontweight='bold', fontsize=12, pad=20)
        
        plt.tight_layout()
        plt.savefig('comprehensive_model_analysis.png', dpi=300, bbox_inches='tight')
        print("\n✓ Saved: comprehensive_model_analysis.png")
    
    def _calc_drawdown(self, values):
        """Calculate maximum drawdown"""
        cumulative = np.array(values)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def _calc_drawdown_series(self, values):
        """Calculate drawdown time series"""
        cumulative = np.array(values)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return drawdown
    
    def _calc_win_rate(self, trades):
        """Calculate win rate from trades"""
        if len(trades) < 2:
            return 0
        
        profits = []
        for i in range(1, len(trades)):
            if trades[i-1]['action'] == 'BUY' and trades[i]['action'] == 'SELL':
                profit = trades[i]['price'] - trades[i-1]['price']
                profits.append(profit > 0)
        
        return (sum(profits) / len(profits) * 100) if profits else 0
    
    def save_text_summary(self, agent, results, test_df):
        """Save text summary to file"""
        metrics = results['metrics']
        
        summary = f"""
================================================================================
PROTAGONIST DQN MODEL SUMMARY
================================================================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

MODEL ARCHITECTURE:
-------------------
State Size: 13 features
Action Space: 3 actions (HOLD, BUY, SELL)
Network: 
  - Input: 13 neurons
  - Hidden Layer 1: 128 neurons (ReLU + Dropout 0.2)
  - Hidden Layer 2: 128 neurons (ReLU + Dropout 0.2)
  - Hidden Layer 3: 64 neurons (ReLU + Dropout 0.2)
  - Output: 3 neurons (Q-values)

Total Parameters: {sum(p.numel() for p in agent.q_network.parameters()):,}
Trainable Parameters: {sum(p.numel() for p in agent.q_network.parameters() if p.requires_grad):,}

TRAINING DATA:
--------------
Symbol: AAPL
Total Records: {len(test_df)}
Date Range: {test_df['timestamp'].iloc[0]} to {test_df['timestamp'].iloc[-1]}
Price Range: ${test_df['close_price'].min():.2f} - ${test_df['close_price'].max():.2f}

TEST PERFORMANCE:
-----------------
Total Return: {metrics['total_return']*100:.2f}%
Final Portfolio Value: ${metrics['final_value']:,.2f}
Sharpe Ratio: {metrics['sharpe_ratio']:.2f}
Maximum Drawdown: {metrics['max_drawdown']*100:.2f}%
Total Trades: {metrics['total_trades']}
Win Rate: {self._calc_win_rate(results['trade_history']):.1f}%

BENCHMARK COMPARISON (Buy & Hold):
-----------------------------------
DQN Return: {metrics['total_return']*100:.2f}%
Buy & Hold Return: {results['buy_hold_return']*100:.2f}%
Outperformance: {(metrics['total_return'] - results['buy_hold_return'])*100:.2f}%

ACTION DISTRIBUTION:
--------------------
HOLD: {results['actions'].count(0)} ({results['actions'].count(0)/len(results['actions'])*100:.1f}%)
BUY:  {results['actions'].count(1)} ({results['actions'].count(1)/len(results['actions'])*100:.1f}%)
SELL: {results['actions'].count(2)} ({results['actions'].count(2)/len(results['actions'])*100:.1f}%)

EXPLORATION PARAMETER:
----------------------
Final Epsilon: {agent.epsilon:.4f}

KEY INSIGHTS:
-------------
1. The DQN agent {'outperformed' if metrics['total_return'] > results['buy_hold_return'] else 'underperformed'} the buy-and-hold strategy
2. Risk-adjusted return (Sharpe): {metrics['sharpe_ratio']:.2f} ({'Excellent' if metrics['sharpe_ratio'] > 2 else 'Good' if metrics['sharpe_ratio'] > 1 else 'Moderate' if metrics['sharpe_ratio'] > 0.5 else 'Needs Improvement'})
3. Maximum drawdown was {abs(metrics['max_drawdown'])*100:.1f}% ({'Acceptable' if abs(metrics['max_drawdown']) < 0.2 else 'High Risk'})
4. Agent executed {metrics['total_trades']} trades (avg {metrics['total_trades']/len(results['actions']):.2f} trades/day)

================================================================================
"""
        
        with open('model_summary.txt', 'w') as f:
            f.write(summary)
        
        print("✓ Saved: model_summary.txt")
        return summary

if __name__ == "__main__":
    PROJECT_ID = "ambient-isotope-463716-u6"
    CHECKPOINT = "protagonist_dqn_aapl.pth"
    
    print("=" * 80)
    print("CREATING COMPREHENSIVE MODEL ANALYSIS")
    print("=" * 80)
    
    analyzer = ModelAnalyzer(PROJECT_ID, CHECKPOINT)
    
    # Load model and data
    agent = analyzer.load_trained_model()
    test_df = analyzer.load_test_data('AAPL')
    
    # Evaluate on test set
    print("\nEvaluating model on test data...")
    results = analyzer.evaluate_on_test(agent, test_df)
    
    print(f"\nTest Results:")
    print(f"  Return: {results['metrics']['total_return']*100:.2f}%")
    print(f"  Sharpe: {results['metrics']['sharpe_ratio']:.2f}")
    print(f"  Max DD: {results['metrics']['max_drawdown']*100:.2f}%")
    print(f"  Trades: {results['metrics']['total_trades']}")
    
    # Create visualizations
    print("\nGenerating comprehensive report...")
    analyzer.create_comprehensive_report(agent, test_df, results)
    
    # Save text summary
    summary = analyzer.save_text_summary(agent, results, test_df)
    print(summary)
    
    print("\n" + "=" * 80)
    print("✓ ANALYSIS COMPLETE!")
    print("=" * 80)
    print("\nGenerated files:")
    print("  📊 comprehensive_model_analysis.png - Visual analysis")
    print("  📄 model_summary.txt - Text summary")
    print("  💾 protagonist_dqn_aapl.pth - Model checkpoint")
    print("=" * 80)
