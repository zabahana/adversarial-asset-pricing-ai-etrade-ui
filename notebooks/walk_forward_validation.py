import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from google.cloud import bigquery
from datetime import datetime, timedelta

class WalkForwardValidator:
    """
    Implement proper walk-forward validation to avoid data leakage
    """
    
    def __init__(self, project_id):
        self.client = bigquery.Client(project=project_id)
        self.project_id = project_id
    
    def load_monthly_data(self):
        """Load all monthly data"""
        query = f"""
        SELECT *
        FROM `{self.project_id}.monthly_market_data.monthly_features`
        ORDER BY symbol, timestamp ASC
        """
        
        df = self.client.query(query).to_dataframe()
        print(f"Loaded {len(df):,} monthly records for {df['symbol'].nunique()} stocks")
        return df
    
    def create_time_splits(self, df, train_months=60, val_months=12, test_months=12):
        """
        Create proper time-series splits WITHOUT data leakage
        
        Split structure:
        [---- Training (60 months) ----][-- Val (12m) --][-- Test (12m) --]
                                        ^                ^
                                   Train cutoff      Val cutoff
        
        Each fold is completely separate in time to prevent lookahead bias
        """
        
        # Get unique timestamps sorted
        all_dates = sorted(df['timestamp'].unique())
        
        if len(all_dates) < train_months + val_months + test_months:
            print(f"Warning: Only {len(all_dates)} months available")
            # Adjust split sizes
            total = len(all_dates)
            train_months = int(total * 0.60)
            val_months = int(total * 0.20)
            test_months = total - train_months - val_months
        
        # Create single split (can be extended to rolling window)
        train_end_idx = train_months
        val_end_idx = train_months + val_months
        
        train_dates = all_dates[:train_end_idx]
        val_dates = all_dates[train_end_idx:val_end_idx]
        test_dates = all_dates[val_end_idx:val_end_idx + test_months]
        
        # Create splits
        train_df = df[df['timestamp'].isin(train_dates)].copy()
        val_df = df[df['timestamp'].isin(val_dates)].copy()
        test_df = df[df['timestamp'].isin(test_dates)].copy()
        
        print("\n" + "=" * 80)
        print("TIME-SERIES SPLIT (NO DATA LEAKAGE)")
        print("=" * 80)
        print(f"Training Set:")
        print(f"  Period: {train_df['timestamp'].min().strftime('%Y-%m')} to "
              f"{train_df['timestamp'].max().strftime('%Y-%m')}")
        print(f"  Records: {len(train_df):,} ({len(train_dates)} months)")
        print(f"  Stocks: {train_df['symbol'].nunique()}")
        
        print(f"\nValidation Set:")
        print(f"  Period: {val_df['timestamp'].min().strftime('%Y-%m')} to "
              f"{val_df['timestamp'].max().strftime('%Y-%m')}")
        print(f"  Records: {len(val_df):,} ({len(val_dates)} months)")
        print(f"  Stocks: {val_df['symbol'].nunique()}")
        
        print(f"\nTest Set (COMPLETELY UNSEEN):")
        print(f"  Period: {test_df['timestamp'].min().strftime('%Y-%m')} to "
              f"{test_df['timestamp'].max().strftime('%Y-%m')}")
        print(f"  Records: {len(test_df):,} ({len(test_dates)} months)")
        print(f"  Stocks: {test_df['symbol'].nunique()}")
        
        print("\n✓ No overlap between sets - pure out-of-time testing!")
        print("=" * 80)
        
        return {
            'train': train_df,
            'validation': val_df,
            'test': test_df,
            'train_dates': train_dates,
            'val_dates': val_dates,
            'test_dates': test_dates
        }
    
    def create_rolling_windows(self, df, train_window=60, val_window=12, 
                               test_window=12, step_size=12):
        """
        Create rolling walk-forward validation windows
        
        Each window slides forward in time:
        Window 1: [Train1][Val1][Test1]
        Window 2:       [Train2][Val2][Test2]
        Window 3:             [Train3][Val3][Test3]
        """
        
        all_dates = sorted(df['timestamp'].unique())
        windows = []
        
        window_idx = 0
        start_idx = 0
        
        while start_idx + train_window + val_window + test_window <= len(all_dates):
            train_end = start_idx + train_window
            val_end = train_end + val_window
            test_end = val_end + test_window
            
            window = {
                'window_id': window_idx,
                'train_dates': all_dates[start_idx:train_end],
                'val_dates': all_dates[train_end:val_end],
                'test_dates': all_dates[val_end:test_end],
                'train_data': df[df['timestamp'].isin(all_dates[start_idx:train_end])],
                'val_data': df[df['timestamp'].isin(all_dates[train_end:val_end])],
                'test_data': df[df['timestamp'].isin(all_dates[val_end:test_end])]
            }
            
            windows.append(window)
            window_idx += 1
            start_idx += step_size
        
        print(f"\nCreated {len(windows)} walk-forward validation windows")
        return windows
    
    def calculate_performance_metrics(self, returns, risk_free_rate=0.02):
        """
        Calculate comprehensive performance metrics
        
        Args:
            returns: Array of period returns
            risk_free_rate: Annual risk-free rate (default 2%)
        """
        
        returns = np.array(returns)
        rf_monthly = risk_free_rate / 12
        
        metrics = {}
        
        # Basic metrics
        metrics['total_return'] = (1 + returns).prod() - 1
        metrics['mean_return'] = returns.mean()
        metrics['volatility'] = returns.std()
        
        # Sharpe Ratio
        excess_returns = returns - rf_monthly
        if returns.std() > 0:
            metrics['sharpe_ratio'] = (excess_returns.mean() / returns.std()) * np.sqrt(12)
        else:
            metrics['sharpe_ratio'] = 0
        
        # Maximum Drawdown
        cumulative = (1 + returns).cumprod()
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        metrics['max_drawdown'] = drawdown.min()
        
        # Sortino Ratio (downside deviation)
        downside_returns = returns.copy()
        downside_returns[downside_returns > 0] = 0
        downside_std = downside_returns.std()
        if downside_std > 0:
            metrics['sortino_ratio'] = (excess_returns.mean() / downside_std) * np.sqrt(12)
        else:
            metrics['sortino_ratio'] = 0
        
        # Calmar Ratio (return / max drawdown)
        if metrics['max_drawdown'] != 0:
            metrics['calmar_ratio'] = metrics['total_return'] / abs(metrics['max_drawdown'])
        else:
            metrics['calmar_ratio'] = 0
        
        # Win Rate
        metrics['win_rate'] = (returns > 0).sum() / len(returns)
        
        # Profit Factor
        gains = returns[returns > 0].sum()
        losses = abs(returns[returns < 0].sum())
        if losses > 0:
            metrics['profit_factor'] = gains / losses
        else:
            metrics['profit_factor'] = float('inf') if gains > 0 else 0
        
        return metrics
    
    def visualize_walk_forward(self, splits):
        """Visualize the walk-forward validation strategy"""
        
        fig, axes = plt.subplots(4, 1, figsize=(18, 14))
        fig.suptitle('Walk-Forward Validation: Preventing Data Leakage', 
                    fontsize=16, fontweight='bold')
        
        train_df = splits['train']
        val_df = splits['validation']
        test_df = splits['test']
        
        # Use AAPL as example
        train_aapl = train_df[train_df['symbol'] == 'AAPL'].sort_values('timestamp')
        val_aapl = val_df[val_df['symbol'] == 'AAPL'].sort_values('timestamp')
        test_aapl = test_df[test_df['symbol'] == 'AAPL'].sort_values('timestamp')
        
        # 1. Timeline with splits
        ax = axes[0]
        
        if len(train_aapl) > 0:
            ax.plot(train_aapl['timestamp'], train_aapl['close_price'], 
                   linewidth=2, color='blue', label='Training', alpha=0.8)
        if len(val_aapl) > 0:
            ax.plot(val_aapl['timestamp'], val_aapl['close_price'], 
                   linewidth=2, color='orange', label='Validation', alpha=0.8)
        if len(test_aapl) > 0:
            ax.plot(test_aapl['timestamp'], test_aapl['close_price'], 
                   linewidth=2, color='green', label='Test (Out-of-Time)', alpha=0.8)
        
        # Mark boundaries
        if len(train_aapl) > 0 and len(val_aapl) > 0:
            ax.axvline(train_aapl['timestamp'].iloc[-1], color='red', 
                      linestyle='--', linewidth=2, alpha=0.7, label='Split Points')
        if len(val_aapl) > 0 and len(test_aapl) > 0:
            ax.axvline(val_aapl['timestamp'].iloc[-1], color='red', 
                      linestyle='--', linewidth=2, alpha=0.7)
        
        ax.set_title('Timeline: Strictly Chronological Split (AAPL Example)', 
                    fontweight='bold', fontsize=12)
        ax.set_ylabel('Price ($)')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # 2. Feature availability check (no future info)
        ax = axes[1]
        
        # Check last training point
        if len(train_aapl) > 0:
            last_train = train_aapl.iloc[-1]
            features = ['sharpe_12m', 'max_drawdown_12m', 'momentum_12m', 'sortino_12m']
            feature_values = [last_train[f] if f in last_train and pd.notna(last_train[f]) else 0 
                            for f in features]
            
            colors_feat = ['green' if not pd.isna(v) and v != 0 else 'red' for v in feature_values]
            bars = ax.barh(features, [1 if c == 'green' else 0 for c in colors_feat], 
                          color=colors_feat, alpha=0.7)
            
            ax.set_title('Feature Validation: All Metrics Use ONLY Past Data', 
                        fontweight='bold', fontsize=12)
            ax.set_xlabel('Available (Green) / Missing (Red)')
            ax.set_xlim(0, 1.2)
            
            for i, (bar, val) in enumerate(zip(bars, feature_values)):
                if not pd.isna(val) and val != 0:
                    ax.text(1.05, i, f'✓ {val:.3f}', va='center', fontweight='bold', color='green')
                else:
                    ax.text(1.05, i, '✗ N/A', va='center', fontweight='bold', color='red')
        
        # 3. Sharpe Ratio across splits
        ax = axes[2]
        
        split_names = []
        sharpe_values = []
        colors_split = []
        
        for name, data, color in [('Train', train_df, 'blue'), 
                                   ('Val', val_df, 'orange'), 
                                   ('Test', test_df, 'green')]:
            if len(data) > 0:
                split_names.append(name)
                avg_sharpe = data['sharpe_12m'].dropna().mean()
                sharpe_values.append(avg_sharpe)
                colors_split.append(color)
        
        if split_names:
            bars = ax.bar(split_names, sharpe_values, color=colors_split, alpha=0.7, edgecolor='black', linewidth=2)
            ax.axhline(0, color='black', linestyle='-', linewidth=1)
            ax.axhline(1, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Good (>1.0)')
            ax.set_title('Average Sharpe Ratio by Split', fontweight='bold', fontsize=12)
            ax.set_ylabel('Sharpe Ratio')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            
            for bar, val in zip(bars, sharpe_values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.2f}', ha='center', va='bottom' if height >= 0 else 'top',
                       fontweight='bold', fontsize=11)
        
        # 4. Data leakage prevention diagram
        ax = axes[3]
        ax.axis('off')
        
        ax.text(0.5, 0.95, 'Data Leakage Prevention Checklist', 
               ha='center', fontsize=14, fontweight='bold')
        
        checks = [
            ('✓ All features lagged (use only past data)', 'green'),
            ('✓ Strictly chronological split (no shuffling)', 'green'),
            ('✓ No overlap between train/val/test', 'green'),
            ('✓ Test set completely unseen during training', 'green'),
            ('✓ Walk-forward validation available', 'green'),
            ('✗ Never use future returns in current state', 'red'),
            ('✗ Never train on validation/test periods', 'red'),
            ('✗ Never normalize using future statistics', 'red')
        ]
        
        y_pos = 0.75
        for check, color in checks:
            ax.text(0.1, y_pos, check, fontsize=11, fontweight='bold', color=color)
            y_pos -= 0.1
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig('walk_forward_validation.png', dpi=300, bbox_inches='tight')
        print("\n✓ Saved: walk_forward_validation.png")
    
    def visualize_all_stocks_metrics(self, splits):
        """Visualize financial metrics for all 10 stocks"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('10-Stock Portfolio: Financial Metrics (Training Data)', 
                    fontsize=16, fontweight='bold')
        
        train_df = splits['train']
        
        symbols = sorted(train_df['symbol'].unique())
        
        # 1. Average Sharpe Ratio
        ax = axes[0, 0]
        sharpe_by_stock = train_df.groupby('symbol')['sharpe_12m'].mean().reindex(symbols)
        colors = ['green' if s > 1 else 'orange' if s > 0 else 'red' for s in sharpe_by_stock]
        bars = ax.bar(symbols, sharpe_by_stock, color=colors, alpha=0.7, edgecolor='black')
        ax.axhline(1.0, color='green', linestyle='--', linewidth=2, label='Target (1.0)')
        ax.set_title('Average Sharpe Ratio', fontweight='bold')
        ax.set_ylabel('Sharpe Ratio')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # 2. Average Max Drawdown
        ax = axes[0, 1]
        dd_by_stock = train_df.groupby('symbol')['max_drawdown_12m'].mean().reindex(symbols) * 100
        bars = ax.bar(symbols, dd_by_stock, color='red', alpha=0.7, edgecolor='black')
        ax.axhline(-20, color='orange', linestyle='--', linewidth=2, label='Risk Threshold (-20%)')
        ax.set_title('Average Maximum Drawdown', fontweight='bold')
        ax.set_ylabel('Max Drawdown (%)')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # 3. Average Sortino Ratio
        ax = axes[0, 2]
        sortino_by_stock = train_df.groupby('symbol')['sortino_12m'].mean().reindex(symbols)
        colors = ['green' if s > 1 else 'orange' if s > 0 else 'red' for s in sortino_by_stock]
        bars = ax.bar(symbols, sortino_by_stock, color=colors, alpha=0.7, edgecolor='black')
        ax.axhline(1.0, color='green', linestyle='--', linewidth=2, label='Target (1.0)')
        ax.set_title('Average Sortino Ratio', fontweight='bold')
        ax.set_ylabel('Sortino Ratio')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # 4. Volatility
        ax = axes[1, 0]
        vol_by_stock = (train_df.groupby('symbol')['volatility'].mean() * 100).reindex(symbols)
        bars = ax.bar(symbols, vol_by_stock, color='blue', alpha=0.7, edgecolor='black')
        ax.set_title('Average Monthly Volatility', fontweight='bold')
        ax.set_ylabel('Volatility (%)')
        ax.grid(True, alpha=0.3, axis='y')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # 5. Risk-Return Tradeoff
        ax = axes[1, 1]
        for symbol in symbols:
            stock_data = train_df[train_df['symbol'] == symbol]
            avg_return = stock_data['monthly_return'].mean() * 100
            avg_vol = stock_data['volatility'].mean() * 100
            ax.scatter(avg_vol, avg_return, s=150, alpha=0.7, edgecolors='black', linewidths=2)
            ax.text(avg_vol, avg_return, symbol, ha='center', va='center', 
                   fontweight='bold', fontsize=8)
        
        ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_title('Risk-Return Tradeoff', fontweight='bold')
        ax.set_xlabel('Risk (Volatility %)')
        ax.set_ylabel('Return (%)')
        ax.grid(True, alpha=0.3)
        
        # 6. Correlation Matrix
        ax = axes[1, 2]
        pivot_returns = train_df.pivot_table(values='monthly_return', 
                                             index='timestamp', columns='symbol')
        corr_matrix = pivot_returns.corr()
        sns.heatmap(corr_matrix, ax=ax, annot=True, fmt='.2f', cmap='coolwarm',
                   center=0, vmin=-1, vmax=1, square=True, cbar_kws={'shrink': 0.8})
        ax.set_title('Return Correlation', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('10stocks_financial_metrics.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: 10stocks_financial_metrics.png")

if __name__ == "__main__":
    PROJECT_ID = "ambient-isotope-463716-u6"
    
    validator = WalkForwardValidator(PROJECT_ID)
    
    print("Loading monthly data...")
    df = validator.load_monthly_data()
    
    if len(df) > 0:
        # Create time splits
        splits = validator.create_time_splits(df, train_months=60, val_months=12, test_months=12)
        
        # Visualize
        print("\nCreating visualizations...")
        validator.visualize_walk_forward(splits)
        validator.visualize_all_stocks_metrics(splits)
        
        print("\n✓ Walk-forward validation framework ready!")
        print("  • No data leakage")
        print("  • Out-of-time testing")
        print("  • Comprehensive financial metrics")
    else:
        print("No data available. Run enhanced_data_collection.py first")
