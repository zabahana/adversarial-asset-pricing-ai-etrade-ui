import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from google.cloud import bigquery
from datetime import datetime, timedelta

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class DataExplorer:
    def __init__(self, project_id):
        self.client = bigquery.Client(project=project_id)
        self.project_id = project_id
        
    def load_data(self, days=30):
        """Load market data from BigQuery"""
        query = f"""
        SELECT 
            r.symbol,
            r.timestamp,
            r.close_price,
            r.volume,
            p.sma_5,
            p.sma_10,
            p.sma_20,
            p.price_change_1d,
            p.price_change_5d,
            p.volatility_10d
        FROM `{self.project_id}.raw_market_data.daily_prices` r
        LEFT JOIN `{self.project_id}.processed_market_data.technical_indicators` p
            ON r.symbol = p.symbol AND DATE(r.timestamp) = DATE(p.timestamp)
        WHERE r.timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {days} DAY)
        ORDER BY r.symbol, r.timestamp
        """
        
        df = self.client.query(query).to_dataframe()
        return df
    
    def descriptive_statistics(self, df):
        """Generate descriptive statistics"""
        print("=" * 80)
        print("DESCRIPTIVE STATISTICS")
        print("=" * 80)
        
        print("\nDataset Shape:", df.shape)
        print("\nFeature Types:")
        print(df.dtypes)
        
        print("\nBasic Statistics:")
        print(df.describe())
        
        print("\nMissing Values:")
        print(df.isnull().sum())
        
        print("\nSymbol Distribution:")
        print(df['symbol'].value_counts())
        
        return df.describe()
    
    def visualize_price_trends(self, df):
        """Visualize price trends for each symbol"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Price Trends by Symbol', fontsize=16, fontweight='bold')
        
        symbols = df['symbol'].unique()[:4]
        
        for idx, symbol in enumerate(symbols):
            ax = axes[idx // 2, idx % 2]
            symbol_data = df[df['symbol'] == symbol].sort_values('timestamp')
            
            ax.plot(symbol_data['timestamp'], symbol_data['close_price'], 
                   linewidth=2, label='Close Price')
            
            if 'sma_20' in symbol_data.columns:
                ax.plot(symbol_data['timestamp'], symbol_data['sma_20'], 
                       linewidth=1.5, alpha=0.7, label='SMA(20)')
            
            ax.set_title(f'{symbol} Price Trend', fontweight='bold')
            ax.set_xlabel('Date')
            ax.set_ylabel('Price ($)')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('price_trends.png', dpi=300, bbox_inches='tight')
        print("\n✓ Saved: price_trends.png")
        
    def visualize_technical_indicators(self, df):
        """Visualize technical indicators"""
        symbols = df['symbol'].unique()
        
        for symbol in symbols[:2]:  # First 2 symbols
            symbol_data = df[df['symbol'] == symbol].sort_values('timestamp')
            
            fig, axes = plt.subplots(3, 1, figsize=(15, 12))
            fig.suptitle(f'{symbol} Technical Analysis', fontsize=16, fontweight='bold')
            
            # Price and Moving Averages
            ax1 = axes[0]
            ax1.plot(symbol_data['timestamp'], symbol_data['close_price'], 
                    label='Close Price', linewidth=2)
            ax1.plot(symbol_data['timestamp'], symbol_data['sma_5'], 
                    label='SMA(5)', alpha=0.7)
            ax1.plot(symbol_data['timestamp'], symbol_data['sma_10'], 
                    label='SMA(10)', alpha=0.7)
            ax1.plot(symbol_data['timestamp'], symbol_data['sma_20'], 
                    label='SMA(20)', alpha=0.7)
            ax1.set_ylabel('Price ($)')
            ax1.set_title('Price and Moving Averages')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Price Changes
            ax2 = axes[1]
            ax2.bar(symbol_data['timestamp'], symbol_data['price_change_1d'], 
                   alpha=0.6, label='1-Day Change %')
            ax2.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
            ax2.set_ylabel('Price Change (%)')
            ax2.set_title('Daily Price Changes')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Volatility
            ax3 = axes[2]
            ax3.plot(symbol_data['timestamp'], symbol_data['volatility_10d'], 
                    color='red', linewidth=2)
            ax3.fill_between(symbol_data['timestamp'], symbol_data['volatility_10d'], 
                           alpha=0.3, color='red')
            ax3.set_ylabel('Volatility (%)')
            ax3.set_xlabel('Date')
            ax3.set_title('10-Day Volatility')
            ax3.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f'technical_indicators_{symbol}.png', dpi=300, bbox_inches='tight')
            print(f"✓ Saved: technical_indicators_{symbol}.png")
    
    def correlation_analysis(self, df):
        """Analyze feature correlations"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        correlation_matrix = df[numeric_cols].corr()
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, fmt='.2f', 
                   cmap='coolwarm', center=0, square=True,
                   linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
        print("\n✓ Saved: correlation_matrix.png")
        
        print("\nHighly Correlated Features (|r| > 0.7):")
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                if abs(correlation_matrix.iloc[i, j]) > 0.7:
                    print(f"  {correlation_matrix.columns[i]} <-> {correlation_matrix.columns[j]}: {correlation_matrix.iloc[i, j]:.3f}")
    
    def distribution_analysis(self, df):
        """Analyze feature distributions"""
        numeric_cols = ['close_price', 'volume', 'price_change_1d', 'volatility_10d']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Feature Distributions', fontsize=16, fontweight='bold')
        
        for idx, col in enumerate(numeric_cols):
            if col in df.columns:
                ax = axes[idx // 2, idx % 2]
                df[col].hist(bins=30, ax=ax, edgecolor='black', alpha=0.7)
                ax.set_title(f'{col} Distribution', fontweight='bold')
                ax.set_xlabel(col)
                ax.set_ylabel('Frequency')
                ax.grid(True, alpha=0.3)
                
                # Add mean and median lines
                mean_val = df[col].mean()
                median_val = df[col].median()
                ax.axvline(mean_val, color='red', linestyle='--', 
                          linewidth=2, label=f'Mean: {mean_val:.2f}')
                ax.axvline(median_val, color='green', linestyle='--', 
                          linewidth=2, label=f'Median: {median_val:.2f}')
                ax.legend()
        
        plt.tight_layout()
        plt.savefig('feature_distributions.png', dpi=300, bbox_inches='tight')
        print("\n✓ Saved: feature_distributions.png")

if __name__ == "__main__":
    print("Starting Data Exploration...")
    print("=" * 80)
    
    explorer = DataExplorer("ambient-isotope-463716-u6")
    
    # Load data
    print("\n1. Loading data from BigQuery...")
    df = explorer.load_data(days=30)
    
    # Descriptive statistics
    print("\n2. Generating descriptive statistics...")
    stats = explorer.descriptive_statistics(df)
    
    # Visualizations
    print("\n3. Creating visualizations...")
    explorer.visualize_price_trends(df)
    explorer.visualize_technical_indicators(df)
    explorer.correlation_analysis(df)
    explorer.distribution_analysis(df)
    
    print("\n" + "=" * 80)
    print("Data Exploration Complete!")
    print("=" * 80)
