"""
Collect 30-stock portfolio - MONTHLY data for 10 YEARS
Optimal for DQN training: Less noise, strategic decisions
"""
import yfinance as yf
import pandas as pd
import json
from google.cloud import pubsub_v1
import time

PORTFOLIO_30 = {
    'Technology': ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META', 'TSLA', 'ORCL'],
    'Financial': ['JPM', 'BAC', 'GS', 'V', 'MA'],
    'Healthcare': ['JNJ', 'UNH', 'PFE', 'LLY'],
    'Consumer Discretionary': ['AMZN', 'HD', 'NKE'],
    'Consumer Staples': ['WMT', 'PG'],
    'Energy': ['XOM', 'CVX'],
    'Industrial': ['CAT', 'BA'],
    'Communication': ['DIS', 'NFLX'],
    'Utilities': ['NEE'],
    'Real Estate & Materials': ['PLD', 'NEM']
}

class MonthlyCollector:
    def __init__(self, project_id):
        self.project_id = project_id
        self.publisher = pubsub_v1.PublisherClient()
        self.topic_path = self.publisher.topic_path(project_id, "market-data-raw")
    
    def collect_monthly_data(self, symbol, period="10y"):
        """Collect and aggregate to monthly data"""
        try:
            print(f"\n📊 {symbol}...", end=" ")
            ticker = yf.Ticker(symbol)
            
            # Get daily data
            daily_df = ticker.history(period=period)
            
            if daily_df.empty:
                print(f"❌ No data")
                return 0
            
            # Aggregate to monthly (end of month)
            monthly_df = pd.DataFrame({
                'open': daily_df['Open'].resample('ME').first(),
                'high': daily_df['High'].resample('ME').max(),
                'low': daily_df['Low'].resample('ME').min(),
                'close': daily_df['Close'].resample('ME').last(),
                'volume': daily_df['Volume'].resample('ME').sum(),
            }).dropna()
            
            # Calculate monthly returns and volatility
            monthly_df['monthly_return'] = monthly_df['close'].pct_change()
            daily_df['daily_return'] = daily_df['Close'].pct_change()
            monthly_volatility = daily_df['daily_return'].resample('ME').std()
            monthly_df['volatility'] = monthly_volatility
            
            monthly_df = monthly_df.reset_index()
            
            print(f"✓ {len(monthly_df)} months ({monthly_df['Date'].min().strftime('%Y-%m')} to {monthly_df['Date'].max().strftime('%Y-%m')})", end=" ")
            
            # Publish each month to Pub/Sub
            published = 0
            for _, row in monthly_df.iterrows():
                data_point = {
                    'symbol': symbol,
                    'timestamp': row['Date'].isoformat() + 'Z',
                    'close_price': float(row['close']),
                    'volume': int(row['volume']),
                    'data_source': 'monthly_10year'
                }
                
                message = json.dumps(data_point).encode('utf-8')
                future = self.publisher.publish(self.topic_path, message)
                future.result()
                published += 1
            
            print(f"→ {published} published")
            return published
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return 0
    
    def collect_all_stocks(self):
        """Collect monthly data for all 30 stocks"""
        print("=" * 80)
        print("COLLECTING 30-STOCK PORTFOLIO - MONTHLY DATA (10 YEARS)")
        print("=" * 80)
        
        all_stocks = [stock for sector in PORTFOLIO_30.values() for stock in sector]
        
        print(f"\nStocks: {len(all_stocks)}")
        print(f"Period: 10 years")
        print(f"Frequency: Monthly")
        print(f"Est. records per stock: ~120 months")
        print(f"Est. total records: ~{len(all_stocks) * 120:,}")
        print(f"Est. time: ~{len(all_stocks) * 0.5:.0f} minutes")
        print("=" * 80)
        
        results = {}
        total_published = 0
        
        for sector, stocks in PORTFOLIO_30.items():
            print(f"\n{sector.upper()}")
            print("-" * 80)
            
            for symbol in stocks:
                published = self.collect_monthly_data(symbol, period="10y")
                
                if published > 0:
                    results[symbol] = {
                        'sector': sector,
                        'months': published
                    }
                    total_published += published
                
                time.sleep(1)  # Rate limiting
        
        # Summary
        print("\n" + "=" * 80)
        print("COLLECTION SUMMARY")
        print("=" * 80)
        print(f"✓ Stocks collected: {len(results)}/{len(all_stocks)}")
        print(f"✓ Total months published: {total_published:,}")
        print(f"✓ Average per stock: {total_published/len(results):.0f} months")
        
        print("\n" + "=" * 80)
        print("BY SECTOR:")
        print("=" * 80)
        for sector in PORTFOLIO_30.keys():
            sector_stocks = [s for s in results if results[s]['sector'] == sector]
            sector_months = sum(results[s]['months'] for s in sector_stocks)
            if sector_stocks:
                print(f"{sector:25s}: {len(sector_stocks):2d} stocks | {sector_months:4d} months")
        
        print("\n" + "=" * 80)
        print("✓ MONTHLY DATA COLLECTION COMPLETE!")
        print("=" * 80)
        print("\n⏰ Processing time: ~5-10 minutes")
        print("   Then ready for DQN training!")
        print("=" * 80)
        
        return results

if __name__ == "__main__":
    PROJECT_ID = "ambient-isotope-463716-u6"
    
    print("\n📅 MONTHLY DATA COLLECTION")
    print("   10 years × 12 months = ~120 data points per stock")
    print("   30 stocks × 120 months = ~3,600 total records")
    print("   Benefits: Less noise, strategic decisions, faster training\n")
    
    collector = MonthlyCollector(PROJECT_ID)
    results = collector.collect_all_stocks()
    
    print("\n✅ Ready for attention-enhanced DQN training!")
