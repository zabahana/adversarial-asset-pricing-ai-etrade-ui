"""
Simplified comparison - just generate the report from existing results
"""
import numpy as np

# Your results from the successful run
results = {
    'AAPL': {'standard': 13.50, 'attention': -4.74, 'buy_hold': 45.03},
    'GOOGL': {'standard': 39.26, 'attention': 32.06, 'buy_hold': 46.33},
    'MSFT': {'standard': 26.12, 'attention': 19.58, 'buy_hold': 32.70},
    'NVDA': {'standard': 14.96, 'attention': 55.71, 'buy_hold': 112.08},
    'TSLA': {'standard': 35.86, 'attention': 68.06, 'buy_hold': 125.61}
}

print("\n" + "="*80)
print("DQN COMPARISON RESULTS")
print("="*80)

for stock, res in results.items():
    winner = "ATTENTION" if res['attention'] > res['standard'] else "STANDARD"
    print(f"\n{stock}:")
    print(f"  Standard DQN:  {res['standard']:6.2f}%")
    print(f"  Attention DQN: {res['attention']:6.2f}%")
    print(f"  Buy & Hold:    {res['buy_hold']:6.2f}%")
    print(f"  Winner: {winner}")

std_avg = np.mean([r['standard'] for r in results.values()])
att_avg = np.mean([r['attention'] for r in results.values()])
att_wins = sum(1 for r in results.values() if r['attention'] > r['standard'])

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"Average Standard DQN:  {std_avg:6.2f}%")
print(f"Average Attention DQN: {att_avg:6.2f}%")
print(f"Attention wins: {att_wins}/5 stocks")
print("="*80)
