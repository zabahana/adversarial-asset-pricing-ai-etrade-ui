"""
Visualize DQN Comparison Results - Separated Plots and Table
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datetime import datetime

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'

# Your results from the successful run
results = {
    'AAPL': {'standard': 13.50, 'attention': -4.74, 'buy_hold': 45.03},
    'GOOGL': {'standard': 39.26, 'attention': 32.06, 'buy_hold': 46.33},
    'MSFT': {'standard': 26.12, 'attention': 19.58, 'buy_hold': 32.70},
    'NVDA': {'standard': 14.96, 'attention': 55.71, 'buy_hold': 112.08},
    'TSLA': {'standard': 35.86, 'attention': 68.06, 'buy_hold': 125.61}
}

# Extract data
symbols = list(results.keys())
std_returns = [results[s]['standard'] for s in symbols]
att_returns = [results[s]['attention'] for s in symbols]
bh_returns = [results[s]['buy_hold'] for s in symbols]

# Calculate statistics
avg_std = np.mean(std_returns)
avg_att = np.mean(att_returns)
avg_bh = np.mean(bh_returns)
att_wins = sum(1 for i in range(len(symbols)) if att_returns[i] > std_returns[i])
std_wins = len(symbols) - att_wins

# ============================================================================
# FIGURE 1: CHARTS AND PLOTS
# ============================================================================
fig1 = plt.figure(figsize=(18, 10))
gs1 = fig1.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
fig1.suptitle('Standard DQN vs Attention-Enhanced DQN Comparison\n5 Stocks, 30 Episodes Each, Monthly Data', 
             fontsize=16, fontweight='bold', y=0.98)

# 1. Returns by Stock (Main Chart)
ax1 = fig1.add_subplot(gs1[0, :])
x = np.arange(len(symbols))
width = 0.25

bars1 = ax1.bar(x - width, std_returns, width, label='Standard DQN', 
                color='#3498db', alpha=0.85, edgecolor='black', linewidth=1.5)
bars2 = ax1.bar(x, att_returns, width, label='Attention DQN', 
                color='#2ecc71', alpha=0.85, edgecolor='black', linewidth=1.5)
bars3 = ax1.bar(x + width, bh_returns, width, label='Buy & Hold', 
                color='#95a5a6', alpha=0.7, edgecolor='black', linewidth=1)

ax1.set_title('Test Returns by Stock', fontweight='bold', fontsize=14, pad=15)
ax1.set_ylabel('Return (%)', fontsize=12, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(symbols, fontsize=11, fontweight='bold')
ax1.legend(fontsize=11, loc='upper left', framealpha=0.9)
ax1.grid(True, alpha=0.3, axis='y')
ax1.axhline(0, color='black', linestyle='-', linewidth=1.5)

# Add value labels
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', 
                va='bottom' if height >= 0 else 'top', 
                fontsize=9, fontweight='bold')

# 2. Average Performance
ax2 = fig1.add_subplot(gs1[1, 0])
models = ['Standard\nDQN', 'Attention\nDQN', 'Buy &\nHold']
avgs = [avg_std, avg_att, avg_bh]
colors_avg = ['#3498db', '#2ecc71', '#95a5a6']

bars = ax2.bar(models, avgs, color=colors_avg, alpha=0.85, 
               edgecolor='black', linewidth=2)
ax2.set_title('Average Return', fontweight='bold', fontsize=13)
ax2.set_ylabel('Return (%)', fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

for bar, val in zip(bars, avgs):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.1f}%', ha='center', va='bottom', 
            fontsize=13, fontweight='bold')

# Highlight winner
winner_idx = np.argmax(avgs)
bars[winner_idx].set_edgecolor('#f39c12')
bars[winner_idx].set_linewidth(4)

# 3. Win Rate
ax3 = fig1.add_subplot(gs1[1, 1])
sizes = [att_wins, std_wins]
colors = ['#2ecc71', '#3498db']
explode = (0.05, 0.05)

wedges, texts, autotexts = ax3.pie(sizes, explode=explode,
                                     labels=['Attention\nWins', 'Standard\nWins'],
                                     colors=colors, autopct='%1.0f%%',
                                     startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
    autotext.set_fontsize(16)

ax3.set_title(f'Win Rate (Total: {len(symbols)} stocks)', 
              fontweight='bold', fontsize=13)

# 4. Performance vs Buy & Hold
ax4 = fig1.add_subplot(gs1[1, 2])
std_vs_bh = [std_returns[i] - bh_returns[i] for i in range(len(symbols))]
att_vs_bh = [att_returns[i] - bh_returns[i] for i in range(len(symbols))]

x = np.arange(len(symbols))
width = 0.35

bars1 = ax4.bar(x - width/2, std_vs_bh, width, label='Standard', 
                color='#3498db', alpha=0.8)
bars2 = ax4.bar(x + width/2, att_vs_bh, width, label='Attention', 
                color='#2ecc71', alpha=0.8)

ax4.set_title('Outperformance vs Buy & Hold', fontweight='bold', fontsize=13)
ax4.set_ylabel('Difference (%)', fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(symbols, fontsize=10)
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3, axis='y')
ax4.axhline(0, color='red', linestyle='--', linewidth=2, alpha=0.7)

plt.tight_layout()
plt.savefig('dqn_comparison_charts.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
print("✓ Saved: dqn_comparison_charts.png")
plt.close()

# ============================================================================
# FIGURE 2: COMPARISON TABLE
# ============================================================================
fig2 = plt.figure(figsize=(14, 6))
ax_table = fig2.add_subplot(111)

fig2.suptitle('Detailed Stock-by-Stock Comparison', 
              fontsize=16, fontweight='bold', y=0.95)

# Create comparison table
cell_text = []
for symbol in symbols:
    std = results[symbol]['standard']
    att = results[symbol]['attention']
    bh = results[symbol]['buy_hold']
    winner = '🧠 Attention' if att > std else '🤖 Standard'
    margin = abs(att - std)
    
    cell_text.append([
        symbol,
        f'{std:.2f}%',
        f'{att:.2f}%',
        f'{bh:.2f}%',
        f'{margin:.2f}%',
        winner
    ])

# Add separator
cell_text.append(['─' * 8, '─' * 12, '─' * 12, '─' * 12, '─' * 10, '─' * 15])

# Add averages
cell_text.append([
    'AVERAGE',
    f'{avg_std:.2f}%',
    f'{avg_att:.2f}%',
    f'{avg_bh:.2f}%',
    f'{abs(avg_att - avg_std):.2f}%',
    f'Att: {att_wins}/{len(symbols)}'
])

columns = ['Stock', 'Standard DQN', 'Attention DQN', 'Buy & Hold', 'Margin', 'Winner']
table = ax_table.table(cellText=cell_text, colLabels=columns,
                       cellLoc='center', loc='center',
                       colWidths=[0.12, 0.18, 0.18, 0.18, 0.14, 0.20])

table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 3)

# Style header
for i in range(len(columns)):
    table[(0, i)].set_facecolor('#34495e')
    table[(0, i)].set_text_props(weight='bold', color='white', fontsize=13)
    table[(0, i)].set_height(0.08)

# Style rows
for i in range(1, len(cell_text) + 1):
    if '─' in cell_text[i-1][0]:  # Separator row
        for j in range(len(columns)):
            table[(i, j)].set_facecolor('#bdc3c7')
            table[(i, j)].set_height(0.03)
    elif i == len(cell_text):  # Average row
        for j in range(len(columns)):
            table[(i, j)].set_facecolor('#f39c12')
            table[(i, j)].set_text_props(weight='bold', fontsize=12, color='white')
            table[(i, j)].set_height(0.08)
    else:
        # Alternate row colors
        color = '#ecf0f1' if i % 2 == 0 else 'white'
        for j in range(len(columns)):
            table[(i, j)].set_facecolor(color)
            table[(i, j)].set_height(0.07)
            
        # Highlight winner column
        winner_cell = table[(i, len(columns)-1)]
        if '🧠' in cell_text[i-1][len(columns)-1]:
            winner_cell.set_facecolor('#d5f4e6')
            winner_cell.set_text_props(weight='bold')
        else:
            winner_cell.set_facecolor('#d6eaf8')
            winner_cell.set_text_props(weight='bold')

ax_table.axis('off')

# Add summary stats below table
summary_text = f"""
Summary Statistics:
- Attention DQN Wins: {att_wins}/{len(symbols)} stocks ({att_wins/len(symbols)*100:.0f}%)
- Average Improvement: {avg_att - avg_std:+.2f}% (Attention vs Standard)
- Best Standard: {symbols[np.argmax(std_returns)]} ({max(std_returns):.2f}%)
- Best Attention: {symbols[np.argmax(att_returns)]} ({max(att_returns):.2f}%)
"""

ax_table.text(0.5, 0.05, summary_text, ha='center', va='top',
              fontsize=11, fontfamily='monospace',
              bbox=dict(boxstyle='round', facecolor='#ecf0f1', alpha=0.8))

plt.tight_layout()
plt.savefig('dqn_comparison_table.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
print("✓ Saved: dqn_comparison_table.png")
plt.close()

# Create text report
report = f"""
{'='*80}
DQN COMPARISON REPORT: STANDARD vs ATTENTION-ENHANCED
{'='*80}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Training: 30 episodes per model per stock
Data: Monthly resampled data (121 datapoints per stock)

STOCKS TESTED: 5
AAPL, GOOGL, MSFT, NVDA, TSLA

{'='*80}
AVERAGE PERFORMANCE
{'='*80}
Standard DQN:    {avg_std:7.2f}% average return
Attention DQN:   {avg_att:7.2f}% average return
Buy & Hold:      {avg_bh:7.2f}% average return

Improvement:     {avg_att - avg_std:+7.2f}% (Attention vs Standard)

{'='*80}
WIN RATE
{'='*80}
Attention DQN wins:  {att_wins} / {len(symbols)} stocks ({att_wins/len(symbols)*100:.0f}%)
Standard DQN wins:   {std_wins} / {len(symbols)} stocks ({std_wins/len(symbols)*100:.0f}%)

{'='*80}
STOCK-BY-STOCK BREAKDOWN
{'='*80}
"""

for symbol in symbols:
    std = results[symbol]['standard']
    att = results[symbol]['attention']
    bh = results[symbol]['buy_hold']
    winner = "ATTENTION ✓" if att > std else "STANDARD ✓"
    margin = abs(att - std)
    
    report += f"""
{symbol}:
  Standard DQN:    {std:7.2f}%
  Attention DQN:   {att:7.2f}%
  Buy & Hold:      {bh:7.2f}%
  Winner:          {winner} (by {margin:.2f}%)
  vs Buy & Hold:   Std {std-bh:+.2f}% | Att {att-bh:+.2f}%
"""

report += f"""
{'='*80}
KEY INSIGHTS
{'='*80}
"""

# Add insights
if att_wins > std_wins:
    report += f"\n✓ Attention-Enhanced DQN outperforms Standard DQN ({att_wins}/{len(symbols)} stocks)\n"
else:
    report += f"\n✓ Standard DQN outperforms Attention-Enhanced DQN ({std_wins}/{len(symbols)} stocks)\n"

if avg_att > avg_std:
    report += f"✓ Average improvement of {avg_att - avg_std:.2f}% with attention mechanism\n"
else:
    report += f"✓ Standard DQN has {avg_std - avg_att:.2f}% higher average return\n"

# Best performers
best_std_idx = np.argmax(std_returns)
best_att_idx = np.argmax(att_returns)
report += f"✓ Best Standard DQN: {symbols[best_std_idx]} ({std_returns[best_std_idx]:.2f}%)\n"
report += f"✓ Best Attention DQN: {symbols[best_att_idx]} ({att_returns[best_att_idx]:.2f}%)\n"

# Attention's biggest wins
att_margins = [(symbols[i], att_returns[i] - std_returns[i]) for i in range(len(symbols))]
att_margins.sort(key=lambda x: x[1], reverse=True)
report += f"\n✓ Attention's biggest advantage: {att_margins[0][0]} (+{att_margins[0][1]:.2f}%)\n"

report += f"\n{'='*80}\n"

with open('dqn_comparison_report.txt', 'w') as f:
    f.write(report)

print("✓ Saved: dqn_comparison_report.txt")
print(report)

print("\n" + "="*80)
print("✓ VISUALIZATION AND REPORT COMPLETE!")
print("="*80)
print("\nGenerated files:")
print("  📊 dqn_comparison_charts.png   (4 comparison charts)")
print("  📋 dqn_comparison_table.png    (detailed table)")
print("  📄 dqn_comparison_report.txt   (text summary)")
print("="*80)
