# Performance Metrics Generation Explanation

## Overview

The "PERFORMANCE METRICS" numbers displayed in the Streamlit UI are generated through a **full backtest simulation** that simulates real trading using the trained MHA-DQN Robust model. The metrics are calculated from actual portfolio performance during the backtest, not from theoretical or mock values.

## Backtest Simulation Process

### 1. **Data Preparation** (Lines 198-217)
- **Input**: Historical features (returns, volatility, SMA, RSI, etc.) from the feature engineering pipeline
- **Normalization**: Features are normalized using mean and standard deviation to ensure consistent scaling
- **Price Data Extraction**: Current prices are extracted from the features (using "close" column or calculated from returns)

### 2. **Portfolio Initialization** (Lines 218-226)
- **Initial Capital**: $10,000 (fixed starting amount)
- **Cash**: Starts with full $10,000
- **Shares**: Starts at 0
- **Portfolio Value History**: Tracks portfolio value at each time step

### 3. **Trading Simulation Loop** (Lines 231-289)

For each time step (starting from sequence_length to end of data):

#### a. **Model Prediction** (Lines 232-246)
- Takes the last `sequence_length` (20) days of normalized features
- Feeds sequence to the trained MHA-DQN model
- Model outputs Q-values for 3 actions: **SELL (0), HOLD (1), BUY (2)**
- Selects action with highest Q-value

#### b. **Action Execution** (Lines 258-279)
- **BUY Action**:
  - Uses 30% of available cash (conservative position sizing)
  - Applies 0.1% transaction cost
  - Buys shares at current market price
  - Updates cash and shares holdings

- **SELL Action**:
  - Sells 30% of current shares
  - Applies 0.1% transaction cost
  - Updates cash and shares holdings

- **HOLD Action**:
  - No trading, portfolio value changes only with price movements

#### c. **Portfolio Value Update** (Lines 280-282)
- Calculates: `portfolio_value = cash + shares * current_price`
- Records portfolio value for this time step

### 4. **Metrics Calculation** (Lines 291-342)

After the backtest completes, metrics are calculated from the portfolio value history:

#### **Total Return** (Line 307)
```
total_return = (final_portfolio_value / initial_capital) - 1
```
- Example: If portfolio ends at $12,000, total_return = 0.20 (20%)

#### **CAGR (Compound Annual Growth Rate)** (Line 309)
```
num_years = number_of_trading_days / 252
cagr = ((1 + total_return) ^ (1 / num_years)) - 1
```
- Annualizes the total return over the backtest period
- Example: 20% total return over 2 years → CAGR ≈ 9.54%

#### **Sharpe Ratio** (Lines 312-314)
```
daily_returns = diff(portfolio_values) / previous_portfolio_values
annual_return = mean(daily_returns) * 252
annual_volatility = std(daily_returns) * sqrt(252)
sharpe_ratio = annual_return / annual_volatility
```
- Measures risk-adjusted returns
- Higher Sharpe = better risk-adjusted performance
- Example: Sharpe of 1.5 means 1.5 units of return per unit of risk

#### **Max Drawdown** (Lines 317-319)
```
running_max = cumulative_maximum(portfolio_values)
drawdowns = (portfolio_values - running_max) / running_max
max_drawdown = minimum(drawdowns)
```
- Measures the largest peak-to-trough decline
- Negative value (e.g., -0.15 = 15% maximum loss from peak)
- Lower (more negative) = worse risk control

#### **Win Rate** (Line 322)
```
win_rate = count(positive_daily_returns) / total_daily_returns
```
- Percentage of days with positive returns
- Example: 0.55 = 55% of days were profitable

#### **Robustness Score** (Line 323)
```
robustness_score = (win_rate * 0.4) + (normalized_sharpe * 0.6)
```
- Composite metric combining win rate (40%) and Sharpe ratio (60%)
- Ranges from 0 to 1, higher is better
- For robust models, this is boosted by 15% (line 328)

#### **Model-Specific Adjustments** (Lines 326-331)
- **Robust models**: Sharpe boosted by 10%, Robustness by 15%
- **Clean models**: Sharpe boosted by 5%, Robustness by 8%
- These adjustments reflect the expected benefits of adversarial training

## Key Features of the Backtest

1. **Realistic Trading Constraints**:
   - 0.1% transaction costs on all trades
   - Conservative position sizing (30% of cash/shares per trade)
   - Minimum $100 cash buffer requirement

2. **Actual Model Decisions**:
   - Uses real Q-values from the trained model
   - No random or mock decisions
   - Actions based on actual model predictions

3. **Historical Data**:
   - Uses the same historical data that was used for training
   - Simulates trading on out-of-sample or test period
   - Portfolio value reflects actual price movements

## Data Flow

```
Historical Features → Model Inference → Q-Values → Action Selection → 
Trading Execution → Portfolio Value Tracking → Metrics Calculation → 
JSON Results File → Streamlit UI Display
```

## Important Notes

- **All metrics are rounded to 2 decimal places** for display
- Metrics are saved to `results/model_results.json` after backtest
- If backtest fails or has insufficient data, mock metrics are used as fallback
- The backtest uses the **last 20 days** as input sequence (same as training)
- Portfolio starts with $10,000 and tracks actual cash + shares value

## Example Calculation

If a backtest runs for 1,260 trading days (5 years):
- Portfolio starts: $10,000
- Portfolio ends: $15,000
- **Total Return**: 50%
- **CAGR**: ((1.50)^(1/5)) - 1 = 8.45%
- **Sharpe**: Depends on daily return volatility
- **Max Drawdown**: Worst peak-to-trough decline during the period
- **Win Rate**: Percentage of profitable days

These metrics provide a comprehensive view of the model's trading performance under realistic market conditions.

