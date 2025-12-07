# Performance Metrics: Fix to Use Only Actual Model Data

## Issues Identified

1. **Synthetic Price Predictions** (lines 422-442):
   - Predicted prices are ESTIMATED from Q-values using formulas
   - Uses default `avg_return = 0.02` (2% daily volatility) - SYNTHETIC
   - Formula: `expected_return = (buy_prob - sell_prob) * avg_return * 1.5` - SYNTHETIC
   - `predicted_next_price = current_price * (1 + expected_return)` - SYNTHETIC

2. **Price Approximations** (lines 415-420, 462-463):
   - Approximates prices when close price isn't available
   - Uses returns to approximate prices - NOT ACTUAL DATA

3. **Mock Metrics Fallback** (lines 512-513, 521-523):
   - Falls back to mock metrics if data insufficient
   - These mock metrics have `is_mock_data: True` flag

4. **Default Fallback Values**:
   - Line 438: `avg_return = 0.02` (2% default) - SYNTHETIC

## What IS Using Actual Data

✅ **Portfolio Values**: Calculated from actual model actions (BUY/SELL/HOLD) and actual historical prices
✅ **Model Actions**: Directly from model Q-values (argmax)
✅ **Actual Prices**: From historical close prices in features
✅ **Returns**: Calculated from actual portfolio value changes

## What Needs to be Fixed

1. ❌ Remove synthetic price predictions - these are only for display/comparison plots
2. ❌ Remove default fallback values - fail gracefully if data insufficient
3. ❌ Remove mock metrics fallback - don't display if insufficient data
4. ❌ Remove price approximations - only use actual close prices
5. ✅ Keep portfolio values from actual model actions
6. ✅ Keep actual prices from historical data

## Solution

The model outputs Q-values for actions, not price predictions. So:
- Portfolio performance metrics ARE from actual model runs (actions + prices)
- Predicted prices for comparison plots are synthetic (can be removed or marked clearly)
- Need to ensure all prices come from actual close prices only
- Need to remove mock metrics fallback completely

