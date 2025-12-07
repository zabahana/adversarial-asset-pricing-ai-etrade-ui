# Performance Metrics: Actual Data Analysis

## Current State Analysis

### ✅ What IS Using Actual Data

1. **Portfolio Values**: Calculated from:
   - Actual model Q-values → actual actions (BUY/SELL/HOLD)
   - Actual close prices from features DataFrame
   - Actual trading simulation with transaction costs

2. **Performance Metrics** (Sharpe, CAGR, Drawdown, etc.):
   - Calculated from actual portfolio value changes
   - Based on actual model actions and actual prices
   - NO synthetic data in these core metrics

3. **Actual Prices for Comparison**:
   - From historical close prices in features DataFrame
   - Real market data

### ❌ What IS Using Synthetic/Approximate Data

1. **Predicted Prices** (lines 422-446):
   - ESTIMATED from Q-values using formulas
   - Uses default `avg_return = 0.02` (synthetic)
   - Formula-based, not actual model price predictions
   - **Note**: These are ONLY for display plots, NOT for performance metrics

2. **Price Approximations** (lines 415-420, 461-464):
   - Approximates prices when close price not available
   - Uses returns to approximate - NOT actual data

3. **Mock Metrics Fallback** (lines 512-513, 521-523):
   - Falls back to mock metrics if insufficient data
   - These have `is_mock_data: True` flag
   - UI already filters these out (line 1529)

4. **Default Fallback Values**:
   - Line 438: `avg_return = 0.02` default
   - Line 436: Falls back to 0.02 if not enough historical data

## Conclusion

**The core performance metrics (Sharpe, CAGR, Max Drawdown, Total Return) ARE from actual model runs and actual data.** 

The synthetic/approximate data is only used for:
- Predicted prices for comparison plots (not used in metrics)
- Fallback when actual data unavailable

## Recommended Fixes

1. ✅ **Keep as-is for performance metrics** - they use actual data
2. ❌ **Remove synthetic predicted prices** - mark clearly or remove
3. ❌ **Fail gracefully if actual close prices unavailable** - don't approximate
4. ❌ **Remove mock metrics fallback** - already filtered by UI
5. ❌ **Remove default fallback values** - require actual data

## Verification

The UI already validates (line 1529):
- `is_actual_backtest == True`
- `is_mock_data == False`
- Has portfolio data

So mock metrics are NOT displayed. The performance metrics shown ARE from actual model runs.

