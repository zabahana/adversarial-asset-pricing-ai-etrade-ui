# Performance Metrics & Backtesting: Actual Data Validation

## Summary

This document outlines the changes made to ensure that the "Performance Metrics & Backtesting" section in the Streamlit UI **only displays results from actual model runs**, not mock data.

## Changes Made

### 1. Added Validation Flags to Metrics

**File**: `lightning_app/works/model_inference_work.py`

- **Actual Backtest Results**: Added flags to distinguish actual backtest results from mock data:
  - `"from_actual_backtest": True` - Indicates metrics come from actual model backtest
  - `"is_mock_data": False` - Confirms these are not mock metrics
  - `"backtest_period_days"` - Number of days in the backtest period

- **Mock Metrics**: Added flags to mark mock data:
  - `"from_actual_backtest": False`
  - `"is_mock_data": True`

### 2. Removed Artificial Adjustments

**File**: `lightning_app/works/model_inference_work.py` (lines 540-546)

- **Removed**: Artificial multipliers that adjusted Sharpe ratios and robustness scores based on model type:
  - Previously: Robust models had `sharpe *= 1.1`
  - Previously: Clean models had `sharpe *= 1.05`
  - Previously: Robustness scores were multiplied by 1.15 or 1.08

- **Now**: All metrics are computed **directly from backtest results** without any artificial adjustments. Metrics reflect actual model performance.

### 3. Enhanced UI Validation

**File**: `streamlit_app.py` (lines 1522-1529)

The UI now validates that metrics are from actual backtests before displaying:

```python
# Only display if we have actual backtest data (not mock)
is_actual_backtest = robust_metrics_perf.get('from_actual_backtest', False)
is_mock_data = robust_metrics_perf.get('is_mock_data', False)
has_portfolio_data = robust_metrics_perf.get('portfolio_values', []) and len(robust_metrics_perf.get('portfolio_values', [])) > 1
robust_has_data_perf = (robust_metrics_perf.get('sharpe', 0) != 0 or robust_metrics_perf.get('total_return', 0) != 0) and has_portfolio_data

# Only show performance metrics if we have actual backtest results (not mock)
if robust_has_data_perf and is_actual_backtest and not is_mock_data:
    # Display performance metrics...
```

### 4. Improved Logging

**File**: `lightning_app/works/model_inference_work.py`

- Added clear console logging to indicate when actual vs mock data is being used:
  - `✅ ACTUAL BACKTEST results from model run:` - When using actual data
  - `⚠️ Model not found, using MOCK metrics` - When using mock data
  - `⚠️ MOCK metrics generated - Performance Metrics section will NOT display these` - Warning that mock data won't be shown

## How It Works

1. **During Model Evaluation**:
   - If a model is successfully loaded: `_evaluate_model()` runs an actual backtest simulation
   - Backtest computes metrics from portfolio performance over the test period
   - Metrics are tagged with `from_actual_backtest: True` and `is_mock_data: False`

2. **If Model Not Available**:
   - `_get_mock_metrics()` generates placeholder metrics
   - Metrics are tagged with `from_actual_backtest: False` and `is_mock_data: True`
   - These metrics are saved but **will not be displayed** in the UI

3. **In the UI**:
   - When loading performance metrics, the UI checks:
     - ✅ `from_actual_backtest == True`
     - ✅ `is_mock_data == False`
     - ✅ Has portfolio values data
     - ✅ Has valid Sharpe ratio or total return
   - Only if all conditions are met, the Performance Metrics section is displayed

## Metrics Computed from Actual Backtests

The following metrics are computed directly from the backtest simulation:

- **Sharpe Ratio**: From actual portfolio returns over the backtest period
- **CAGR**: Compound Annual Growth Rate from portfolio value progression
- **Max Drawdown**: Maximum decline from peak portfolio value
- **Total Return**: Overall return over the backtest period
- **Win Rate**: Percentage of positive daily returns
- **Robustness Score**: Based on win rate and Sharpe ratio
- **Portfolio Values**: Daily portfolio values throughout the backtest
- **Drawdowns**: Daily drawdown values
- **Predicted vs Actual Prices**: Model predictions vs actual next-day prices

All metrics are computed without artificial adjustments or multipliers.

## Validation Checklist

To ensure you're seeing actual model run results:

- ✅ Check console logs for "✅ ACTUAL BACKTEST results"
- ✅ Verify `portfolio_values` array has multiple data points
- ✅ Confirm dates match your historical data range
- ✅ Ensure metrics make sense relative to the backtest period length

If you see:
- ⚠️ "MOCK metrics" in console logs → Performance Metrics section will NOT display
- ❌ "Model not found" → Model needs to be trained first

## Next Steps

1. **Run a full analysis** with a ticker to train models
2. **Check console logs** to confirm models are being evaluated
3. **Verify Performance Metrics section** displays only when actual backtest data exists
4. **Review metrics** to ensure they reflect actual model performance

