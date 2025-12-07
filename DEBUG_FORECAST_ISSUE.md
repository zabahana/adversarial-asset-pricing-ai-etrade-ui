# Debugging Model Forecast Issue

## Problem
Model forecast is not appearing in the UI even after training completes.

## Root Cause Analysis

### Expected Flow:
1. ✅ Model training completes → saves checkpoint
2. ✅ Model inference runs → generates backtest metrics
3. ❓ Forecast generation → should generate next-day forecast
4. ❓ Results saved → should include forecast data with `available: True`
5. ❓ UI reads results → should display forecast

### Current Code Flow:

#### In `model_inference_work.py`:
- Line 66-113: Forecast generation is called AFTER backtesting
- Forecast generation calls `forecast_all_models()`
- Results should include `mha_dqn_robust.available = True`

#### In `streamlit_app.py`:
- Line 795-807: Reads `model_results.json` and checks `mha_dqn_robust.available`
- Line 798: `if model_results.get("mha_dqn_robust", {}).get("available", False)`

## Potential Issues:

1. **Model Loading Failure**: Model might fail to load during forecast
   - Check: Does model checkpoint exist after training?
   - Check: Is input_dim correct when loading?

2. **Forecast Generation Error**: Silent exception caught
   - Check: Look for `[WARNING] Error generating forecast` in logs
   - Check: Full traceback should be printed

3. **Data Structure Mismatch**: Forecast data not in expected format
   - Expected: `results["mha_dqn_robust"]["available"] = True`
   - Expected: `results["last_data_date"]` and `results["forecast_date"]` at top level

4. **Timing Issue**: Forecast generation happens but results not saved
   - Check: Is results file written AFTER forecast generation?

## Debugging Steps:

1. **Check if forecast generation is called**:
   ```python
   # Look for: "[FORECAST] Generating next-day forecasts..."
   ```

2. **Check if forecast succeeds**:
   ```python
   # Look for: "[FORECAST] ✅ Forecast generated: ..."
   # Or: "[WARNING] Forecast generation failed..."
   ```

3. **Check results JSON file**:
   ```bash
   cat results/nvda_model_results.json | jq '.mha_dqn_robust.available'
   cat results/nvda_model_results.json | jq '.last_data_date'
   cat results/nvda_model_results.json | jq '.forecast_date'
   ```

4. **Check model checkpoint exists**:
   ```bash
   ls -la models/mha_dqn/adversarial.ckpt
   ```

## Expected Results JSON Structure:

```json
{
  "mha_dqn_robust": {
    "available": true,
    "recommendation": "BUY",
    "price_change_pct": 1.23,
    "confidence": 0.75,
    "forecasted_price": 150.50,
    "last_actual_price": 148.50,
    "forecast_date": "2024-01-15",
    "q_values": [...],
    "explainability": {...},
    "metrics": {...},
    "plots": {...}
  },
  "last_data_date": "2024-01-14",
  "forecast_date": "2024-01-15",
  "current_price": 148.50
}
```

## Next Steps:

1. Add more detailed logging to forecast generation
2. Verify model checkpoint exists and is loadable
3. Check if errors are being silently caught
4. Verify results JSON structure matches expectations
