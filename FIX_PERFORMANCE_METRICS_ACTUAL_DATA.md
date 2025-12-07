# Fix Performance Metrics to Use Only Actual Model Data

## Issues Found

1. **Predicted Prices are Synthetic**: Lines 422-442 use approximations from Q-values, not actual model predictions
2. **Default Fallback Values**: Line 438 uses `avg_return = 0.02` (2% default) which is synthetic
3. **Mock Metrics Fallback**: Lines 511-513 and 521-523 fall back to mock metrics
4. **Price Approximations**: Lines 415-420 approximate prices when close price isn't available

## Required Changes

1. Remove all synthetic price predictions - use actual model predictions only
2. Remove default fallback values - fail if data is insufficient
3. Ensure mock metrics are NEVER used for performance display
4. Only use actual historical price data from features
5. Add validation to ensure all data comes from actual model runs

