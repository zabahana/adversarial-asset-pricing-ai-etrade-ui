# Earnings Call Data Integration Guide

## Current Status

**Earnings call data is currently NOT integrated into historical features.**

### Where Earnings Data is Fetched:
- **File**: `lightning_app/works/fundamental_analysis_work.py`
- **Method**: `_get_earnings_call_transcript(ticker)`
- **API Sources**:
  - Alpha Vantage: Earnings numbers (EPS, surprise, dates)
  - Financial Modeling Prep (FMP): Full transcripts (if API key provided)
- **Usage**: Only displayed in Streamlit UI, **NOT added to features**

### Where It Should Be Integrated:
- **File**: `lightning_app/works/feature_engineering_work.py`
- **Method**: `run()` - This is where features are created from price data
- **Integration Point**: After price features are calculated, before saving

---

## How to Integrate Earnings Data into Historical Features

### Option 1: Add Earnings Features in FeatureEngineeringWork

**Location**: `lightning_app/works/feature_engineering_work.py`

Add earnings data fetching and merge it with price data:

```python
def run(self, price_path: str, ticker: str, macro_market_data: Optional[Dict[str, str]] = None) -> str:
    # ... existing code ...
    
    # Add earnings data integration
    df = self._add_earnings_features(df, ticker)
    
    # ... rest of existing code ...

def _add_earnings_features(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Add historical earnings data as features."""
    from ..config import ALPHA_VANTAGE_API_KEY, ALPHA_VANTAGE_URL
    import requests
    
    try:
        # Fetch historical earnings data from Alpha Vantage
        params = {
            "function": "EARNINGS",
            "symbol": ticker.upper(),
            "apikey": ALPHA_VANTAGE_API_KEY,
        }
        response = requests.get(ALPHA_VANTAGE_URL, params=params, timeout=15)
        response.raise_for_status()
        payload = response.json()
        
        if "quarterlyEarnings" not in payload:
            return df
        
        # Convert to DataFrame
        earnings_list = []
        for earning in payload["quarterlyEarnings"]:
            earnings_list.append({
                "date": pd.to_datetime(earning.get("fiscalDateEnding", "")),
                "reportedEPS": float(earning.get("reportedEPS", 0)) if earning.get("reportedEPS") else None,
                "estimatedEPS": float(earning.get("estimatedEPS", 0)) if earning.get("estimatedEPS") else None,
                "surprise": float(earning.get("surprise", 0)) if earning.get("surprise") else None,
                "surprisePercentage": float(earning.get("surprisePercentage", 0)) if earning.get("surprisePercentage") else None,
            })
        
        if not earnings_list:
            return df
        
        earnings_df = pd.DataFrame(earnings_list)
        earnings_df.set_index("date", inplace=True)
        earnings_df.sort_index(inplace=True)
        
        # Forward fill earnings data to daily frequency
        earnings_df = earnings_df.reindex(df.index, method='ffill')
        
        # Add earnings features
        df["earnings_reported_eps"] = earnings_df["reportedEPS"]
        df["earnings_estimated_eps"] = earnings_df["estimatedEPS"]
        df["earnings_surprise"] = earnings_df["surprise"]
        df["earnings_surprise_pct"] = earnings_df["surprisePercentage"]
        
        # Calculate earnings momentum features
        df["earnings_eps_growth"] = earnings_df["reportedEPS"].pct_change()
        df["earnings_surprise_momentum"] = earnings_df["surprisePercentage"].rolling(4).mean()
        
        # Days since last earnings (useful for predicting next earnings)
        last_earnings_dates = earnings_df.dropna().index
        if len(last_earnings_dates) > 0:
            df["days_since_earnings"] = (df.index - last_earnings_dates[-1]).days
            
            # Feature: approaching earnings date (0-90 days window)
            df["approaching_earnings"] = (df["days_since_earnings"] >= -90) & (df["days_since_earnings"] <= 0)
            df["approaching_earnings"] = df["approaching_earnings"].astype(int)
        
        print(f"[INFO] Added earnings features: {len(earnings_df)} earnings dates")
        
    except Exception as e:
        print(f"[WARNING] Failed to add earnings features: {e}")
    
    return df
```

---

### Option 2: Create a Separate Earnings Features Work

Create `lightning_app/works/earnings_features_work.py`:

```python
class EarningsFeaturesWork(LightningWork):
    """Fetches historical earnings data and creates earnings-based features."""
    
    def run(self, ticker: str, years: int = 5) -> str:
        """Fetch historical earnings data and save as parquet."""
        # Fetch from Alpha Vantage
        # Process into daily features
        # Forward fill to match price data frequency
        # Save to parquet
        pass
```

Then merge in `feature_engineering_work.py` similar to how macro/market data is merged.

---

## Recommended Approach

**I recommend Option 1** - directly integrating earnings features into `FeatureEngineeringWork` because:
1. Earnings data is stock-specific (unlike macro data which is market-wide)
2. Simpler pipeline (fewer files)
3. Easier to align dates with price data

---

## Earnings Features to Add (Priority Order)

### High Priority (Direct Impact):
1. **`reported_eps`** - Latest reported EPS
2. **`earnings_surprise_pct`** - Percentage surprise vs. estimates
3. **`days_since_earnings`** - Days since last earnings announcement

### Medium Priority (Derived Features):
4. **`eps_growth`** - Quarterly EPS growth rate
5. **`surprise_momentum`** - Rolling average of surprises (4 quarters)
6. **`approaching_earnings`** - Binary flag (within 90 days of next earnings)

### Low Priority (Advanced):
7. **`earnings_volatility`** - Volatility of earnings surprises
8. **`beat_streak`** - Consecutive quarters beating estimates

---

## Feature Group Assignment

Earnings features should be added to **Group 7: Sentiment** in the multi-head attention structure:
- They provide fundamental context similar to sentiment
- They capture market expectations and surprises
- They align with news/sentiment timing

---

## Next Steps

1. **Add `_add_earnings_features()` method** to `FeatureEngineeringWork`
2. **Call it in `run()` method** after price features are calculated
3. **Test with a ticker** that has earnings data
4. **Verify features are saved** in the parquet file
5. **Check feature counts** in the model input dimension

---

## Example Integration in FeatureEngineeringWork

See the enhanced `feature_engineering_work.py` file - I can add the earnings integration if you'd like.

