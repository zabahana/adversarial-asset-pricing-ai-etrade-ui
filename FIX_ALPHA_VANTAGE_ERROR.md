# Fix: Alpha Vantage "Invalid API Call" Error

## The Error

```
Alpha Vantage API Error: Invalid API call. Please retry or visit the documentation...
```

## Common Causes & Solutions

### 1. Invalid API Key (Most Common)

**Symptoms:**
- "Invalid API call" error
- Works sometimes but fails randomly

**Solution:**
```bash
# Get a free API key
1. Visit: https://www.alphavantage.co/support/#api-key
2. Sign up for free account
3. Copy your API key

# Set it in environment (recommended)
export ALPHA_VANTAGE_API_KEY='your_new_key_here'

# Or update config file
# Edit: lightning_app/config.py
ALPHA_VANTAGE_API_KEY = "your_new_key_here"

# Restart Streamlit
```

### 2. Rate Limit Exceeded

**Symptoms:**
- Works initially, then starts failing
- Error mentions "call frequency" or "rate limit"

**Solution:**
- Free tier: 5 calls/minute, 500 calls/day
- Wait 1-2 minutes between requests
- Consider upgrading to premium plan

### 3. Invalid Ticker Symbol

**Symptoms:**
- Error only for specific tickers
- Works for some stocks but not others

**Solution:**
- Verify ticker is correct (e.g., NVDA not NVDAX)
- Check stock is listed (not delisted)
- Ensure it's a US stock symbol

## Quick Diagnostic Test

Run this to test your API key:

```bash
python test_alpha_vantage_diagnostic.py
```

This will:
- ✅ Check if API key is set
- ✅ Test API connectivity
- ✅ Identify the specific issue
- ✅ Provide fix recommendations

## Immediate Fix Steps

1. **Test your API key:**
   ```bash
   python test_alpha_vantage_diagnostic.py
   ```

2. **If API key is invalid:**
   - Get new key: https://www.alphavantage.co/support/#api-key
   - Set environment variable: `export ALPHA_VANTAGE_API_KEY='new_key'`
   - Restart Streamlit

3. **If rate limited:**
   - Wait 1-2 minutes
   - Try again
   - Reduce number of API calls

4. **If ticker is invalid:**
   - Double-check the symbol
   - Try a different ticker (AAPL, MSFT, GOOGL)
   - Ensure it's a valid US stock

## Updated Error Messages

The app now provides specific error messages:
- ❌ Invalid ticker → Shows which ticker and how to fix
- ❌ Invalid API key → Shows how to get/set new key
- ⏱️ Rate limit → Shows limits and wait time

## Still Having Issues?

1. Check the diagnostic script output
2. Verify API key at Alpha Vantage website
3. Check console logs for detailed error messages
4. Try a different ticker symbol
5. Wait a few minutes if rate limited

