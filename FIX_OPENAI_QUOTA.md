# Fix OpenAI Quota Exceeded Error

## Problem
You're seeing this error:
```
Error code: 429 - {'error': {'message': 'You exceeded your current quota, please check your plan and billing details.'}}
```

## Quick Fix Steps

### Option 1: Check and Add Credits (Recommended)

1. **Visit OpenAI Usage Dashboard**
   - Go to: https://platform.openai.com/usage
   - Log in with your OpenAI account
   - Check your current usage and remaining credits

2. **Add Payment Method / Credits**
   - Go to: https://platform.openai.com/account/billing
   - Add a payment method if you haven't already
   - Set up usage limits or add credits to your account

3. **Check Your Plan**
   - Free tier has limited credits
   - Consider upgrading to a paid plan for production use
   - Paid plans start at $5 minimum credit

### Option 2: Get a New API Key

If your current key is expired or invalid:

1. **Generate New API Key**
   - Visit: https://platform.openai.com/api-keys
   - Click "Create new secret key"
   - Copy the key (you'll only see it once!)

2. **Update Configuration**
   - Open: `lightning_app/config.py`
   - Replace the `OPENAI_API_KEY` value:
     ```python
     OPENAI_API_KEY = "sk-proj-your-new-key-here"
     ```

3. **Restart Application**
   - Restart your Streamlit app or Lightning app
   - The new key will be used automatically

### Option 3: Use Environment Variable (More Secure)

Instead of hardcoding the key:

1. **Set Environment Variable**
   ```bash
   export OPENAI_API_KEY="sk-proj-your-key-here"
   ```

2. **Update config.py to read from environment**
   ```python
   import os
   OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
   ```

3. **Restart Application**

## Diagnostic Tool

Run the diagnostic script to check your API status:

```bash
cd /Users/zelalemabahana/adversarial-asset-pricing-ai-etrade-ui
python check_openai_quota.py
```

This will:
- ✅ Check if API key is configured
- ✅ Test API connection
- ✅ Diagnose quota/billing issues
- ✅ Provide specific fix instructions

## Understanding OpenAI Pricing

### Free Tier Limits
- Limited credits (usually $5-18 worth)
- After credits are used, you need to add payment method
- No monthly subscription required, pay-as-you-go

### Paid Plans
- **Pay-as-you-go**: Add credits as needed
- **Monthly subscription**: Predictable costs
- Check pricing: https://openai.com/pricing

### GPT-3.5-turbo Costs
- **Input**: $0.50 per 1M tokens
- **Output**: $1.50 per 1M tokens
- Typical sentiment analysis: ~500 tokens per call
- Cost per analysis: ~$0.001 (very cheap!)

## Current API Key Location

Your API key is currently stored in:
- **File**: `lightning_app/config.py`
- **Variable**: `OPENAI_API_KEY`
- **Line**: 8

**⚠️ Security Note**: Consider moving this to environment variables for production!

## Temporary Workaround

While fixing the quota issue, the application will:
- ✅ Continue working with Alpha Vantage sentiment only
- ✅ Fall back gracefully (no crashes)
- ✅ Still provide sentiment analysis (slightly less detailed)

You can use the app normally, but OpenAI-enhanced sentiment won't be available until the quota is fixed.

## Still Having Issues?

1. **Check API Key Format**
   - Should start with: `sk-proj-` or `sk-`
   - Should be ~50 characters long

2. **Verify Account Status**
   - Make sure your OpenAI account is active
   - Check for any account restrictions

3. **Contact OpenAI Support**
   - Visit: https://help.openai.com/
   - They can help with billing/quota issues

## Next Steps After Fixing

Once you've fixed the quota:

1. **Test the fix**:
   ```bash
   python check_openai_quota.py
   ```

2. **Verify in application**:
   - Run your Streamlit app
   - Check logs for "Combined sentiment: Alpha Vantage + OpenAI"
   - Should see both scores instead of just Alpha Vantage

3. **Monitor Usage**:
   - Check https://platform.openai.com/usage regularly
   - Set up usage alerts if available
   - Consider usage limits to prevent surprise bills


