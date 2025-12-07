# Earnings Call Transcript APIs

## Current Status

### Alpha Vantage (Currently Used)
- ✅ **What it provides**: Earnings numbers only (EPS, dates, surprises)
- ❌ **What it doesn't provide**: Earnings call transcripts
- ✅ **Status**: Working - provides basic earnings metrics
- **Cost**: Free (with rate limits)

### Financial Modeling Prep (FMP) - **RECOMMENDED**
- ✅ **What it provides**: Full earnings call transcripts
- ✅ **Coverage**: 8,200+ companies, 10+ years of data
- ✅ **Free Tier**: 250 requests/day
- **Cost**: Free tier available, paid plans start at $14/month
- **Sign up**: https://site.financialmodelingprep.com/developer/docs/
- **API Endpoint**: `https://financialmodelingprep.com/api/v3/earning_call_transcript`

### API Ninjas - **FREE OPTION**
- ✅ **What it provides**: Full earnings call transcripts
- ✅ **Coverage**: 8,000+ companies (US large/mid/small-cap)
- ✅ **Free Tier**: 50,000 requests/month
- **Cost**: Free tier available
- **Sign up**: https://api-ninjas.com/api/earningscalltranscript
- **API Endpoint**: `https://api.api-ninjas.com/v1/earningscalltranscript`

### Polygon.io
- ✅ **What it provides**: Earnings call transcripts and data
- ✅ **Coverage**: Comprehensive, real-time
- ❌ **Cost**: Paid only (starts at $29/month)
- **Website**: https://polygon.io/

### Other Options
- **EarningsAPI.com**: Paid service, ready-to-publish summaries
- **Borsa Earnings Calls**: Paid service
- **Quartr API**: Paid service, 13,000+ companies
- **Yahoo Finance**: Free but requires web scraping (not recommended)

## Implementation

### Current Implementation
The app now supports:
1. **Alpha Vantage** (always used) - for earnings numbers
2. **Financial Modeling Prep** (optional) - for transcripts

### To Enable FMP:
1. Sign up at https://site.financialmodelingprep.com/developer/docs/
2. Get your free API key
3. Add to `lightning_app/config.py`:
   ```python
   FMP_API_KEY = "your_fmp_api_key_here"
   ```

### To Use API Ninjas (Free Alternative):
1. Sign up at https://api-ninjas.com/api/earningscalltranscript
2. Get your free API key
3. We can add support for this in the code if you prefer

## What You Get

### With Alpha Vantage Only (Current Default):
- ✅ Earnings numbers (EPS, surprise, dates)
- ❌ No transcript analysis

### With FMP API Key Added:
- ✅ Earnings numbers (from Alpha Vantage)
- ✅ Full earnings call transcript (from FMP)
- ✅ AI analysis of transcript + numbers (via OpenAI)

## Recommendation

For a **free solution**, use **API Ninjas** - it offers 50,000 requests/month for free and provides full transcripts.

For a **premium solution**, **FMP** is the most popular and well-documented option.



