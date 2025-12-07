
# Daily News Sentiment API - Current Setup

## ✅ **Alpha Vantage NEWS_SENTIMENT API** (Currently Used)

### What It Provides:
- **Daily news articles** with sentiment scores
- **Overall sentiment score** per article (-1 to +1)
- **Ticker-specific sentiment scores** for multiple stocks
- **Time-based filtering** (can filter by date range)
- Up to **1000 articles per request**

### Current Implementation:
- **Location**: `lightning_app/works/sentiment_work.py`
- **Function**: `NEWS_SENTIMENT`
- **Current limit**: 50 articles (can be increased)
- **Sort order**: LATEST (most recent first)

### API Endpoint:
```
https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers=NVDA&sort=LATEST&limit=50&apikey=YOUR_KEY
```

### Response Format:
Each article includes:
- `time_published`: Article timestamp
- `title`: Article title  
- `summary`: Article summary
- `overall_sentiment_score`: Numeric score from -1 (bearish) to +1 (bullish)
- `overall_sentiment_label`: "Bearish", "Bullish", or "Neutral"
- `ticker_sentiment`: Array of ticker-specific sentiment scores

### Daily Usage:
The API returns news articles published daily. You can:
- Filter by date range using `time_from` and `time_to` parameters
- Sort by LATEST to get most recent news
- Get up to 1000 articles per request

## Alternative APIs (Not Currently Used):

1. **Financial Modeling Prep (FMP)** - Does NOT provide daily news sentiment
   - FMP provides: earnings transcripts, financial statements, company profiles
   - Does NOT provide: news sentiment analysis

2. **Other Options** (if needed):
   - Advanced Logic Analytics Sentiment API
   - Financial News API
   - PulseBit Sentiment API

## ✅ Your Current Setup:
- ✅ **Alpha Vantage** - Provides daily news sentiment
- ✅ Already integrated in your app
- ✅ Working with NEWS_SENTIMENT function
- ✅ Gets latest 50 articles sorted by date

## Recommendation:
**Stick with Alpha Vantage** - it's already working and provides exactly what you need for daily news sentiment!

