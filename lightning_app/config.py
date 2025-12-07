"""Configuration constants for the Lightning AI application."""

import os

# Alpha Vantage API Configuration
# Get your free API key from: https://www.alphavantage.co/support/#api-key
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "")
ALPHA_VANTAGE_URL = "https://www.alphavantage.co/query"

# OpenAI API Configuration
# Get your API key from: https://platform.openai.com/api-keys
# Set via environment variable OPENAI_API_KEY or in .streamlit/secrets.toml
# Priority: Environment variable > Streamlit secrets > Empty (optional feature)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# FRED API Configuration (for macroeconomic data)
# Get your API key from: https://fred.stlouisfed.org/docs/api/api_key.html
FRED_API_KEY = os.getenv("FRED_API_KEY", "")  # Optional - leave empty if not using FRED

# Financial Modeling Prep API Configuration (for earnings call transcripts)
# Sign up at https://site.financialmodelingprep.com/developer/docs/ for free tier (250 requests/day)
# Set via environment variable FMP_API_KEY
FMP_API_KEY = os.getenv("FMP_API_KEY", "")

# Default Settings
DEFAULT_TICKER = "NVDA"
DEFAULT_DATA_YEARS = 5
DEFAULT_SENTIMENT_ITEMS = 10

# Fallback Metrics Configuration
# Set to True to enable realistic fallback metrics when models are not trained
# Fallback uses historical ticker data and market benchmarks for realistic estimates
ENABLE_REALISTIC_FALLBACK = os.getenv("ENABLE_REALISTIC_FALLBACK", "True").lower() == "true"
