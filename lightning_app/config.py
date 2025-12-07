"""Configuration constants for the Lightning AI application."""

import os


def _get_api_key(env_var_name: str, streamlit_key: str = None) -> str:
    """
    Get API key from Streamlit secrets or environment variable.
    Priority: Environment variable > Streamlit secrets > Empty string
    
    Args:
        env_var_name: Environment variable name (e.g., "ALPHA_VANTAGE_API_KEY")
        streamlit_key: Streamlit secrets key (if different from env_var_name)
    
    Returns:
        API key string or empty string if not found
    """
    # Try environment variable first
    api_key = os.getenv(env_var_name, "")
    if api_key:
        return api_key
    
    # Try Streamlit secrets (only if running in Streamlit)
    try:
        import streamlit as st
        secrets_key = streamlit_key or env_var_name
        if hasattr(st, 'secrets') and secrets_key in st.secrets:
            api_key = st.secrets[secrets_key]
            if api_key:
                return api_key
    except (ImportError, AttributeError, RuntimeError):
        # Not running in Streamlit or secrets not available
        pass
    
    return ""


# Alpha Vantage API Configuration
# Get your free API key from: https://www.alphavantage.co/support/#api-key
# Priority: Environment variable > Streamlit secrets > Empty
ALPHA_VANTAGE_API_KEY = _get_api_key("ALPHA_VANTAGE_API_KEY")
ALPHA_VANTAGE_URL = "https://www.alphavantage.co/query"

# OpenAI API Configuration
# Get your API key from: https://platform.openai.com/api-keys
# Priority: Environment variable > Streamlit secrets > Empty (optional feature)
OPENAI_API_KEY = _get_api_key("OPENAI_API_KEY")

# FRED API Configuration (for macroeconomic data)
# Get your API key from: https://fred.stlouisfed.org/docs/api/api_key.html
# Priority: Environment variable > Streamlit secrets > Empty
FRED_API_KEY = _get_api_key("FRED_API_KEY")  # Optional - leave empty if not using FRED

# Financial Modeling Prep API Configuration (for earnings call transcripts and market data)
# Sign up at https://site.financialmodelingprep.com/developer/docs/ for free tier (250 requests/day)
# Priority: Environment variable > Streamlit secrets > Empty
FMP_API_KEY = _get_api_key("FMP_API_KEY")

# Default Settings
DEFAULT_TICKER = "NVDA"
DEFAULT_DATA_YEARS = 5
DEFAULT_SENTIMENT_ITEMS = 10

# Fallback Metrics Configuration
# Set to True to enable realistic fallback metrics when models are not trained
# Fallback uses historical ticker data and market benchmarks for realistic estimates
ENABLE_REALISTIC_FALLBACK = os.getenv("ENABLE_REALISTIC_FALLBACK", "True").lower() == "true"

