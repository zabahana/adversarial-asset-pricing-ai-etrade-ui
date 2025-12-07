from __future__ import annotations

from pathlib import Path
from typing import Final

import pandas as pd
import requests
from . import LightningWork, HAS_LIGHTNING

from ..config import ALPHA_VANTAGE_API_KEY, ALPHA_VANTAGE_URL

ALPHA_VANTAGE_URL: Final[str] = ALPHA_VANTAGE_URL


class DataFetchWork(LightningWork):
    """Downloads price history for a ticker using the Alpha Vantage API."""

    def __init__(self, cache_dir: str) -> None:
        if HAS_LIGHTNING:
            super().__init__(parallel=True)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def run(self, ticker: str, years: int = 5) -> str:
        # Validate ticker symbol
        if not ticker:
            raise ValueError("Please enter a ticker symbol.")
        
        ticker = ticker.strip().upper()
        if not ticker or len(ticker) < 1 or len(ticker) > 5:
            raise ValueError(f"Invalid ticker symbol: '{ticker}'. Ticker must be 1-5 characters.")
        
        # Validate API key
        api_key = ALPHA_VANTAGE_API_KEY
        if not api_key or api_key == "":
            raise ValueError("Alpha Vantage API key is not set. Please set ALPHA_VANTAGE_API_KEY environment variable.")
        
        params = {
            "function": "TIME_SERIES_DAILY_ADJUSTED",
            "symbol": ticker,
            "outputsize": "full",
            "apikey": api_key,
        }
        
        # Retry logic with exponential backoff for timeout issues
        import time
        max_retries = 3
        timeout_seconds = 60  # Increased timeout to 60 seconds
        
        for attempt in range(max_retries):
            try:
                response = requests.get(ALPHA_VANTAGE_URL, params=params, timeout=timeout_seconds)
                response.raise_for_status()
                break  # Success, exit retry loop
            except requests.exceptions.Timeout as e:
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * 5  # Exponential backoff: 5s, 10s, 20s
                    print(f"[WARNING] Alpha Vantage timeout (attempt {attempt + 1}/{max_retries}). Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"[ERROR] Alpha Vantage timeout after {max_retries} attempts")
                    raise
            except requests.exceptions.RequestException as e:
                print(f"[ERROR] Alpha Vantage request error: {e}")
                raise

        payload = response.json()
        
        # Check for Alpha Vantage error messages first - handle multiple error formats
        error_message = None
        
        # Check different error message formats
        if "Error Message" in payload:
            error_message = payload["Error Message"]
        elif "error" in payload and isinstance(payload["error"], str):
            error_message = payload["error"]
        elif "errors" in payload:
            error_message = str(payload["errors"])
        
        if error_message:
            # Provide more specific error messages based on error content
            error_lower = error_message.lower()
            
            if "invalid api call" in error_lower:
                # Invalid API call could mean: invalid symbol, invalid function, or invalid API key
                if "symbol" in error_lower or ticker.lower() in error_lower:
                    raise ValueError(
                        f"❌ Invalid ticker symbol '{ticker}'. "
                        f"Please check that '{ticker}' is a valid stock symbol. "
                        f"Common issues: typos, delisted stocks, or non-US stocks."
                    )
                else:
                    raise ValueError(
                        f"❌ Alpha Vantage API Error: {error_message}\n\n"
                        f"Possible causes:\n"
                        f"1. Invalid API key (current key: {api_key[:8]}...)\n"
                        f"2. Invalid API function call\n"
                        f"3. API rate limit exceeded\n\n"
                        f"Please check:\n"
                        f"- API key is valid at https://www.alphavantage.co/support/#api-key\n"
                        f"- You haven't exceeded rate limits (free tier: 5 calls/min, 500/day)\n"
                        f"- The ticker symbol '{ticker}' is correct"
                    )
            elif "api key" in error_lower or "invalid key" in error_lower:
                raise ValueError(
                    f"❌ Invalid Alpha Vantage API key.\n\n"
                    f"Your current API key ({api_key[:8]}...) appears to be invalid or has expired.\n\n"
                    f"To fix:\n"
                    f"1. Get a free API key at: https://www.alphavantage.co/support/#api-key\n"
                    f"2. Set it as environment variable: export ALPHA_VANTAGE_API_KEY='your_key'\n"
                    f"3. Or update it in lightning_app/config.py"
                )
            elif "rate limit" in error_lower or "call frequency" in error_lower:
                raise ValueError(
                    f"⏱️ Alpha Vantage rate limit exceeded.\n\n"
                    f"Free tier limits: 5 API calls per minute, 500 per day.\n\n"
                    f"Please wait a few minutes and try again, or upgrade to a premium plan."
                )
            else:
                # Generic error message
                raise ValueError(f"Alpha Vantage API Error: {error_message}")
        
        if "Note" in payload:
            # Rate limit or API call frequency notice
            note = payload["Note"]
            raise ValueError(f"⏱️ Alpha Vantage API Notice: {note}\n\nPlease wait a moment and try again.")
        
        # Check for the expected data key (can be "Time Series (Daily)" or variations)
        time_series_key = None
        for key in payload.keys():
            if "Time Series" in key and "Daily" in key:
                time_series_key = key
                break
        
        if not time_series_key:
            # If no time series data found, raise error with helpful message
            available_keys = list(payload.keys())
            raise ValueError(
                f"Unexpected response from Alpha Vantage for {ticker}. "
                f"Expected 'Time Series (Daily)' but found keys: {available_keys}. "
                f"Response: {payload}"
            )

        data = payload[time_series_key]
        df = pd.DataFrame.from_dict(data, orient="index").sort_index()
        df.index = pd.to_datetime(df.index)
        # Filter to last N years (replace deprecated .last() method)
        cutoff_date = pd.Timestamp.now() - pd.DateOffset(years=years)
        df = df.loc[df.index >= cutoff_date]

        cache_path = self.cache_dir / f"{ticker.lower()}_prices.parquet"
        df.to_parquet(cache_path)
        return str(cache_path)
