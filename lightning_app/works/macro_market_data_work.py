"""
Macro and Market Data Work - Fetches historical macroeconomic, commodity, forex, 
and market index data from Alpha Vantage for use as RL agent features.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import time

import pandas as pd
import requests
from . import LightningWork, HAS_LIGHTNING


def _make_request_with_retry(url: str, params: dict, max_retries: int = 3, timeout: int = 60) -> requests.Response:
    """
    Make HTTP request with retry logic and exponential backoff.
    
    Args:
        url: Request URL
        params: Request parameters
        max_retries: Maximum number of retry attempts (default: 3)
        timeout: Request timeout in seconds (default: 60)
        
    Returns:
        Response object
        
    Raises:
        requests.exceptions.RequestException: If all retries fail
    """
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            return response  # Success
        except requests.exceptions.Timeout as e:
            if attempt < max_retries - 1:
                wait_time = (2 ** attempt) * 5  # Exponential backoff: 5s, 10s, 20s
                print(f"[WARNING] Alpha Vantage timeout (attempt {attempt + 1}/{max_retries}). Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"[ERROR] Alpha Vantage timeout after {max_retries} attempts")
                raise
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1 and hasattr(e, 'response') and e.response is not None:
                # Retry on transient errors (5xx)
                if 500 <= e.response.status_code < 600:
                    wait_time = (2 ** attempt) * 5
                    print(f"[WARNING] Alpha Vantage server error {e.response.status_code} (attempt {attempt + 1}/{max_retries}). Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
            # Don't retry on 4xx errors or other non-transient errors
            print(f"[ERROR] Alpha Vantage request error: {e}")
            raise


class MacroMarketDataWork(LightningWork):
    """
    Fetches historical macroeconomic, commodity, forex, and market data from Alpha Vantage.
    Organizes data into feature groups for multi-head attention.
    """

    def __init__(self, cache_dir: str) -> None:
        if HAS_LIGHTNING:
            super().__init__(parallel=True)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.base_url = "https://www.alphavantage.co/query"
        
        # Feature groups for multi-head attention (8 groups to align with 8 heads)
        self.feature_groups = {
            # Group 1: Price & Returns (handled in feature_engineering_work.py)
            # Group 2: Macroeconomic Indicators
            "macro": {
                "REAL_GDP": {"interval": "quarterly", "priority": 1},
                "CPI": {"interval": "monthly", "priority": 1},
                "UNEMPLOYMENT": {"interval": "monthly", "priority": 1},
                "FEDERAL_FUNDS_RATE": {"interval": "monthly", "priority": 1},
                # T10Y2Y removed - not available in Alpha Vantage API
                # Use T10Y2Y spread via FRED API directly if needed
            },
            # Group 3: Commodities
            "commodities": {
                "WTI": {"interval": "daily", "priority": 1},  # Crude oil
                "BRENT": {"interval": "daily", "priority": 2},  # Alternative oil price
                "NATURAL_GAS": {"interval": "daily", "priority": 2},
                "COPPER": {"interval": "daily", "priority": 1},  # Economic activity indicator
                "GOLD": {"interval": "daily", "priority": 1},  # Safe haven / inflation hedge
            },
            # Group 4: Market Indices & Sector ETFs
            "market_indices": {
                "SPY": {"interval": "daily", "priority": 1},  # S&P 500
                "VIX": {"interval": "daily", "priority": 1},  # Volatility index
                "XLK": {"interval": "daily", "priority": 1},  # Technology sector
                "XLF": {"interval": "daily", "priority": 1},  # Financials sector
                "XLE": {"interval": "daily", "priority": 1},  # Energy sector
            },
            # Group 5: Forex (Currency Pairs)
            "forex": {
                "EURUSD": {"interval": "daily", "priority": 1},  # Euro/USD
                "GBPUSD": {"interval": "daily", "priority": 1},  # British Pound/USD
                "USDJPY": {"interval": "daily", "priority": 1},  # USD/Japanese Yen
                "AUDUSD": {"interval": "daily", "priority": 2},  # Australian Dollar/USD
            },
            # Group 6: Cryptocurrency (Risk-on indicator)
            "crypto": {
                "BTC": {"interval": "daily", "priority": 1},  # Bitcoin
                "ETH": {"interval": "daily", "priority": 2},  # Ethereum
            },
        }

    def run(self, years: int = 5, api_key: Optional[str] = None) -> Dict[str, str]:
        """
        Fetch historical macro and market data.
        
        Args:
            years: Number of years of historical data to fetch
            api_key: Alpha Vantage API key (if None, reads from config)
        
        Returns:
            Dictionary mapping feature group names to parquet file paths
        """
        from ..config import ALPHA_VANTAGE_API_KEY
        
        api_key = api_key or ALPHA_VANTAGE_API_KEY
        
        if not api_key:
            print("[WARNING] Alpha Vantage API key not found. Returning empty macro data.")
            return self._get_empty_results()
        
        print(f"[1] Fetching macro/market data for {years} years...")
        
        results = {}
        
        # Fetch each feature group
        for group_name, indicators in self.feature_groups.items():
            print(f"[2] Processing {group_name} group...")
            group_data = self._fetch_group_data(group_name, indicators, years, api_key)
            
            if not group_data.empty:
                file_path = self.cache_dir / f"macro_market_{group_name}.parquet"
                group_data.to_parquet(file_path)
                results[group_name] = str(file_path)
                print(f"[SUCCESS] Saved {group_name}: {len(group_data)} rows, {len(group_data.columns)} columns")
            else:
                print(f"[WARNING] No data fetched for {group_name}")
                results[group_name] = ""
        
        return results

    def _fetch_group_data(
        self, 
        group_name: str, 
        indicators: Dict[str, Dict], 
        years: int, 
        api_key: str
    ) -> pd.DataFrame:
        """Fetch data for a specific feature group."""
        all_dataframes = []
        
        # Sort by priority
        sorted_indicators = sorted(indicators.items(), key=lambda x: x[1]["priority"])
        
        for indicator_name, config in sorted_indicators:
            try:
                if group_name == "macro":
                    df = self._fetch_economic_indicator(indicator_name, config["interval"], years, api_key)
                elif group_name == "commodities":
                    df = self._fetch_commodity(indicator_name, config["interval"], years, api_key)
                elif group_name == "market_indices":
                    df = self._fetch_market_index(indicator_name, years, api_key)
                elif group_name == "forex":
                    df = self._fetch_forex(indicator_name, years, api_key)
                elif group_name == "crypto":
                    df = self._fetch_crypto(indicator_name, years, api_key)
                else:
                    continue
                
                if not df.empty:
                    # Prefix column names with group and indicator name
                    df.columns = [f"{group_name}_{indicator_name}_{col}" for col in df.columns]
                    all_dataframes.append(df)
                    print(f"[INFO] Fetched {indicator_name}: {len(df)} rows")
                
                # Rate limiting: 5 calls per minute for free tier, 75 for premium
                import time
                time.sleep(12)  # Conservative delay (5 calls/min = 12 sec between calls)
                
            except Exception as e:
                print(f"[WARNING] Failed to fetch {indicator_name}: {e}")
                continue
        
        if not all_dataframes:
            return pd.DataFrame()
        
        # Merge all dataframes on date index
        result = all_dataframes[0]
        for df in all_dataframes[1:]:
            result = result.join(df, how="outer")
        
        # Sort by date and fill missing values (forward fill for macro data)
        result = result.sort_index()
        result = result.ffill().bfill()  # Forward fill then backward fill
        
        # Filter to requested years
        cutoff_date = pd.Timestamp.now() - pd.DateOffset(years=years)
        result = result.loc[result.index >= cutoff_date]
        
        return result

    def _fetch_economic_indicator(self, indicator: str, interval: str, years: int, api_key: str) -> pd.DataFrame:
        """Fetch economic indicator from Alpha Vantage (FRED data)."""
        params = {
            "function": indicator,
            "interval": interval,
            "apikey": api_key,
        }
        
        try:
            response = _make_request_with_retry(self.base_url, params, max_retries=3, timeout=60)
            data = response.json()
            
            # Handle Alpha Vantage response format
            if "Error Message" in data:
                error_msg = data['Error Message']
                print(f"[WARNING] Alpha Vantage error for {indicator}: {error_msg}")
                return pd.DataFrame()
            
            if "Note" in data:
                note = data['Note']
                print(f"[WARNING] Rate limit note for {indicator}: {note}")
                return pd.DataFrame()
            
            # Check if entire response is an error string (API sometimes returns string errors)
            if isinstance(data, str):
                print(f"[WARNING] Alpha Vantage returned string error for {indicator}: {data}")
                return pd.DataFrame()
            
            # Extract time series data
            time_series_key = None
            for key in data.keys():
                if key not in ["Meta Data", "Information"]:
                    time_series_key = key
                    break
            
            if not time_series_key or time_series_key not in data:
                # Check if data contains error information as string values
                for key, value in data.items():
                    if isinstance(value, str) and ("error" in value.lower() or "invalid" in value.lower() or "does not exist" in value.lower()):
                        print(f"[WARNING] Alpha Vantage error for {indicator}: {value}")
                        return pd.DataFrame()
                return pd.DataFrame()
            
            # Parse data
            observations = data[time_series_key]
            if not observations:
                return pd.DataFrame()
            
            # Check if observations is a dictionary (API may return error string)
            if not isinstance(observations, dict):
                # If observations is a string, it's likely an error message or indicator name
                if isinstance(observations, str):
                    # Check if it's an error message or just the indicator description
                    error_keywords = ["error", "invalid", "does not exist", "not found", "unavailable"]
                    if any(keyword in observations.lower() for keyword in error_keywords):
                        print(f"[WARNING] Alpha Vantage error for {indicator}: {observations}")
                    else:
                        # If it's just the indicator name/description, the API call likely failed
                        print(f"[WARNING] Alpha Vantage returned indicator name instead of data for {indicator}: {observations}")
                else:
                    print(f"[WARNING] Alpha Vantage unexpected response type for {indicator}: Expected dict, got {type(observations).__name__}: {observations}")
                return pd.DataFrame()
            
            # Additional check: if observations dict is empty or has wrong structure
            if len(observations) == 0:
                print(f"[WARNING] Alpha Vantage returned empty data for {indicator}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df_data = []
            for date_str, value in observations.items():
                try:
                    date = pd.to_datetime(date_str)
                    value_float = float(value) if value != "." else None
                    if value_float is not None:
                        df_data.append({"date": date, indicator: value_float})
                except:
                    continue
            
            if not df_data:
                return pd.DataFrame()
            
            df = pd.DataFrame(df_data)
            df.set_index("date", inplace=True)
            df.sort_index(inplace=True)
            
            return df
            
        except Exception as e:
            print(f"[ERROR] Error fetching {indicator}: {e}")
            return pd.DataFrame()

    def _fetch_commodity(self, symbol: str, interval: str, years: int, api_key: str) -> pd.DataFrame:
        """Fetch commodity price data."""
        params = {
            "function": "WTI" if symbol == "WTI" else "BRENT" if symbol == "BRENT" else symbol.upper(),
            "interval": interval,
            "apikey": api_key,
        }
        
        try:
            response = _make_request_with_retry(self.base_url, params, max_retries=3, timeout=60)
            data = response.json()
            
            if "Error Message" in data or "Note" in data:
                return pd.DataFrame()
            
            # Parse commodity data (similar structure to economic indicators)
            time_series_key = None
            for key in data.keys():
                if key not in ["Meta Data", "Information"]:
                    time_series_key = key
                    break
            
            if not time_series_key:
                return pd.DataFrame()
            
            observations = data[time_series_key]
            if not isinstance(observations, dict):
                print(f"[ERROR] Error fetching {symbol}: Expected dict, got {type(observations).__name__}: {observations}")
                return pd.DataFrame()
            
            df_data = []
            for date_str, value in observations.items():
                try:
                    date = pd.to_datetime(date_str)
                    value_float = float(value) if value != "." else None
                    if value_float is not None:
                        df_data.append({"date": date, "value": value_float})
                except:
                    continue
            
            if not df_data:
                return pd.DataFrame()
            
            df = pd.DataFrame(df_data)
            df.set_index("date", inplace=True)
            df.rename(columns={"value": symbol}, inplace=True)
            df.sort_index(inplace=True)
            
            return df
            
        except Exception as e:
            return pd.DataFrame()

    def _fetch_market_index(self, symbol: str, years: int, api_key: str) -> pd.DataFrame:
        """Fetch market index/ETF data using TIME_SERIES_DAILY."""
        params = {
            "function": "TIME_SERIES_DAILY_ADJUSTED",
            "symbol": symbol,
            "outputsize": "full",
            "apikey": api_key,
        }
        
        try:
            response = _make_request_with_retry(self.base_url, params, max_retries=3, timeout=60)
            data = response.json()
            
            if "Error Message" in data or "Note" in data:
                return pd.DataFrame()
            
            # Parse time series
            time_series_key = "Time Series (Daily)"
            if time_series_key not in data:
                return pd.DataFrame()
            
            observations = data[time_series_key]
            if not isinstance(observations, dict):
                print(f"[ERROR] Error fetching {symbol}: Expected dict, got {type(observations).__name__}: {observations}")
                return pd.DataFrame()
            
            df_data = []
            for date_str, values in observations.items():
                try:
                    date = pd.to_datetime(date_str)
                    close_price = float(values.get("4. close", 0))
                    if close_price > 0:
                        df_data.append({"date": date, "close": close_price})
                except:
                    continue
            
            if not df_data:
                return pd.DataFrame()
            
            df = pd.DataFrame(df_data)
            df.set_index("date", inplace=True)
            df.sort_index(inplace=True)
            df.rename(columns={"close": symbol}, inplace=True)
            
            return df
            
        except Exception as e:
            return pd.DataFrame()

    def _fetch_forex(self, pair: str, years: int, api_key: str) -> pd.DataFrame:
        """Fetch forex pair data."""
        # Parse pair (e.g., "EURUSD" -> "EUR" and "USD")
        if len(pair) == 6:
            from_symbol = pair[:3]
            to_symbol = pair[3:]
        else:
            return pd.DataFrame()
        
        params = {
            "function": "FX_DAILY",
            "from_symbol": from_symbol,
            "to_symbol": to_symbol,
            "outputsize": "full",
            "apikey": api_key,
        }
        
        try:
            response = _make_request_with_retry(self.base_url, params, max_retries=3, timeout=60)
            data = response.json()
            
            if "Error Message" in data or "Note" in data:
                return pd.DataFrame()
            
            time_series_key = "Time Series FX (Daily)"
            if time_series_key not in data:
                return pd.DataFrame()
            
            observations = data[time_series_key]
            if not isinstance(observations, dict):
                print(f"[ERROR] Error fetching {pair}: Expected dict, got {type(observations).__name__}: {observations}")
                return pd.DataFrame()
            
            df_data = []
            for date_str, values in observations.items():
                try:
                    date = pd.to_datetime(date_str)
                    close_price = float(values.get("4. close", 0))
                    if close_price > 0:
                        df_data.append({"date": date, "close": close_price})
                except:
                    continue
            
            if not df_data:
                return pd.DataFrame()
            
            df = pd.DataFrame(df_data)
            df.set_index("date", inplace=True)
            df.sort_index(inplace=True)
            df.rename(columns={"close": pair}, inplace=True)
            
            return df
            
        except Exception as e:
            return pd.DataFrame()

    def _fetch_crypto(self, symbol: str, years: int, api_key: str) -> pd.DataFrame:
        """Fetch cryptocurrency data."""
        params = {
            "function": "DIGITAL_CURRENCY_DAILY",
            "symbol": symbol,
            "market": "USD",
            "apikey": api_key,
        }
        
        try:
            response = _make_request_with_retry(self.base_url, params, max_retries=3, timeout=60)
            data = response.json()
            
            if "Error Message" in data or "Note" in data:
                return pd.DataFrame()
            
            time_series_key = "Time Series (Digital Currency Daily)"
            if time_series_key not in data:
                return pd.DataFrame()
            
            observations = data[time_series_key]
            if not isinstance(observations, dict):
                print(f"[ERROR] Error fetching {symbol}: Expected dict, got {type(observations).__name__}: {observations}")
                return pd.DataFrame()
            
            df_data = []
            for date_str, values in observations.items():
                try:
                    date = pd.to_datetime(date_str)
                    close_price = float(values.get("4a. close (USD)", 0))
                    if close_price > 0:
                        df_data.append({"date": date, "close": close_price})
                except:
                    continue
            
            if not df_data:
                return pd.DataFrame()
            
            df = pd.DataFrame(df_data)
            df.set_index("date", inplace=True)
            df.sort_index(inplace=True)
            df.rename(columns={"close": symbol}, inplace=True)
            
            return df
            
        except Exception as e:
            return pd.DataFrame()

    def _get_empty_results(self) -> Dict[str, str]:
        """Return empty results dictionary."""
        return {group: "" for group in self.feature_groups.keys()}

