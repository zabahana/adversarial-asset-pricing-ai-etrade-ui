from __future__ import annotations

from typing import Dict

import pandas as pd
import requests
from . import LightningWork, HAS_LIGHTNING

FRED_ENDPOINT = "https://api.stlouisfed.org/fred/series/observations"
MACRO_SERIES = {
    "inflation": {"series_id": "CPIAUCSL", "description": "CPI (YoY %)"},
    "growth": {"series_id": "GDPC1", "description": "Real GDP (SAAR)"},
    "unemployment": {"series_id": "UNRATE", "description": "Unemployment Rate"},
}


class MacroWork(LightningWork):
    """Pulls macroeconomic series from the FRED API."""

    def run(self, years: int = 5) -> Dict[str, Dict[str, object]]:
        from ..config import FRED_API_KEY
        
        fred_key = FRED_API_KEY
        if not fred_key:
            # Return mock data if FRED API key not provided
            return {
                "inflation": {"description": "CPI (YoY %)", "latest": 3.2},
                "growth": {"description": "Real GDP (SAAR)", "latest": 2.1},
                "unemployment": {"description": "Unemployment Rate", "latest": 3.7},
            }

        results: Dict[str, Dict[str, object]] = {}
        for key, info in MACRO_SERIES.items():
            params = {
                "series_id": info["series_id"],
                "api_key": fred_key,
                "file_type": "json",
            }
            # Retry logic with exponential backoff for timeout issues
            import time
            max_retries = 3
            timeout_seconds = 60  # Increased timeout to 60 seconds
            
            for attempt in range(max_retries):
                try:
                    response = requests.get(FRED_ENDPOINT, params=params, timeout=timeout_seconds)
                    response.raise_for_status()
                    break  # Success, exit retry loop
                except requests.exceptions.Timeout as e:
                    if attempt < max_retries - 1:
                        wait_time = (2 ** attempt) * 5  # Exponential backoff: 5s, 10s, 20s
                        print(f"[WARNING] FRED API timeout for {key} (attempt {attempt + 1}/{max_retries}). Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        print(f"[ERROR] FRED API timeout for {key} after {max_retries} attempts")
                        raise
                except requests.exceptions.RequestException as e:
                    print(f"[ERROR] FRED API request error for {key}: {e}")
                    raise
            observations = response.json().get("observations", [])
            df = pd.DataFrame(observations)
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            df["date"] = pd.to_datetime(df["date"])
            if years:
                cutoff = pd.Timestamp.today() - pd.DateOffset(years=years)
                df = df[df["date"] >= cutoff]
            results[key] = {
                "description": info["description"],
                "latest": float(df["value"].iloc[-1]),
                "history": df[["date", "value"]].to_dict("records"),
            }
        return results
