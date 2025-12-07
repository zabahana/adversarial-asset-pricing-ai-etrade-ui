#!/usr/bin/env python3
"""
Diagnostic script to test Alpha Vantage API connectivity and identify issues.
"""

import os
import requests
from pathlib import Path
import sys

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from lightning_app.config import ALPHA_VANTAGE_API_KEY, ALPHA_VANTAGE_URL

def test_alpha_vantage_api():
    """Test Alpha Vantage API with diagnostic information."""
    
    print("=" * 60)
    print("Alpha Vantage API Diagnostic Test")
    print("=" * 60)
    print()
    
    # Check API key
    api_key = ALPHA_VANTAGE_API_KEY
    print(f"1. API Key Check:")
    print(f"   - API Key Present: {'Yes' if api_key else 'No'}")
    if api_key:
        print(f"   - API Key (first 8 chars): {api_key[:8]}...")
        print(f"   - API Key Length: {len(api_key)} characters")
    else:
        print("   ❌ ERROR: No API key found!")
        print("   → Set ALPHA_VANTAGE_API_KEY environment variable")
        return False
    print()
    
    # Test with a known good ticker (NVDA)
    test_ticker = "NVDA"
    print(f"2. Testing API Call for {test_ticker}:")
    print(f"   - URL: {ALPHA_VANTAGE_URL}")
    
    params = {
        "function": "TIME_SERIES_DAILY_ADJUSTED",
        "symbol": test_ticker,
        "outputsize": "compact",  # Use compact for faster testing
        "apikey": api_key,
    }
    
    try:
        print("   - Making API request...")
        response = requests.get(ALPHA_VANTAGE_URL, params=params, timeout=30)
        print(f"   - Status Code: {response.status_code}")
        
        payload = response.json()
        
        # Check for errors
        if "Error Message" in payload:
            error_msg = payload["Error Message"]
            print(f"   ❌ ERROR: {error_msg}")
            
            # Provide specific guidance
            error_lower = error_msg.lower()
            if "invalid api call" in error_lower:
                print()
                print("   Diagnostic:")
                print("   → 'Invalid API call' usually means:")
                print("     1. Invalid API key (most common)")
                print("     2. Invalid ticker symbol")
                print("     3. API function parameter issue")
                print()
                print("   Solutions:")
                print("     1. Verify API key at: https://www.alphavantage.co/support/#api-key")
                print("     2. Get a new free API key if needed")
                print("     3. Check if you've exceeded rate limits")
                return False
            elif "api key" in error_lower:
                print("   → Your API key appears to be invalid or expired")
                print("   → Get a new one at: https://www.alphavantage.co/support/#api-key")
                return False
            else:
                print(f"   → Unrecognized error: {error_msg}")
                return False
        
        if "Note" in payload:
            note = payload["Note"]
            print(f"   ⚠️  NOTE: {note}")
            print("   → This usually means rate limit exceeded")
            print("   → Free tier: 5 calls/min, 500/day")
            return False
        
        # Check for successful response
        time_series_key = None
        for key in payload.keys():
            if "Time Series" in key and "Daily" in key:
                time_series_key = key
                break
        
        if time_series_key:
            data = payload[time_series_key]
            print(f"   ✅ SUCCESS! Retrieved {len(data)} days of data")
            print(f"   - Data key: {time_series_key}")
            return True
        else:
            print(f"   ❌ ERROR: No time series data found in response")
            print(f"   - Response keys: {list(payload.keys())}")
            return False
            
    except requests.exceptions.Timeout:
        print("   ❌ ERROR: Request timed out (>30s)")
        print("   → Alpha Vantage may be experiencing high load")
        print("   → Try again in a few minutes")
        return False
    except Exception as e:
        print(f"   ❌ ERROR: {type(e).__name__}: {str(e)}")
        return False

def check_environment():
    """Check environment configuration."""
    print()
    print("3. Environment Check:")
    
    # Check if running in proper directory
    cwd = Path.cwd()
    print(f"   - Current directory: {cwd}")
    
    # Check for config file
    config_file = Path("lightning_app/config.py")
    if config_file.exists():
        print(f"   ✅ Config file found: {config_file}")
    else:
        print(f"   ⚠️  Config file not found at: {config_file}")
    
    # Check API key sources
    env_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if env_key:
        print(f"   ✅ API key found in environment variable")
        print(f"      - Env var key: {env_key[:8]}...")
        if env_key == ALPHA_VANTAGE_API_KEY:
            print(f"   ✅ Environment variable matches config")
        else:
            print(f"   ⚠️  Environment variable differs from config")
    else:
        print(f"   ⚠️  No ALPHA_VANTAGE_API_KEY in environment")
        print(f"   → Using hardcoded key from config.py")

if __name__ == "__main__":
    check_environment()
    print()
    success = test_alpha_vantage_api()
    print()
    print("=" * 60)
    if success:
        print("✅ Diagnostic Test PASSED - API is working correctly!")
    else:
        print("❌ Diagnostic Test FAILED - See errors above")
        print()
        print("Quick Fixes:")
        print("1. Get a free API key: https://www.alphavantage.co/support/#api-key")
        print("2. Set it: export ALPHA_VANTAGE_API_KEY='your_key_here'")
        print("3. Or update lightning_app/config.py with your key")
    print("=" * 60)

