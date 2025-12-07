#!/usr/bin/env python3
"""
Diagnostic script to check if forecast data is in model results JSON.
"""

import json
from pathlib import Path
import sys

def check_forecast_in_results(results_path: str):
    """Check if forecast data exists in results JSON."""
    
    print("=" * 60)
    print("FORECAST DIAGNOSTIC CHECK")
    print("=" * 60)
    
    results_file = Path(results_path)
    
    if not results_file.exists():
        print(f"❌ Results file not found: {results_path}")
        print(f"   Current directory: {Path.cwd()}")
        print(f"   Please run analysis first to generate results.")
        return False
    
    print(f"✅ Results file found: {results_path}")
    
    try:
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        print(f"\n📊 Results Structure:")
        print(f"   Top-level keys: {list(results.keys())}")
        
        # Check for mha_dqn_robust
        if "mha_dqn_robust" not in results:
            print(f"\n❌ 'mha_dqn_robust' not found in results")
            print(f"   Available model keys: {[k for k in results.keys() if 'dqn' in k.lower() or 'model' in k.lower()]}")
            return False
        
        robust_data = results.get("mha_dqn_robust", {})
        print(f"\n📋 mha_dqn_robust structure:")
        print(f"   Keys: {list(robust_data.keys())}")
        
        # Check available flag
        available = robust_data.get("available", False)
        print(f"\n🔍 Available flag: {available}")
        
        if not available:
            print(f"❌ Forecast not available (available=False)")
            print(f"   Error message: {robust_data.get('error', 'No error message')}")
            return False
        
        # Check forecast fields
        recommendation = robust_data.get("recommendation", None)
        price_change = robust_data.get("price_change_pct", None)
        confidence = robust_data.get("confidence", None)
        
        print(f"\n✅ Forecast Data Found:")
        print(f"   Recommendation: {recommendation}")
        print(f"   Price Change: {price_change}%")
        print(f"   Confidence: {confidence}")
        
        # Check top-level metadata
        last_data_date = results.get("last_data_date", None)
        forecast_date = results.get("forecast_date", None)
        current_price = results.get("current_price", None)
        
        print(f"\n📅 Metadata:")
        print(f"   Last Data Date: {last_data_date}")
        print(f"   Forecast Date: {forecast_date}")
        print(f"   Current Price: ${current_price}")
        
        if recommendation and price_change is not None:
            print(f"\n✅ Forecast appears to be valid!")
            return True
        else:
            print(f"\n⚠️  Forecast data incomplete")
            return False
            
    except Exception as e:
        print(f"\n❌ Error reading results file: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Try common paths
    possible_paths = [
        "results/nvda_model_results.json",
        "results/NVDA_model_results.json",
        f"results/{sys.argv[1].lower()}_model_results.json" if len(sys.argv) > 1 else None,
    ]
    
    found = False
    for path in possible_paths:
        if path and Path(path).exists():
            check_forecast_in_results(path)
            found = True
            break
    
    if not found:
        print("❌ No results file found. Common locations:")
        for path in possible_paths:
            if path:
                print(f"   - {path}")
        print("\n   Run analysis first, then try again with:")
        print(f"   python3 check_forecast.py <TICKER>")


