#!/usr/bin/env python3
"""
Startup script for Streamlit app in Cloud Run.
Validates imports before starting to catch errors early.
"""
import os
import sys
import subprocess
import time

def test_imports():
    """Test critical imports to catch errors early."""
    print("Testing imports...")
    try:
        import streamlit
        print(f"✓ Streamlit {streamlit.__version__}")
        
        import pandas
        print(f"✓ Pandas {pandas.__version__}")
        
        import plotly
        print(f"✓ Plotly {plotly.__version__}")
        
        # Test app imports (optional - app will handle missing imports)
        try:
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).parent))
            
            from lightning_app.config import ALPHA_VANTAGE_API_KEY, DEFAULT_TICKER
            print("✓ Config loaded")
            
            from lightning_app.works.data_fetch_work import DataFetchWork
            print("✓ DataFetchWork imported")
            
            from lightning_app.works.feature_engineering_work import FeatureEngineeringWork
            print("✓ FeatureEngineeringWork imported")
            
            from lightning_app.works.model_inference_work import ModelInferenceWork
            print("✓ ModelInferenceWork imported")
            
            from lightning_app.works.sentiment_work import SentimentWork
            print("✓ SentimentWork imported")
            
            from lightning_app.works.macro_work import MacroWork
            print("✓ MacroWork imported")
            
            try:
                from lightning_app.utils.llm_summarizer import ModelResultsSummarizer
                print("✓ ModelResultsSummarizer imported")
            except ImportError:
                print("⚠ ModelResultsSummarizer not available (optional)")
                
        except ImportError as e:
            print(f"⚠ App module import warning: {e}")
            print("(App may still work with fallbacks)")
        
        print("Core imports successful!")
        return True
        
    except Exception as e:
        print(f"✗ Import error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main startup function."""
    port = os.environ.get("PORT", "8501")
    
    print("=" * 50)
    print("Starting Adversarial Asset Pricing App")
    print("=" * 50)
    print(f"Port: {port}")
    print(f"Python: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    print()
    
    # Test imports first
    if not test_imports():
        print("ERROR: Import tests failed!")
        sys.exit(1)
    
    # Check if streamlit_app.py exists
    if not os.path.exists("streamlit_app.py"):
        print("ERROR: streamlit_app.py not found!")
        sys.exit(1)
    
    print()
    print("Starting Streamlit...")
    print()
    
    # Start Streamlit
    cmd = [
        sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
        "--server.port", port,
        "--server.address", "0.0.0.0",
        "--server.headless", "true",
        "--server.enableCORS", "false",
        "--server.enableXsrfProtection", "false",
        "--server.allowRunOnSave", "false",
        "--browser.gatherUsageStats", "false",
        "--server.runOnSave", "false",
        "--server.fileWatcherType", "none",
        "--server.maxUploadSize", "200",
        "--server.maxMessageSize", "200",
        "--logger.level", "error",
        "--server.healthCheckPath", "/_stcore/health"
    ]
    
    # Execute Streamlit (this will block)
    try:
        os.execvpe(sys.executable, cmd, os.environ)
    except Exception as e:
        print(f"ERROR starting Streamlit: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

