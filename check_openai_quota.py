#!/usr/bin/env python3
"""
OpenAI API Quota Checker and Diagnostics

This script helps diagnose and fix OpenAI API quota issues.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

def check_openai_api_key():
    """Check if OpenAI API key is configured."""
    from lightning_app.config import OPENAI_API_KEY
    
    if not OPENAI_API_KEY or OPENAI_API_KEY == "":
        print("❌ OpenAI API key is not set in lightning_app/config.py")
        return False
    
    if OPENAI_API_KEY.startswith("sk-proj-"):
        print(f"✅ OpenAI API key is configured (starts with: {OPENAI_API_KEY[:12]}...)")
        return True
    else:
        print(f"⚠️  OpenAI API key format looks incorrect: {OPENAI_API_KEY[:20]}...")
        return False

def test_openai_api():
    """Test OpenAI API access and check quota status."""
    from lightning_app.config import OPENAI_API_KEY
    
    if not OPENAI_API_KEY:
        print("❌ Cannot test API: No API key configured")
        return False
    
    try:
        try:
            from openai import OpenAI
        except ImportError:
            print("\n❌ OpenAI Python package not installed")
            print("="*60)
            print("📋 TO INSTALL:")
            print("   pip install openai")
            print("   or")
            print("   pip install -r requirements.txt")
            print("="*60 + "\n")
            return False
            
        client = OpenAI(api_key=OPENAI_API_KEY)
        
        print("🔄 Testing OpenAI API connection...")
        
        # Try a simple test call
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": "Say 'API test successful' if you can read this."}
            ],
            max_tokens=10
        )
        
        print("✅ OpenAI API is working! Quota is available.")
        print(f"   Response: {response.choices[0].message.content}")
        return True
        
    except Exception as e:
        error_str = str(e)
        
        if "429" in error_str or "quota" in error_str.lower() or "insufficient_quota" in error_str.lower():
            print("\n" + "="*60)
            print("❌ OPENAI QUOTA EXCEEDED")
            print("="*60)
            print("\nYour OpenAI API key has exceeded its quota limit.")
            print("\n📋 HOW TO FIX:")
            print("\n1. Check your OpenAI account:")
            print("   → Visit: https://platform.openai.com/usage")
            print("   → Log in and check your usage and billing")
            
            print("\n2. Add credits or upgrade your plan:")
            print("   → Visit: https://platform.openai.com/account/billing")
            print("   → Add payment method if needed")
            print("   → Set up usage limits or billing budget")
            
            print("\n3. Get a new API key (if needed):")
            print("   → Visit: https://platform.openai.com/api-keys")
            print("   → Create a new API key")
            print("   → Update lightning_app/config.py with the new key")
            
            print("\n4. Check free tier limits:")
            print("   → Free tier has limited credits")
            print("   → Consider upgrading to a paid plan for production use")
            
            print("\n💡 Quick Fix - Update API Key:")
            print("   1. Open: lightning_app/config.py")
            print("   2. Replace OPENAI_API_KEY with your new key")
            print("   3. Restart the application")
            print("\n" + "="*60 + "\n")
            
            return False
        elif "401" in error_str or "authentication" in error_str.lower() or "invalid" in error_str.lower():
            print("\n❌ INVALID API KEY")
            print("="*60)
            print("Your OpenAI API key is invalid or expired.")
            print("\n📋 HOW TO FIX:")
            print("1. Visit: https://platform.openai.com/api-keys")
            print("2. Create a new API key")
            print("3. Update lightning_app/config.py with the new key")
            print("="*60 + "\n")
            return False
        else:
            print(f"\n⚠️  Unexpected error: {e}")
            return False

def check_usage_stats():
    """Check OpenAI API usage statistics."""
    from lightning_app.config import OPENAI_API_KEY
    
    if not OPENAI_API_KEY:
        return
    
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        
        print("\n📊 Checking usage statistics...")
        
        # Note: OpenAI doesn't have a direct usage API endpoint in the Python SDK
        # Users need to check on the web dashboard
        print("   → Usage stats available at: https://platform.openai.com/usage")
        print("   → Billing info available at: https://platform.openai.com/account/billing")
        
    except Exception as e:
        print(f"   ⚠️  Could not check usage: {e}")

def main():
    """Main diagnostic function."""
    print("="*60)
    print("OpenAI API Quota Diagnostic Tool")
    print("="*60)
    print()
    
    # Step 1: Check if API key is configured
    has_key = check_openai_api_key()
    print()
    
    if not has_key:
        print("📋 TO CONFIGURE OPENAI API KEY:")
        print("   1. Visit: https://platform.openai.com/api-keys")
        print("   2. Create a new API key")
        print("   3. Update: lightning_app/config.py")
        print("   4. Set OPENAI_API_KEY = 'your-new-key-here'")
        return
    
    # Step 2: Test API connection
    is_working = test_openai_api()
    print()
    
    # Step 3: Check usage (informational)
    check_usage_stats()
    print()
    
    if is_working:
        print("✅ All checks passed! OpenAI API is ready to use.")
    else:
        print("❌ OpenAI API is not working. Follow the instructions above to fix.")

if __name__ == "__main__":
    main()

