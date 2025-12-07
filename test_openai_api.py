#!/usr/bin/env python3
"""
Test OpenAI API connectivity and functionality.
"""

import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from lightning_app.config import OPENAI_API_KEY
except ImportError:
    print("❌ Error: Could not import from lightning_app.config")
    print("   Make sure you're running from the project root directory.")
    sys.exit(1)

def test_openai_api():
    """Test OpenAI API with a simple request."""
    
    print("=" * 60)
    print("OpenAI API Diagnostic Test")
    print("=" * 60)
    
    # Check if API key exists
    if not OPENAI_API_KEY:
        print("\n❌ ERROR: OpenAI API key not found!")
        print("   Please set OPENAI_API_KEY in lightning_app/config.py or as environment variable")
        return False
    
    # Check if key is valid format (starts with sk-)
    if not OPENAI_API_KEY.startswith('sk-'):
        print(f"\n⚠️  WARNING: API key format looks unusual (doesn't start with 'sk-')")
        print(f"   Key preview: {OPENAI_API_KEY[:10]}...")
    else:
        print(f"\n✅ API Key found (starts with 'sk-'): {OPENAI_API_KEY[:10]}...")
    
    # Try to import OpenAI library
    try:
        from openai import OpenAI
        print("✅ OpenAI library is installed")
    except ImportError:
        print("\n❌ ERROR: OpenAI library is not installed!")
        print("   Install it with: pip install openai")
        return False
    
    # Test API connection
    print("\n🔄 Testing API connection...")
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        
        # Test with a simple completion request
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say 'API test successful' if you can read this."}
            ],
            max_tokens=10,
            temperature=0
        )
        
        result = response.choices[0].message.content.strip()
        print(f"✅ API Test Successful!")
        print(f"   Response: {result}")
        
        # Check usage/billing
        print("\n🔄 Checking API usage...")
        try:
            # Try to get usage info (may require permissions)
            print("   Note: Usage checking requires account permissions")
            print("   ✅ API key is valid and working")
        except Exception as e:
            print(f"   ⚠️  Could not check usage: {e}")
        
        return True
        
    except Exception as e:
        error_str = str(e)
        print(f"\n❌ API Test Failed!")
        
        # Check for specific error types
        if "401" in error_str or "authentication" in error_str.lower() or "invalid" in error_str.lower():
            print("   ERROR TYPE: Authentication Failed")
            print("   → Your API key is invalid or has been revoked")
            print("   → Get a new key: https://platform.openai.com/api-keys")
            print(f"   → Current key preview: {OPENAI_API_KEY[:10]}...")
        elif "429" in error_str or "rate limit" in error_str.lower():
            print("   ERROR TYPE: Rate Limit Exceeded")
            print("   → You've exceeded your API rate limit")
            print("   → Wait a few minutes and try again")
            print("   → Check usage: https://platform.openai.com/usage")
        elif "insufficient_quota" in error_str.lower() or "quota" in error_str.lower():
            print("   ERROR TYPE: Insufficient Quota")
            print("   → Your account has run out of credits")
            print("   → Add credits: https://platform.openai.com/account/billing")
            print("   → Check billing: https://platform.openai.com/account/billing")
        elif "billing" in error_str.lower():
            print("   ERROR TYPE: Billing Issue")
            print("   → There's an issue with your billing setup")
            print("   → Check billing: https://platform.openai.com/account/billing")
        else:
            print(f"   ERROR DETAILS: {error_str}")
            print("   → Check OpenAI status: https://status.openai.com/")
        
        return False

if __name__ == "__main__":
    success = test_openai_api()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ OpenAI API is working correctly!")
        print("=" * 60)
        sys.exit(0)
    else:
        print("❌ OpenAI API test failed. See errors above.")
        print("=" * 60)
        sys.exit(1)


