# 🔑 Setting Up API Keys for Streamlit Cloud

## Step 1: Get Your API Keys

### Alpha Vantage API Key (Required)
1. Go to: https://www.alphavantage.co/support/#api-key
2. Enter your email address
3. Copy the API key you receive

### OpenAI API Key (Optional - for LLM summaries)
1. Go to: https://platform.openai.com/api-keys
2. Sign in or create an account
3. Click "Create new secret key"
4. Copy the key (you won't be able to see it again!)

## Step 2: Add Keys to Streamlit Cloud

After your app is deployed on Streamlit Cloud:

1. **Go to your app**: https://share.streamlit.io
2. **Click on your app** → "Manage app"
3. **Click "Secrets"** in the left sidebar
4. **Paste this template** and fill in your keys:

```toml
ALPHA_VANTAGE_API_KEY = "YOUR_ALPHA_VANTAGE_KEY_HERE"
OPENAI_API_KEY = "YOUR_OPENAI_KEY_HERE"
```

5. **Click "Save"**
6. **Restart your app** (if it's running)

## Step 3: Verify It Works

1. Go to your Streamlit app
2. Try analyzing a stock (e.g., "NVDA")
3. Check the logs if there are any API errors

## 🔒 Security Notes

- ✅ Keys are encrypted and secure in Streamlit Cloud
- ✅ Never commit API keys to GitHub
- ✅ Keys are only accessible to your app
- ✅ You can update keys anytime in the Secrets section

## 📋 Quick Copy Template

Copy and paste this into Streamlit Cloud Secrets, then replace with your actual keys:

```toml
ALPHA_VANTAGE_API_KEY = "your-alpha-vantage-key"
OPENAI_API_KEY = "your-openai-key"
```

