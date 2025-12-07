#!/bin/bash
# Quick start script for local Streamlit app

cd /Users/zelalemabahana/adversarial-asset-pricing-ai-etrade-ui

# Try to activate venv if it exists
if [ -f "venv/bin/activate" ]; then
    echo "📦 Activating virtual environment..."
    source venv/bin/activate
fi

# Check if streamlit is available
if ! python3 -c "import streamlit" 2>/dev/null; then
    echo "❌ Streamlit not found. Installing..."
    pip install streamlit
fi

echo ""
echo "🚀 Starting Streamlit app..."
echo "🌐 URL: http://localhost:8501"
echo ""
echo "📊 Logs will appear below:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

streamlit run streamlit_app.py --server.port=8501 --server.address=localhost
