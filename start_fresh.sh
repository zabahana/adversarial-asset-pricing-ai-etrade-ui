#!/bin/bash
# Start Streamlit from the CORRECT directory

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 Starting Streamlit from CORRECT directory"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

cd /Users/zelalemabahana/adversarial-asset-pricing-ai-etrade-ui

# Kill any existing Streamlit processes
echo "🔄 Stopping any existing Streamlit processes..."
pkill -f "streamlit run" 2>/dev/null
sleep 2

# Activate venv
if [ -f "venv/bin/activate" ]; then
    echo "📦 Activating virtual environment..."
    source venv/bin/activate
else
    echo "⚠️  Virtual environment not found, using system Python"
fi

# Start Streamlit
echo ""
echo "✅ Starting Streamlit..."
echo "🌐 URL: http://localhost:8501"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

streamlit run streamlit_app.py \
    --server.port=8501 \
    --server.address=localhost \
    --server.headless=false \
    --logger.level=info
