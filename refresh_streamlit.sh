#!/bin/bash

# Script to refresh/restart Streamlit UI

echo "🔄 Refreshing Streamlit UI..."

# Find Streamlit processes
STREAMLIT_PIDS=$(ps aux | grep streamlit | grep -v grep | awk '{print $2}')

if [ -z "$STREAMLIT_PIDS" ]; then
    echo "No Streamlit processes found."
    echo "Starting Streamlit..."
    cd /Users/zelalemabahana/adversarial-asset-pricing-ai-etrade-ui
    streamlit run streamlit_app.py --server.port 8501
else
    echo "Found Streamlit processes: $STREAMLIT_PIDS"
    echo "Killing existing Streamlit processes..."
    kill -9 $STREAMLIT_PIDS 2>/dev/null
    
    sleep 2
    
    echo "Starting Streamlit..."
    cd /Users/zelalemabahana/adversarial-asset-pricing-ai-etrade-ui
    streamlit run streamlit_app.py --server.port 8501
fi

