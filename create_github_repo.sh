#!/bin/bash
set -e

echo "🔧 GitHub Repository Setup"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

REPO_NAME="adversarial-asset-pricing-ai-etrade-ui"
GITHUB_USER="zabahana"
REPO_URL="https://github.com/${GITHUB_USER}/${REPO_NAME}.git"

echo "📋 Repository Details:"
echo "   • User: $GITHUB_USER"
echo "   • Repository: $REPO_NAME"
echo "   • URL: $REPO_URL"
echo ""

# Check if GitHub CLI is available
if command -v gh &> /dev/null; then
    echo "✅ GitHub CLI (gh) detected!"
    read -p "Create repository using GitHub CLI? (y/n): " USE_CLI
    
    if [[ $USE_CLI == "y" ]]; then
        echo ""
        echo "📦 Creating repository on GitHub..."
        gh repo create "$REPO_NAME" \
            --public \
            --description "MHA-DQN Portfolio Optimization with Streamlit UI - Enterprise Agentic AI System for Asset Pricing" \
            --source=. \
            --remote=origin \
            --push
        
        echo ""
        echo "✅ Repository created and pushed!"
        exit 0
    fi
fi

# Manual instructions
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📝 MANUAL STEPS:"
echo ""
echo "1. Go to: https://github.com/new"
echo ""
echo "2. Fill in the form:"
echo "   • Repository name: $REPO_NAME"
echo "   • Description: MHA-DQN Portfolio Optimization with Streamlit UI"
echo "   • Visibility: Public (or Private)"
echo "   • ⚠️  DO NOT initialize with README, .gitignore, or license"
echo "   • Click 'Create repository'"
echo ""
echo "3. After creating, run this command to push:"
echo ""
echo "   git push -u origin main"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

read -p "Press Enter after you've created the repository on GitHub..."
echo ""
echo "🚀 Pushing to GitHub..."
git push -u origin main

echo ""
echo "✅ Done! Your code is now on GitHub!"
echo "   View at: $REPO_URL"

