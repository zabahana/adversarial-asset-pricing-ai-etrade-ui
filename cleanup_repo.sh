#!/bin/bash
set -e

echo "🧹 Cleaning Repository for Production"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Delete fix documentation files
echo "📝 Removing fix documentation files..."
rm -f *fix*.md
rm -f *FIX*.md
rm -f DUPLICATE_DISPLAY*.md
rm -f FIX_*.md
rm -f PERFORMANCE_METRICS_FIX*.md
echo "✅ Removed fix documentation"

# Delete temporary Python files
echo "🐍 Removing temporary Python files..."
rm -f *temp*.py
rm -f *_original.py
echo "✅ Removed temporary files"

# Delete fallback utility file
echo "🔄 Removing fallback logic files..."
rm -f lightning_app/utils/realistic_fallback_metrics.py
echo "✅ Removed fallback utilities"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Cleanup Complete!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

