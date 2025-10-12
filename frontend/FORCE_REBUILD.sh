#!/bin/bash

echo "🔧 Forcing complete frontend rebuild..."
echo ""

cd /Users/macbook2024/Library/CloudStorage/Dropbox/Droid-FineTuning/frontend

echo "1. Stopping any running processes..."
pkill -f "react-scripts" || true
sleep 2

echo "2. Clearing all caches..."
rm -rf node_modules/.cache
rm -rf build
rm -rf .cache

echo "3. Clearing browser cache files..."
rm -rf public/static

echo "4. Rebuilding..."
npm run build

echo ""
echo "✅ Rebuild complete!"
echo ""
echo "Now start the dev server:"
echo "  npm start"
echo ""
echo "Then in browser:"
echo "  1. Open http://localhost:3000"
echo "  2. Press Cmd+Shift+R (hard refresh)"
echo "  3. Or open DevTools (F12) > Application > Clear Storage > Clear site data"
