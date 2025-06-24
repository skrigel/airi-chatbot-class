#!/bin/bash

echo "🚀 AIRI Chatbot Deployment Script"
echo "=================================="

# Check if backend URL is provided
if [ -z "$1" ]; then
    echo "❌ Error: Please provide your Railway backend URL"
    echo "Usage: ./deploy.sh https://your-backend.up.railway.app"
    exit 1
fi

BACKEND_URL=$1
echo "📡 Backend URL: $BACKEND_URL"

# Update frontend API patch with backend URL
echo "🔧 Updating frontend configuration..."
sed -i.bak "s|API_BASE_URL = window.location.origin; // For same-origin deployment|API_BASE_URL = '$BACKEND_URL'; // Backend URL|g" frontend/api-patch.js

# Update netlify.toml with backend URL
echo "🔧 Updating Netlify configuration..."
sed -i.bak "s|https://your-backend.up.railway.app|$BACKEND_URL|g" netlify.toml

echo "✅ Configuration updated!"
echo ""
echo "📦 Next steps:"
echo "1. Commit and push these changes to GitHub"
echo "2. Deploy backend to Railway using your GitHub repo"
echo "3. Deploy frontend to Netlify by dragging the 'frontend' folder"
echo ""
echo "🔗 Your app will be accessible via the Netlify URL!"

# Restore backup files
mv frontend/api-patch.js.bak frontend/api-patch.js.backup 2>/dev/null
mv netlify.toml.bak netlify.toml.backup 2>/dev/null