# 🚀 Quick Deploy Guide - Make Your Chatbot Live in 10 Minutes!

## What We've Prepared

Your chatbot is now **deployment-ready** with:
- ✅ Docker containerization
- ✅ Production CORS settings  
- ✅ Railway deployment config
- ✅ Netlify frontend setup
- ✅ Flexible API routing

## Deploy in 3 Simple Steps

### Step 1: Deploy Backend (5 minutes)
1. Go to [railway.app](https://railway.app) and sign up
2. Click "New Project" → "Deploy from GitHub repo"
3. Select your `airi-chatbot-class` repository
4. Add environment variable: `GEMINI_API_KEY` = your actual API key
5. Wait for deployment - you'll get a URL like `https://your-app.up.railway.app`

### Step 2: Deploy Frontend (3 minutes)
1. Go to [netlify.com](https://netlify.com) and sign up
2. Drag and drop your `frontend/` folder to Netlify
3. Your frontend is live instantly!

### Step 3: Test Everything (2 minutes)
1. Visit your Netlify URL
2. Test the chat functionality
3. Check that citations work
4. Share the URL with your team!

## Alternative: Separate Backend/Frontend

If you want different URLs for backend and frontend:

1. **Update API URL**: Edit `frontend/api-patch.js` line 17:
   ```javascript
   API_BASE_URL = 'https://your-backend.up.railway.app';
   ```

2. **Update Netlify**: Edit `netlify.toml` line 7:
   ```toml
   to = "https://your-backend.up.railway.app/api/:splat"
   ```

## Files We Created for You

- `Dockerfile` - Containerizes your Python app
- `railway.json` - Railway deployment config
- `netlify.toml` - Netlify routing config
- `.env.example` - Production environment template
- `deploy.sh` - Automated configuration script
- `DEPLOYMENT.md` - Detailed instructions

## Your App is Ready! 🎉

**Your exact UI and functionality**, now accessible via a clean URL that anyone can visit and test. No code changes needed - just deploy and share!

**Result**: Professional chatbot with working citations, accessible anywhere, ready for user feedback.