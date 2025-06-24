# Deployment Instructions

## Deploy Backend to Railway

1. **Create Railway Account**: Go to [railway.app](https://railway.app) and sign up
2. **Connect GitHub**: Link your GitHub account to Railway
3. **Create New Project**: 
   - Click "New Project"
   - Select "Deploy from GitHub repo" 
   - Choose your `airi-chatbot-class` repository
4. **Set Environment Variables**:
   - Go to project settings → Variables
   - Add: `GEMINI_API_KEY` = your actual API key
   - Railway will auto-set `PORT`
5. **Deploy**: Railway will automatically build and deploy using the Dockerfile

## Your Backend URL
After deployment, Railway will give you a URL like:
`https://your-project-name.up.railway.app`

## Deploy Frontend to Netlify

1. **Create Netlify Account**: Go to [netlify.com](https://netlify.com)
2. **Drag & Drop Deploy**:
   - Zip the `frontend/` folder
   - Drag the zip file to Netlify's deploy area
3. **Update API Endpoint**:
   - Before zipping, edit `frontend/assets/index-*.js` 
   - Replace `localhost:8090` with your Railway URL
   - Then zip and deploy

## Alternative: One-Click Deploy

1. **Push to GitHub**: Make sure all changes are pushed
2. **Railway Deploy**: Use GitHub integration for auto-deploys
3. **Netlify Deploy**: Connect GitHub repo for auto-deploys

## Testing Your Deployment

Visit your Netlify URL and test:
- Chat functionality
- Citation links
- Health check at `/api/health`

Your app is now live and shareable! 🚀