# Integration Guide: Advanced Backend with GitHub Frontend

This guide explains how to integrate your enhanced backend with the GitHub UI frontend.

## Overview

The integration uses an adapter pattern to bridge between:
- Your current backend with advanced RAG capabilities and fallback mechanisms
- The GitHub frontend UI that expects a different API contract

## Key Components

1. **Your Enhanced Backend**
   - `vector_store.py`: Contains your RAG implementation with robust fallbacks
   - `app.py`: Your Flask server with RAG-enabled endpoints
   - `monitor.py`: Analyzes user queries
   - `gemini_model.py`: Interfaces with Gemini API

2. **GitHub Frontend**
   - Modern UI in TypeScript with Tailwind CSS
   - Expects API endpoints at `/api/v1/sendMessage` and `/api/v1/stream`

3. **Adapter Layer**
   - `adapter.py`: Bridges between the two systems
   - Maintains compatibility with GitHub frontend API expectations
   - Leverages your enhanced backend functionality

## Integration Steps

### 1. Clone the GitHub Repository

```bash
git clone https://github.com/skrigel/airi-chatbot-class.git
cd airi-chatbot-class
```

### 2. Build the GitHub Frontend

```bash
cd chatbot
npm install
npm run build
```

This creates a production build of the frontend.

### 3. Create Directory for the Frontend Build

```bash
cd ..
mkdir github-frontend-build
cp -r chatbot/dist/* github-frontend-build/
```

### 4. Copy Your Enhanced Backend Files

Ensure the following files are in your project root:
- `vector_store.py` (with your fallback mechanisms)
- `app.py`
- `monitor.py`
- `gemini_model.py`
- `requirements.txt`
- `adapter.py` (the adapter script)

### 5. Install Requirements

```bash
pip install -r requirements.txt
```

### 6. Run the Adapter Server

```bash
python adapter.py
```

The adapter server will:
- Initialize your enhanced RAG system
- Expose API endpoints that the GitHub frontend expects
- Translate between the two API formats

### 7. Access the Application

Open your browser to `http://localhost:8090`

## How the Integration Works

### API Transformation

The adapter performs these key transformations:

1. **Request Transformation**
   - GitHub frontend sends: `{message: "...", conversationId: "..."}`
   - Adapter transforms to: `{message: "...", conversation_id: "..."}`

2. **Response Transformation**
   - Your backend returns: `{response: "...", conversation_id: "..."}`
   - Adapter transforms to: `{id: "...", response: "...", status: "complete"}`

3. **Streaming Format Adaptation**
   - Adapts your SSE format to what the GitHub frontend expects

### Authentication

Both systems use the same Gemini API key (taken from environment variables or using the default).

## Deployment Considerations

For production deployment:
1. Don't use the hardcoded API key
2. Set proper CORS settings if deploying frontend and backend separately
3. Consider using proper web server (Gunicorn, etc.) instead of Flask's built-in server

## Troubleshooting

If you encounter issues:

1. **API Format Mismatch**
   - Check the browser's network tab to see what format the frontend is sending
   - Adjust adapter.py accordingly

2. **CORS Issues**
   - These appear in browser console
   - Ensure CORS is properly configured in adapter.py

3. **Frontend Build Problems**
   - Delete node_modules and reinstall
   - Check for TypeScript errors in the frontend code

4. **Backend Initialization Failures**
   - Check log messages
   - Ensure all paths are correct
   - Verify API key is valid