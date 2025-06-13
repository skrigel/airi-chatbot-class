# Migration Guide: AIRI Chatbot v2.0

## Overview

The AIRI Chatbot now uses a clean, modular architecture with working citations and robust initialization. This guide explains the changes and how to use the new system.

## What Changed

### Before (Monolithic Structure)
```
airi-chatbot-class/
├── airi_adapter.py         # 951 lines - everything in one file
├── app.py                  # Simple entry point
├── vector_store.py         # Vector storage logic
├── gemini_model.py         # AI model wrapper
├── monitor.py              # Query monitoring
├── simple_vector_store.py  # Backup storage
├── info_files/             # Data files
├── chroma_db/              # Vector database
├── doc_snippets/           # Document snippets
└── frontend/               # Frontend files
```

### After (Modular Architecture)
```
airi-chatbot-class/
├── src/
│   ├── api/                    # Web API layer
│   │   ├── routes/
│   │   │   ├── chat.py         # Chat endpoints
│   │   │   ├── health.py       # Health/status endpoints
│   │   │   └── snippets.py     # Document snippet endpoints
│   │   └── app.py              # Flask app factory
│   ├── core/                   # Business logic
│   │   ├── models/             # AI models
│   │   │   ├── gemini.py
│   │   │   └── base.py
│   │   ├── storage/            # Data storage
│   │   │   ├── vector_store.py
│   │   │   ├── simple_store.py
│   │   │   └── document_processor.py
│   │   ├── query/              # Query processing
│   │   │   ├── processor.py
│   │   │   └── monitor.py
│   │   └── services/           # Business services
│   │       ├── chat_service.py
│   │       └── citation_service.py
│   ├── config/                 # Configuration
│   │   ├── settings.py
│   │   └── logging.py
│   └── utils/                  # Utilities (future use)
├── data/                       # Data storage (moved)
│   ├── info_files/
│   ├── chroma_db/
│   ├── doc_snippets/
│   └── simple_store.pkl
├── scripts/                    # Utility scripts
│   ├── rebuild_database.py
│   └── setup.py
├── frontend/                   # Frontend files
├── requirements.txt
└── main.py                     # New application entry point
```

## Key Improvements

### 1. **Separation of Concerns**
- **API Layer**: Handles HTTP requests/responses
- **Core Business Logic**: Models, storage, query processing
- **Services**: Orchestrate business operations
- **Configuration**: Centralized settings management

### 2. **Dependency Injection**
- Services are injected into routes
- Easy to mock for testing
- Better error handling

### 3. **Configuration Management**
- All settings in `src/config/settings.py`
- Environment variable support
- Type hints and validation

### 4. **Improved Error Handling**
- Proper logging throughout
- Graceful fallbacks
- Better error messages

### 5. **Maintainability**
- Small, focused modules
- Clear interfaces
- Easy to extend and modify

## How to Use the New System

### 1. **Start the Application**
```bash
# Current method
python3 main.py

# The old app.py and airi_adapter.py have been refactored
```

### 2. **Setup and Configuration**
```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export GEMINI_API_KEY="your-api-key-here"

# The system auto-initializes on startup
```

### 3. **Environment Variables**
```bash
# Set your API key
export GEMINI_API_KEY="your-api-key-here"

# Optional: Set custom port
export PORT=8090

# Optional: Set custom repository path
export REPOSITORY_PATH="/path/to/your/data"
```

### 4. **API Endpoints** (unchanged)
- `POST /api/v1/sendMessage` - Non-streaming chat
- `POST /api/v1/stream` - Streaming chat  
- `POST /api/v1/reset` - Reset conversation
- `GET /api/health` - Health check
- `GET /api/snippet/{id}` - Get document snippet

## Backward Compatibility

The API endpoints remain **exactly the same**, so:
- ✅ Frontend code doesn't need changes
- ✅ Existing integrations continue to work
- ✅ Same functionality, better implementation

## Migration Steps

1. **Backup your data** (if you have custom data):
   ```bash
   cp -r chroma_db chroma_db_backup
   cp -r info_files info_files_backup
   ```

2. **Test the new system**:
   ```bash
   python main.py
   ```

3. **Verify functionality**:
   ```bash
   curl -X POST http://localhost:8080/api/v1/sendMessage \
     -H "Content-Type: application/json" \
     -d '{"message": "What does the AI Risk Repository say about job loss risk?", "conversationId": "test"}'
   ```

4. **Update any deployment scripts** to use `main.py` instead of `app.py`

## Benefits Realized

### Performance
- ✅ Better document retrieval for employment queries
- ✅ Improved error handling and fallbacks
- ✅ More efficient caching

### Maintainability  
- ✅ 951-line monolith split into focused modules
- ✅ Clear separation of concerns
- ✅ Easy to add new features
- ✅ Easier testing and debugging

### Reliability
- ✅ Better error handling
- ✅ Graceful degradation
- ✅ Comprehensive logging
- ✅ Configuration validation

## Troubleshooting

### Common Issues

1. **Import errors**: Make sure you're running `python main.py` from the root directory

2. **Missing data**: The `data/` directory should contain your `info_files/`, `chroma_db/`, etc.

3. **Port conflicts**: The system will automatically try alternative ports (8080, 8000, 3000)

4. **Environment variables**: Set `GEMINI_API_KEY` for production use

### Getting Help

1. Check the logs - the new system has comprehensive logging
2. Run `python scripts/setup.py` to verify setup
3. Use `GET /api/health` to check component status

## Old Files

The following files are now **deprecated** but kept for reference:
- `airi_adapter.py` - Replaced by modular architecture
- `app.py` - Replaced by `main.py`
- `vector_store.py` - Moved to `src/core/storage/vector_store.py`
- `gemini_model.py` - Moved to `src/core/models/gemini.py`
- `monitor.py` - Moved to `src/core/query/monitor.py`

These can be safely removed after confirming the new system works correctly.