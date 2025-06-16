# AIRI Chatbot

A sophisticated chatbot for the MIT AI Risk Repository with advanced RAG capabilities, modular architecture, and clickable citations.

## Quick Start

```bash
# Clone the repository
git clone https://github.com/skrigel/airi-chatbot-class.git
cd airi-chatbot-class

# Create and activate virtual environment (IMPORTANT!)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set your API key (required)
export GEMINI_API_KEY="your-api-key-here"

# Run the chatbot
python3 main.py
```

The chatbot will automatically find an available port (usually 8000, 8080, or 8090) and display the URL.

## Key Features

- **Smart Employment Query Processing**: Enhanced handling for job-related AI risk questions
- **Clickable Citations**: Each response includes links to view original source documents
- **Modular Architecture**: Clean, maintainable codebase with proper separation of concerns
- **Automatic Port Detection**: Finds available ports automatically
- **Robust Error Handling**: Graceful fallbacks and detailed status reporting

## System Architecture

The system uses a clean, modular architecture:

```
src/
├── api/                     # Web API layer
│   ├── routes/             # Route handlers (chat, health, snippets)
│   └── app.py              # Flask application factory
├── core/                   # Business logic
│   ├── models/             # AI model implementations
│   ├── storage/            # Data storage and retrieval
│   ├── query/              # Query processing and analysis
│   └── services/           # Orchestration services
└── config/                 # Configuration and settings
```

### Key Components

1. **Vector Store**: Hybrid search combining semantic similarity with keyword matching
2. **Query Processor**: Enhanced handling for employment and domain-specific queries  
3. **Citation Service**: Generates clickable links to source documents
4. **Chat Service**: Orchestrates the complete conversation flow

## Configuration

### Environment Variables

Create a `.env` file or set environment variables:

```bash
# Required
GEMINI_API_KEY=your-api-key-here

# Optional
PORT=8090
REPOSITORY_PATH=/path/to/data/info_files
```

### Data Setup

Place your MIT AI Risk Repository files in `data/info_files/`:
- Excel files (`.xlsx`, `.xls`) 
- Text files (`.txt`, `.md`)
- The system automatically processes and indexes all files

## Example Queries

Try these queries to test the system:

**Employment & Job Impact:**
- "How will AI affect my job as an accountant?"
- "What are the risks of AI automation in the workplace?"

**General AI Risks:**  
- "What are the main types of AI risks?"
- "What concerns exist regarding bias in AI systems?"

**Domain-Specific:**
- "What are the risks of AI in healthcare?"
- "What might happen if self-driving cars get compromised?"

## API Endpoints

The system provides RESTful API endpoints:

- `POST /api/v1/stream` - Streaming chat responses
- `POST /api/v1/sendMessage` - Non-streaming chat
- `GET /api/health` - System health check
- `GET /snippet/{id}` - View source document citations

## Troubleshooting

**Common Issues:**

- **Port Conflicts**: The app automatically finds available ports (8000, 8080, 8090)
- **Missing API Key**: Set `GEMINI_API_KEY` environment variable
- **No Citations**: Ensure documents are in `data/info_files/` directory
- **Frontend Issues**: Frontend files are included, no additional setup needed

## License

See the LICENSE file for details.