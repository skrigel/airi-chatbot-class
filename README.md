# AIRI Chatbot

A sophisticated chatbot for the MIT AI Risk Repository with advanced RAG capabilities and a modern React frontend.

## Quick Start for Your Teammate

```bash
# Clone the repository
git clone https://github.com/YOUR-USERNAME/airi-chatbot.git
cd airi-chatbot

# Make the setup and run scripts executable
chmod +x setup.sh run.sh

# Complete first-time setup (builds frontend, installs dependencies)
./setup.sh

# Run the chatbot
./run.sh
```

That's it! The chatbot will be available at http://localhost:5000 (or another port if 5000 is unavailable).

## What Each Script Does

### setup.sh (run once when first setting up)
- Installs all Python dependencies
- Builds/sets up the frontend UI
- Creates necessary directories
- Prepares the repository structure

### run.sh (use each time to start the chatbot)
- Detects the correct Python interpreter
- Verifies Flask is installed (and installs it if missing)
- Starts the Flask server with the AIRI adapter
- Automatically finds an available port if the default is in use

## Features

- **Robust RAG System**: Hybrid search combining vector and keyword-based retrieval
- **Clickable Citations**: References link to original document sources
- **Domain Detection**: Special handling for topic-specific queries
- **Streaming Responses**: Real-time text generation
- **Modern UI**: React-based frontend with responsive design
- **Multi-level Fallbacks**: Resilient operation with graceful degradation
- **Excel-specific References**: Sheet and row citations for structured data

## System Architecture

The adapter integrates three main components:

1. **Vector Store** (`vector_store.py`): Advanced RAG implementation with hybrid search
2. **Query Monitor** (`monitor.py`): Analyzes and classifies user questions
3. **LLM Integration** (`gemini_model.py`): Uses Gemini models for response generation

## Advanced RAG Implementation

The system uses a sophisticated Retrieval-Augmented Generation (RAG) approach:

1. **Hybrid Retrieval**: Combines vector search (semantic similarity) with keyword search (BM25)
2. **Query Preprocessing**: Enhances queries with key terms and special handling for questions
3. **Smart Context Formatting**: Organizes retrieved information with clear structure and source citations
4. **Domain-Specific Handling**: Special processing for different query types
5. **Performance Optimization**: Caching mechanism for frequently asked questions
6. **Robust Fallback Mechanisms**: Multi-level fallbacks for reliable operation

## Detailed Setup Instructions

If you need more control over the setup process:

### Prerequisites

- Python 3.6+
- Valid Gemini API key

### Step-by-Step Setup

1. **Make scripts executable**:
   ```bash
   chmod +x run.sh install_deps.sh setup.sh
   ```

2. **Install dependencies**:
   ```bash
   ./install_deps.sh
   ```

3. **Set up the frontend** (only needed if frontend files are missing):
   ```bash
   ./setup.sh
   ```

4. **Prepare the MIT AI Risk Repository data**:
   - Place AI Risk Repository files in the `info_files` directory
   - The application automatically processes Excel, CSV, and text files

5. **Start the application**:
   ```bash
   ./run.sh
   ```

### Troubleshooting for First-Time Users

- **If Flask installation fails**, try manually:
  ```bash
  pip install --user flask flask-cors
  # or
  python3 -m pip install --user flask flask-cors
  ```

- **For complete installation** of all dependencies:
  ```bash
  pip install --user -r requirements.txt
  ```

- **Frontend Not Loading**: Run `./setup.sh` to rebuild the frontend
- **Port Conflicts**: The app automatically finds an available port, or specify one:
  ```bash
  ./run.sh 8090
  ```
- **Missing Documents**: Add files to the `info_files` directory

## License

See the LICENSE file for details.