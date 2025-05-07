# AI Risk Repository Chatbot

A RAG (Retrieval-Augmented Generation) chatbot for the MIT AI Risk Repository that helps users navigate and understand AI risks.

## New Features

- **Modern UI**: Enhanced frontend with TypeScript and Tailwind CSS
- **Advanced RAG**: Hybrid retrieval with vector and keyword search
- **Robust Fallbacks**: Multi-level error handling for reliable operation
- **Domain-Specific Knowledge**: Pre-defined information for when retrieval fails
- **API Adapter**: Seamless integration between backend and frontend

## System Architecture

### Components

1. **Enhanced Backend**
   - **Model Interface**: `gemini_model.py` handles interactions with the Gemini API
   - **Monitor**: `monitor.py` analyzes user inquiries to determine type and appropriateness
   - **Vector Store**: `vector_store.py` for efficient document retrieval with robust fallbacks
   - **Backend API**: Flask-based REST API in `app.py` with streaming support
   - **Adapter**: Integration layer in `adapter.py` that bridges backends and frontend

2. **Modern Frontend**
   - TypeScript-based React application
   - Tailwind CSS for styling
   - Optimized build with Vite

## Setup Instructions

### Prerequisites

- Python 3.8+
- Node.js 14+
- Flask
- Google Generative AI Python SDK
- Valid Gemini API key

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/skrigel/airi-chatbot-class.git
   cd airi-chatbot-class
   ```

2. Set up a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install Python dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Build the frontend:
   ```
   cd chatbot
   npm install
   npm run build
   mkdir -p ../github-frontend-build
   cp -r dist/* ../github-frontend-build/
   cd ..
   ```

5. Configure environment variables:
   ```
   export GEMINI_API_KEY=your-api-key
   export GEMINI_MODEL_NAME=gemini-2.0-flash  # or another Gemini model
   ```

6. Prepare the MIT AI Risk Repository data:
   
   Ensure the AI Risk Repository Excel file is in the `info_files` directory. The application automatically detects and processes any Excel files in this directory that contain "risk" or "repository" in their filename.

### Running the Application

1. Start the adapter server:
   ```
   python adapter.py
   ```

2. Access the web interface:
   Open your browser and navigate to `http://localhost:8090`

## Advanced RAG Implementation

The system uses a sophisticated Retrieval-Augmented Generation (RAG) approach:

1. **Hybrid Retrieval**: Combines vector search (semantic similarity) with keyword search (BM25) for more comprehensive results
2. **Query Preprocessing**: Enhances queries with key terms and special handling for questions
3. **Smart Context Formatting**: Organizes retrieved information with clear structure and source citations
4. **Domain-Specific Handling**: Special processing for different types of questions (e.g., employment-related queries)
5. **Performance Optimization**: Caching mechanism for frequently asked questions
6. **Robust Fallback Mechanisms**: Multi-level fallbacks for reliable operation even with problematic data

### Fallback Mechanisms

The system includes robust fallback mechanisms at multiple levels:

1. **File Processing Fallbacks**: Gracefully handles issues in Excel, CSV, and text files
2. **Format Detection Fallbacks**: Tries multiple methods to parse challenging data formats
3. **Retrieval Fallbacks**: Provides domain-specific information when retrieval fails
4. **Document Processing Fallbacks**: Creates simplified documents when structured parsing fails
5. **Empty Result Handling**: Generates informative responses even when no matching documents are found

## Development Notes

### Architecture Decisions

- **Gemini Model**: Used for its speed, cost-efficiency, and "human-like" responses
- **Vector Store**: ChromaDB implementation for efficient document retrieval
- **Monitor**: First-step analysis for better request handling
- **Streaming**: Implemented to improve perceived latency
- **Flask Backend**: Lightweight, easy to deploy, and scalable
- **Adapter Pattern**: Clean separation between API contracts

### Future Improvements

- Add database integration for conversation logging (MongoDB)
- Enhance monitor with more sophisticated classification
- Implement full cross-encoder reranking for even more precise retrieval
- Add user authentication and personalization
- Expand domain-specific handling to more domains beyond employment
- Implement feedback loops to improve retrieval quality over time
- Enhance fallback content with more comprehensive domain-specific information
- Add mechanisms to automatically repair and normalize problematic data

## Integration Guide

For detailed instructions on integrating the backend with the frontend, see [INTEGRATION.md](INTEGRATION.md).

## License

See the LICENSE file for details.