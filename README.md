# AI Risk Repository Chatbot

A RAG (Retrieval-Augmented Generation) chatbot for the MIT AI Risk Repository that helps users navigate and understand AI risks.

## Features

- Advanced RAG capabilities with hybrid retrieval (vector + keyword search)
- Conversational interface to explore the AI Risk Repository
- Advanced query preprocessing for better search results
- Context formatting with source citations
- Domain-specific handling (especially for employment questions)
- Performance optimizations with caching
- Web-based chat interface with streaming responses
- Flask backend with REST API
- Gemini model integration
- Monitor component for question classification
- Status updates during processing for improved user experience
- Robust fallback mechanisms for resilient operation
- Intelligent handling of problematic data formats
- Always-available domain-specific knowledge even when retrieval fails

## System Architecture

### Components

1. **Model Interface**: `gemini_model.py` handles interactions with the Gemini API
2. **Monitor**: `monitor.py` analyzes user inquiries to determine type and appropriateness
3. **Vector Store**: `vector_store.py` for efficient document retrieval and relevance ranking
4. **Backend API**: Flask-based REST API in `app.py` with streaming support
5. **Frontend**: Simple HTML/CSS/JS interface in the `static/` directory with SSE support

## Setup Instructions

### Prerequisites

- Python 3.8+
- Flask
- Google Generative AI Python SDK
- Valid Gemini API key

### Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd airi-chatbot-class
   ```

2. Set up a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Configure environment variables:
   ```
   cp .env.example .env
   ```
   Edit the `.env` file and add your Gemini API key.

5. Prepare the MIT AI Risk Repository data:
   
   Ensure the AI Risk Repository Excel file is in the `info_files` directory. The application automatically detects and processes any Excel files in this directory that contain "risk" or "repository" in their filename.

### Running the Application

1. Start the Flask server:
   ```
   ./run.sh
   ```
   or
   ```
   python app.py
   ```

2. Access the web interface:
   Open your browser and navigate to `http://localhost:5000`

## API Documentation

### Endpoints

- `POST /api/chat`: Send a message to the chatbot
  - Request body: `{"message": "Your question here", "conversation_id": "optional-id", "stream": false}`
  - Response (non-streaming): `{"response": "Bot's response", "conversation_id": "conversation-id"}`
  - Response (streaming with SSE): Series of events with status updates and final response

- `POST /api/reset`: Reset the conversation history
  - Request body: (empty)
  - Response: `{"status": "Conversation reset successfully"}`

- `GET /api/health`: Health check endpoint
  - Response: `{"status": "ok", "components": {"model": "ok", "monitor": "ok", "vector_store": "ok"}, "model_name": "gemini-2.0-flash"}`

## Advanced RAG Implementation

The system uses a sophisticated Retrieval-Augmented Generation (RAG) approach:

1. **Hybrid Retrieval**: Combines vector search (semantic similarity) with keyword search (BM25) for more comprehensive results
2. **Query Preprocessing**: Enhances queries with key terms and special handling for questions
3. **Smart Context Formatting**: Organizes retrieved information with clear structure and source citations
4. **Domain-Specific Handling**: Special processing for different types of questions (e.g., employment-related queries)
5. **Performance Optimization**: Caching mechanism for frequently asked questions
6. **Robust Fallback Mechanisms**: Multi-level fallbacks for reliable operation even with problematic data

### How it works:

1. Documents from the info_files directory are loaded, chunked, and processed for both vector and keyword search
2. User queries are enhanced through preprocessing to improve retrieval quality
3. The hybrid retriever combines results from both search methods, ranking by relevance
4. Retrieved context is formatted with citations and structured presentation
5. The model's response is augmented with this knowledge, including source attributions
6. When data processing or retrieval fails, intelligent fallbacks provide relevant information

This advanced approach significantly improves the chatbot's ability to provide accurate, relevant, and trustworthy information about the AI Risk Repository.

### Fallback Mechanisms

The system includes robust fallback mechanisms at multiple levels:

1. **File Processing Fallbacks**: Gracefully handles issues in Excel, CSV, and text files
2. **Format Detection Fallbacks**: Tries multiple methods to parse challenging data formats
3. **Retrieval Fallbacks**: Provides domain-specific information when retrieval fails
4. **Document Processing Fallbacks**: Creates simplified documents when structured parsing fails
5. **Empty Result Handling**: Generates informative responses even when no matching documents are found

These fallbacks ensure the system is resilient against data processing errors and always provides helpful information to users, regardless of underlying data quality issues.

## Monitor Component

The monitor serves as the first step in the processing pipeline:

1. Analyzes user inquiries to determine their type (General, Specific Risk, Recommendation, etc.)
2. Checks for override attempts or inappropriate requests
3. This classification determines how the system processes the request
4. Enhances security and improves response relevance

## Streaming Responses

To improve perceived latency and user experience:

1. Implements Server-Sent Events (SSE) for streaming responses
2. Provides intermediate status updates during processing (e.g., "Determining inquiry type...")
3. Fallback to traditional request/response for browsers without SSE support
4. Progressive updates make the system feel more responsive even when processing takes time

## Development Notes

### Architecture Decisions

- **Gemini Model**: Used for its speed, cost-efficiency, and "human-like" responses
- **Vector Store**: ChromaDB implementation for efficient document retrieval
- **Monitor**: First-step analysis for better request handling
- **Streaming**: Implemented to improve perceived latency
- **Flask Backend**: Lightweight, easy to deploy, and scalable

### Cost Considerations

- Gemini-2.0-Flash: Lower cost option with reasonable performance
- Estimated costs: ~$20/month for 1000 users/week (based on meeting notes)

### Future Improvements

- Add database integration for conversation logging (MongoDB)
- Enhance monitor with more sophisticated classification
- Implement full cross-encoder reranking for even more precise retrieval
- Add user authentication and personalization
- Expand domain-specific handling to more domains beyond employment
- Implement feedback loops to improve retrieval quality over time
- Enhance fallback content with more comprehensive domain-specific information
- Add mechanisms to automatically repair and normalize problematic data
- Implement structured data extraction and knowledge graph integration
- Develop adaptive fallback that improves based on user interactions

## License

See the LICENSE file for details.