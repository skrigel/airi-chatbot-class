# AI Risk Repository Chatbot

A RAG (Retrieval-Augmented Generation) chatbot for the MIT AI Risk Repository that helps users navigate and understand AI risks.

## Features

- Conversational interface to explore the AI Risk Repository
- Retrieval-augmented generation using vector database
- Web-based chat interface with streaming responses
- Flask backend with REST API
- Gemini model integration
- Monitor component for question classification
- Status updates during processing for improved user experience

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

## Vector Store for RAG

The system uses a vector store (ChromaDB) to enhance the RAG capabilities:

1. Documents from the info_files directory are loaded, chunked, and embedded
2. User queries are analyzed against the vector store to find relevant context
3. Retrieved information is provided to the model along with the user's query
4. The model's response is augmented with knowledge from the repository

This approach significantly improves the chatbot's ability to provide accurate and relevant information about the AI Risk Repository.

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
- Implement more advanced RAG techniques (query reformulation, etc.)
- Add user authentication

## License

See the LICENSE file for details.