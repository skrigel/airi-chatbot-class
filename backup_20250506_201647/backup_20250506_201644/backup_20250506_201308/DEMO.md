# Demo Script for AI Risk Repository Chatbot

This script outlines a demonstration of the enhanced RAG capabilities in the AI Risk Repository Chatbot.

## Setup

1. Ensure all dependencies are installed:
   ```bash
   # Start in virtual environment
   python -m venv venv
   source venv/bin/activate
   
   # Install dependencies
   pip install -r requirements.txt
   ```

2. Ensure the AI Risk Repository Excel file is in the info_files directory:
   
   The application looks for any Excel files in the info_files directory that contain "risk" or "repository" in their filename. Make sure at least one such file is present.

3. Start the application:
   ```bash
   ./run.sh
   ```
   
   This will:
   - Clear any existing vector database
   - Process any Excel files found in info_files
   - Build the vector database with the full MIT AI Risk Repository data
   - Start the web server

4. Access the web interface at the URL shown in the terminal (e.g., http://localhost:8080)

## Demo Flow

### 1. Excel Integration & Comprehensive Repository (2-3 min)

**Demonstrate:**
- Show the setup process finding and using the Excel file
- Point out the log messages showing Excel processing

**Talking Points:**
- "We've added support for the complete MIT AI Risk Repository Excel file"
- "The system processes each entry in the database as a separate document"
- "Over 1600 risk entries from 65 frameworks are now searchable"

### 2. Hybrid Search Capability (3-4 min)

**Query examples:**
- "What are risks of AI in job markets?"
- "How might artificial intelligence affect employment?"

**Talking Points:**
- "Our hybrid retriever combines semantic search with keyword matching"
- "The system retrieves information even without exact term matches"
- "Results are combined from both search methods and ranked by relevance"

### 3. Context Formatting & Citations (3-4 min)

**Query examples:**
- "What are potential biases in AI systems?"
- "How do AI systems impact privacy?"

**Talking Points:**
- "Context is now formatted with clear sections and citations"
- "Citations increase trust and provide attribution to source material"
- "Structure makes it easier for both the model and users to understand the information"

### 4. Domain-Specific Handling (3-4 min)

**Query examples:**
- "Will AI replace human workers?"
- "How might AI increase economic inequality?"

**Talking Points:**
- "We've implemented specialized handling for employment-related queries"
- "The system prioritizes employment-specific content when relevant"
- "Different prompt templates are used for different types of questions"

### 5. Performance Optimizations (2-3 min)

**Demonstrate by:**
- Asking the same question twice to show caching

**Talking Points:**
- "Results are cached to improve response times for repeated queries"
- "The log message shows when cached results are used"
- "Queries are preprocessed to improve retrieval accuracy"

## Technical Implementation Highlights

Key components to mention:
1. `HybridRetriever` class in vector_store.py - combines vector and keyword search
2. Query preprocessing - enhances queries with key terms
3. Context formatting with source attribution
4. Domain-specific formatting
5. Caching mechanism

## Conclusion

Summarize the improvements:
- Better retrieval quality through hybrid search
- Improved user experience with source attribution
- More relevant responses for domain-specific questions
- Better performance through caching