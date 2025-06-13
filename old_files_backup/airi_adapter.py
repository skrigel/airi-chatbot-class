#!/usr/bin/env python
"""
AIRI Chatbot Adapter

This is a complete Flask adapter that connects the GitHub frontend UI with the
robust AI Risk Repository backend components. It provides:
1. Full RAG implementation with hybrid search
2. Domain-specific query handling
3. Clickable citations with source viewer
4. Streaming response format compatibility
"""

from flask import Flask, request, jsonify, Response, stream_with_context, send_from_directory
from flask_cors import CORS
import os
import logging
import json
import time
import re
import hashlib
from pathlib import Path
from dotenv import load_dotenv

# Import all backend components
from vector_store import VectorStore
try:
    from monitor import Monitor as QueryMonitor
except ImportError:
    try:
        from monitor import QueryMonitor
    except ImportError:
        QueryMonitor = None
from gemini_model import GeminiModel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Create Flask app with CORS
app = Flask(__name__)
CORS(app)

# Get directory paths
current_dir = os.path.dirname(os.path.abspath(__file__))
frontend_path = os.path.join(current_dir, 'frontend')
repository_path = os.environ.get('REPOSITORY_PATH', os.path.join(current_dir, 'info_files'))
persist_directory = os.path.join(current_dir, 'chroma_db')

# Create a directory for storing document snippets that can be referenced
snippets_dir = os.path.join(current_dir, 'doc_snippets')
os.makedirs(snippets_dir, exist_ok=True)

# Get API key for models
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', 'AIzaSyDosr02ZzhOptpHts-zFcyZcUxwpVSjszI')

#------------------------------------------------------
# Initialize backend components
#------------------------------------------------------

# 1. Vector store for RAG
vector_store = None
simple_vector_store = None

try:
    if os.path.exists(repository_path):
        logger.info(f"Initializing vector store with repository: {repository_path}")
        
        # Clean up any stale chromadb files that might be corrupted
        if os.path.exists(persist_directory) and not os.listdir(persist_directory):
            logger.warning(f"Found empty persist directory {persist_directory}, removing it")
            try:
                import shutil
                shutil.rmtree(persist_directory)
            except Exception as cleanup_err:
                logger.error(f"Error cleaning up empty directory: {str(cleanup_err)}")
        
        # Initialize the main vector store
        vector_store = VectorStore(
            embedding_provider="google",
            api_key=GEMINI_API_KEY,
            repository_path=repository_path,
            persist_directory=persist_directory,
            use_hybrid_search=True  # Enable hybrid search (vector + keyword)
        )
        
        # Ingest documents if not already done
        if not os.path.exists(persist_directory):
            logger.info("Ingesting documents into vector store...")
            try:
                ingestion_result = vector_store.ingest_documents()
                if ingestion_result:
                    logger.info("Document ingestion completed successfully")
                else:
                    logger.error("Document ingestion failed, vector store may not work correctly")
            except Exception as ingest_err:
                logger.error(f"Error during document ingestion: {str(ingest_err)}")
        
        # Always initialize the simple vector store as a backup
        try:
            from simple_vector_store import SimpleVectorStore
            logger.info("Initializing simple vector store as backup...")
            simple_vector_store = SimpleVectorStore(repository_path)
            # Save to pickle file for future use
            import pickle
            with open(os.path.join(current_dir, 'simple_store.pkl'), 'wb') as f:
                pickle.dump(simple_vector_store, f)
            logger.info("Simple vector store initialized and saved as backup")
        except Exception as simple_err:
            logger.error(f"Error initializing simple vector store: {str(simple_err)}")
        
        logger.info("Vector store initialization complete")
    else:
        logger.warning(f"Repository path {repository_path} not found")
except Exception as e:
    logger.error(f"Error initializing vector store: {str(e)}")
    vector_store = None
    
    # Try to load simple vector store as fallback
    try:
        simple_store_path = os.path.join(current_dir, 'simple_store.pkl')
        if os.path.exists(simple_store_path):
            logger.info("Trying to load simple vector store as fallback...")
            import pickle
            with open(simple_store_path, 'rb') as f:
                simple_vector_store = pickle.load(f)
            logger.info("Loaded simple vector store as fallback")
    except Exception as simple_load_err:
        logger.error(f"Error loading simple vector store: {str(simple_load_err)}")
        
        # Last resort - create a new simple vector store
        try:
            from simple_vector_store import SimpleVectorStore
            logger.info("Creating new simple vector store as last resort...")
            simple_vector_store = SimpleVectorStore(repository_path)
        except Exception as last_err:
            logger.error(f"Could not create simple vector store: {str(last_err)}")
            # No vector store available at all

# 2. Query monitor for classification
query_monitor = None
try:
    if QueryMonitor is not None:
        logger.info("Initializing query monitor")
        query_monitor = QueryMonitor(api_key=GEMINI_API_KEY)
        logger.info("Query monitor initialized successfully")
    else:
        logger.warning("QueryMonitor class not found. Query type detection will be disabled.")
except Exception as e:
    logger.error(f"Error initializing query monitor: {str(e)}")
    query_monitor = None

# 3. LLM model for responses
gemini_model = None
try:
    logger.info("Initializing Gemini model")
    gemini_model = GeminiModel(
        api_key=GEMINI_API_KEY,
        model_name="gemini-2.0-flash"  # Use more capable model
    )
    logger.info("Gemini model initialized successfully")
except Exception as e:
    logger.error(f"Error initializing Gemini model: {str(e)}")
    gemini_model = None

# Conversation history (simple in-memory store)
conversations = {}

#------------------------------------------------------
# Helper functions for document formatting and citations
#------------------------------------------------------

def clean_for_filename(text, max_length=50):
    """Clean a string to make it suitable for use as a filename"""
    # Remove invalid filename characters
    clean = re.sub(r'[\\/*?:"<>|]', '', text)
    # Replace spaces and underscores with hyphens
    clean = re.sub(r'[\s_]+', '-', clean)
    # Truncate to reasonable length
    return clean[:max_length]

def format_excel_citation(doc, include_link=True):
    """Format a citation specifically for Excel files with sheet and row info"""
    citation = ""
    file_name = os.path.basename(doc.metadata.get('source', 'Unknown'))
    sheet_name = doc.metadata.get('sheet', 'Unknown Sheet')
    row = doc.metadata.get('row', None)
    
    # Create a shortened display name
    display_name = clean_for_filename(file_name).replace('-', ' ').title()
    
    # Create unique ID for this document reference
    doc_id = hashlib.md5(f"{file_name}_{sheet_name}_{row}".encode()).hexdigest()[:8]
    
    # Save snippet for reference
    snippet_path = os.path.join(snippets_dir, f"doc_{doc_id}.txt")
    with open(snippet_path, 'w') as f:
        f.write(f"Source: {file_name}\n")
        f.write(f"Sheet: {sheet_name}\n")
        if row is not None:
            f.write(f"Row: {row}\n")
        f.write(f"\nContent:\n{doc.page_content}")
    
    # Create citation
    if row is not None:
        citation = f"[{display_name}: Sheet '{sheet_name}', Row {row}](#/snippet/{doc_id})"
    else:
        citation = f"[{display_name}: Sheet '{sheet_name}']](#/snippet/{doc_id})"
    
    return citation

def format_document_citation(doc):
    """Format a citation for a document based on its type and metadata"""
    if not doc or not hasattr(doc, 'metadata'):
        return "[Unknown Source]"
    
    # Get basic metadata
    file_type = doc.metadata.get('file_type', '')
    source = doc.metadata.get('source', 'Unknown')
    
    # Create a unique ID for this document reference
    doc_id = hashlib.md5(f"{source}_{file_type}".encode()).hexdigest()[:8]
    
    # Save snippet for reference
    snippet_path = os.path.join(snippets_dir, f"doc_{doc_id}.txt")
    try:
        with open(snippet_path, 'w') as f:
            f.write(f"Source: {source}\n")
            for key, value in doc.metadata.items():
                if key not in ['source', 'page_content']:
                    f.write(f"{key}: {value}\n")
            f.write(f"\nContent:\n{doc.page_content}")
    except Exception as e:
        logger.error(f"Error saving document snippet: {str(e)}")
    
    # Create appropriate citation based on file type
    if 'excel' in file_type.lower():
        return format_excel_citation(doc)
    elif doc.metadata.get('citation'):
        # Use existing citation if available but add link
        citation = doc.metadata.get('citation')
        display_name = clean_for_filename(os.path.basename(source)).replace('-', ' ').title()
        return f"[{citation}](#/snippet/{doc_id})"
    else:
        # Create a generic citation with link
        display_name = clean_for_filename(os.path.basename(source)).replace('-', ' ').title()
        return f"[{display_name}](#/snippet/{doc_id})"

def enhance_response_with_citations(response, docs):
    """Add clickable citations to the response text based on patterns"""
    if not docs:
        return response
    
    # Create mapping of docs to their citations
    doc_citations = {}
    for i, doc in enumerate(docs):
        doc_citations[f"SECTION {i+1}"] = format_document_citation(doc)
        doc_citations[f"Source {i+1}"] = format_document_citation(doc)
        doc_citations[f"Document {i+1}"] = format_document_citation(doc)
        doc_citations[f"Entry {i+1}"] = format_document_citation(doc)
        
        # Also add by filename if available
        if 'source' in doc.metadata:
            filename = os.path.basename(doc.metadata['source'])
            doc_citations[filename] = format_document_citation(doc) 
    
    # Replace section references with clickable citations
    for pattern, citation in doc_citations.items():
        if pattern in response:
            response = response.replace(f"[{pattern}]", citation)
            response = response.replace(pattern, citation)
    
    return response

def get_snippet_page(snippet_id):
    """Generate an HTML page for viewing a document snippet"""
    snippet_path = os.path.join(snippets_dir, f"doc_{snippet_id}.txt")
    
    if os.path.exists(snippet_path):
        try:
            with open(snippet_path, 'r') as f:
                content = f.read()
                
            # Simple HTML formatting for the snippet content
            formatted_content = content.replace('\n', '<br>')
            
            html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Document Source</title>
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 0; padding: 20px; max-width: 800px; margin: 0 auto; }}
        h1 {{ color: #333; }}
        .snippet {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; border: 1px solid #ddd; }}
        .metadata {{ font-weight: bold; color: #555; }}
        .content {{ margin-top: 20px; white-space: pre-wrap; }}
        .back-button {{ display: inline-block; margin-top: 20px; padding: 10px 15px; background-color: #4285f4; color: white; text-decoration: none; border-radius: 4px; }}
    </style>
</head>
<body>
    <h1>Document Source Reference</h1>
    <div class="snippet">
        <div class="content">{formatted_content}</div>
    </div>
    <a href="javascript:history.back()" class="back-button">Go Back</a>
</body>
</html>"""
            return html
        except Exception as e:
            return f"Error reading snippet: {str(e)}", 500
    else:
        return "Snippet not found", 404

#------------------------------------------------------
# Query processing functions
#------------------------------------------------------

def process_query(message, conversation_id):
    """Process a query using all backend components (non-streaming)"""
    # 1. Analyze the query with the monitor
    query_type = "general"
    domain = None
    
    if query_monitor:
        try:
            # Use the correct method based on the implementation
            if hasattr(query_monitor, 'analyze_query'):
                query_analysis = query_monitor.analyze_query(message)
                query_type = query_analysis.get('query_type', 'general')
            elif hasattr(query_monitor, 'determine_inquiry_type'):
                inquiry_result = query_monitor.determine_inquiry_type(message)
                query_type = inquiry_result.get('inquiry_type', 'GENERAL').lower()
                domain = inquiry_result.get('primary_domain', 'OTHER').lower()
                
                # Map inquiry types to our internal types
                if query_type == "employment_risk":
                    query_type = "employment"
            
            logger.info(f"Query type detected: {query_type}")
            if domain:
                logger.info(f"Domain detected: {domain}")
        except Exception as e:
            logger.error(f"Error analyzing query: {str(e)}")
    
    # Enhanced detection for employment questions
    employment_keywords = ['job', 'employ', 'work', 'career', 'unemployment', 'labor', 'worker', 'workforce', 'occupation']
    socioeconomic_keywords = ['socioeconomic', 'economic', 'inequality', 'social', 'financial']
    
    if not query_type or query_type == "general":
        message_lower = message.lower()
        if any(keyword in message_lower for keyword in employment_keywords):
            query_type = "employment"
            domain = "socioeconomic"
            logger.info("Enhanced detection found employment related query")
        elif any(keyword in message_lower for keyword in socioeconomic_keywords):
            query_type = "socioeconomic"
            domain = "socioeconomic"
            logger.info("Enhanced detection found socioeconomic related query")
    
    # 2. Get relevant documents using enhanced retrieval strategy
    docs = []
    if vector_store:
        try:
            # For employment/socioeconomic queries, use enhanced search strategy
            if query_type in ["employment", "socioeconomic"]:
                # First, try to get domain-specific documents
                enhanced_query = message
                if query_type == "employment":
                    enhanced_query += " employment job work career labor workforce inequality"
                elif query_type == "socioeconomic":
                    enhanced_query += " socioeconomic economic social inequality"
                
                # Get more documents for employment queries since they're specific
                docs = vector_store.get_relevant_documents(enhanced_query, k=8)
                logger.info(f"Retrieved {len(docs)} documents using enhanced employment search")
                
                # Filter to prioritize employment-related domains
                employment_docs = []
                other_docs = []
                
                for doc in docs:
                    doc_domain = doc.metadata.get('domain', '').lower()
                    doc_subdomain = doc.metadata.get('subdomain', '').lower()
                    doc_specific_domain = doc.metadata.get('specific_domain', '').lower()
                    
                    # Check if this document is employment-related
                    is_employment_related = any(keyword in doc_domain + doc_subdomain + doc_specific_domain 
                                              for keyword in ['employ', 'job', 'work', 'labor', 'socioeconomic', 'economic', 'inequality'])
                    
                    if is_employment_related:
                        employment_docs.append(doc)
                    else:
                        other_docs.append(doc)
                
                # Prioritize employment docs, but include some others for context
                docs = employment_docs[:6] + other_docs[:2]
                logger.info(f"Filtered to {len(employment_docs)} employment-specific documents and {min(2, len(other_docs))} general documents")
            else:
                # Standard retrieval for other queries
                docs = vector_store.get_relevant_documents(message, k=5)
                logger.info(f"Retrieved {len(docs)} relevant documents using standard search")
        except Exception as e:
            logger.error(f"Error retrieving documents from vector store: {str(e)}")
            
            # Use the already loaded simple_vector_store as fallback if available
            if 'simple_vector_store' in globals() and simple_vector_store is not None:
                logger.info("Using already loaded simple vector store as fallback...")
                docs = simple_vector_store.get_relevant_documents(message, k=5)
                logger.info(f"Retrieved {len(docs)} documents from simple vector store")
            else:
                # Try to load from pickle as a second fallback
                try:
                    simple_store_path = os.path.join(current_dir, 'simple_store.pkl')
                    if os.path.exists(simple_store_path):
                        logger.info("Loading simple vector store from pickle file as fallback...")
                        import pickle
                        with open(simple_store_path, 'rb') as f:
                            simple_store = pickle.load(f)
                        
                        docs = simple_store.get_relevant_documents(message, k=5)
                        logger.info(f"Retrieved {len(docs)} documents from simple vector store")
                    else:
                        logger.error("No simple_store.pkl file found for fallback")
                except Exception as simple_err:
                    logger.error(f"Error using simple vector store: {str(simple_err)}")
    
    # 3. Format context from documents
    context = ""
    if docs:
        try:
            # Check if we're using the standard vector store or the simple store
            if 'simple_store' in locals():
                # Use the simple store's formatting method
                context = simple_store.format_context_from_docs(docs)
            else:
                # Use the standard vector store formatting
                # Use special formatting for employment-related queries
                if query_type == "employment" or domain == "socioeconomic":
                    if hasattr(vector_store, 'format_context_for_domain'):
                        context = vector_store.format_context_for_domain(docs, "employment")
                    else:
                        context = vector_store.format_context_from_docs(docs)
                else:
                    context = vector_store.format_context_from_docs(docs)
            
            logger.info(f"Formatted context with {len(context)} characters")
        except Exception as e:
            logger.error(f"Error formatting context: {str(e)}")
            
            # Fallback formatting if all else fails
            try:
                context = "INFORMATION FROM THE AI RISK REPOSITORY:\n\n"
                for i, doc in enumerate(docs, 1):
                    context += f"SECTION {i}:\n{doc.page_content}\n\n"
            except Exception as fallback_err:
                logger.error(f"Even fallback formatting failed: {str(fallback_err)}")
    
    # 4. Generate response using Gemini with retrieved context
    if gemini_model:
        try:
            # Prepare history for the model (if needed)
            history = conversations.get(conversation_id, [])[-5:]  # Last 5 messages
            
            # Convert our history format to the model's expected format if needed
            model_history = []
            for msg in history:
                if msg['role'] == 'user':
                    model_history.append({"role": "user", "parts": [{"text": msg['content']}]})
                elif msg['role'] == 'assistant':
                    model_history.append({"role": "model", "parts": [{"text": msg['content']}]})
            
            # Prepare the prompt with context, enhanced for specific query types
            if context:
                base_prompt = """You are an AI assistant that helps users understand AI risks based on information from the MIT AI Risk Repository. 
Answer based on the retrieved context when possible. If the context doesn't contain relevant information, say so honestly.

When referring to sources, use 'SECTION X' or 'Document X' format which will be replaced with proper citations later."""
                
                if query_type in ["employment", "socioeconomic"]:
                    specific_guidance = """

IMPORTANT: This question is about employment, job, or socioeconomic risks from AI. The repository contains specific information about:
- Increased inequality and decline in employment quality (Domain 6.2)
- Economic and cultural devaluation of human effort (Domain 6.3) 
- Socioeconomic and Environmental risks (Domain 6)

Focus your answer on these specific employment-related risks when available in the context."""
                    prompt = f"""{base_prompt}{specific_guidance}

Context: {context}

Question: {message}"""
                else:
                    prompt = f"""{base_prompt}

Context: {context}

Question: {message}"""
            else:
                if query_type in ["employment", "socioeconomic"]:
                    prompt = f"""You are an AI assistant that helps users understand AI risks based on the MIT AI Risk Repository. 
The repository contains information about employment and socioeconomic risks from AI, including:
- Job displacement and automation impacts
- Increased inequality from AI systems
- Decline in employment quality
- Economic impacts on workers

Answer the following question about AI employment/socioeconomic risks: {message}

When referring to sources, use 'SECTION X' or 'Document X' format which will be replaced with proper citations later."""
                else:
                    prompt = f"""You are an AI assistant that helps users understand AI risks based on the MIT AI Risk Repository. 
Answer the following question about AI risk: {message}

When referring to sources, use 'SECTION X' or 'Document X' format which will be replaced with proper citations later."""
            
            # Use the correct method based on implementation
            if hasattr(gemini_model, 'generate'):
                response = gemini_model.generate(prompt, model_history)
            else:
                response = gemini_model.generate_response(prompt)
            
            return response, docs
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"I encountered an error while generating a response: {str(e)}", docs
    else:
        # Fallback if Gemini model is not available - use raw document content
        if docs:
            # More detailed response with multiple documents
            response = f"I found {len(docs)} relevant documents in the AI Risk Repository.\n\n"
            
            for i, doc in enumerate(docs[:3]):  # Show up to 3 documents
                # Format the document content
                response += f"Document {i+1}:\n{doc.page_content[:800]}...\n\n"
                
                if i < len(docs[:3]) - 1:
                    response += "\n\n"
        else:
            response = "I'm sorry, but I couldn't find specific information in the AI Risk Repository for your query. The repository covers risks related to discrimination, privacy, misinformation, malicious use, human-computer interaction, socioeconomic impacts, and system safety."
        
        return response, docs

#------------------------------------------------------
# API Routes for the frontend
#------------------------------------------------------

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    components_status = {
        "vector_store": "ok" if vector_store else "disabled",
        "query_monitor": "ok" if query_monitor else "disabled",
        "gemini_model": "ok" if gemini_model else "disabled"
    }
    
    if "disabled" in components_status.values():
        overall_status = "degraded"
    else:
        overall_status = "ok"
    
    return jsonify({
        "status": overall_status,
        "components": components_status
    })

@app.route('/api/snippet/<doc_id>', methods=['GET'])
def get_snippet(doc_id):
    """Retrieve a document snippet by its ID"""
    snippet_path = os.path.join(snippets_dir, f"doc_{doc_id}.txt")
    
    if os.path.exists(snippet_path):
        try:
            with open(snippet_path, 'r') as f:
                content = f.read()
            return jsonify({
                "content": content
            })
        except Exception as e:
            return jsonify({"error": f"Error reading snippet: {str(e)}"}), 500
    else:
        return jsonify({"error": "Snippet not found"}), 404

@app.route('/api/v1/reset', methods=['POST'])
def reset():
    """Reset conversation endpoint - required by frontend"""
    # Get conversation ID from request
    data = request.json or {}
    conversation_id = data.get('conversationId', 'default')
    
    # Reset the conversation history
    if conversation_id in conversations:
        conversations[conversation_id] = []
    
    # Also reset the Gemini model if available
    if gemini_model and hasattr(gemini_model, 'reset_conversation'):
        gemini_model.reset_conversation()
    
    return jsonify({
        "success": True, 
        "message": "Conversation reset successfully"
    })

@app.route('/api/v1/sendMessage', methods=['POST'])
def send_message():
    """Non-streaming chat endpoint"""
    data = request.json
    
    # Log the incoming request
    logger.info(f"Received message: {data}")
    message = data.get('message', '')
    conversation_id = data.get('conversationId', 'default')
    
    # Initialize conversation history if it doesn't exist
    if conversation_id not in conversations:
        conversations[conversation_id] = []
    
    # Add user message to history
    conversations[conversation_id].append({"role": "user", "content": message})
    
    # Full processing with all backend components
    response_text, docs = process_query(message, conversation_id)
    
    # Enhance the response with clickable citations
    enhanced_response = enhance_response_with_citations(response_text, docs)
    
    # Add response to conversation history
    conversations[conversation_id].append({"role": "assistant", "content": enhanced_response})
    
    return jsonify({
        "id": conversation_id,
        "response": enhanced_response,
        "status": "complete"
    })

@app.route('/api/v1/stream', methods=['POST'])
def stream():
    """Streaming chat endpoint"""
    data = request.json
    
    # Log the incoming request
    logger.info(f"Received stream request: {data}")
    message = data.get('message', '')
    conversation_id = data.get('conversationId', 'default')
    
    # Initialize conversation history if it doesn't exist
    if conversation_id not in conversations:
        conversations[conversation_id] = []
    
    # Add user message to history
    conversations[conversation_id].append({"role": "user", "content": message})
    
    # For this request, we'll track docs to add citations later
    request_docs = []
    
    # Define a streaming response generator
    def generate():
        # Initialize variables
        query_type = "general"
        domain = None
        docs = []
        
        # Initial status update
        yield json.dumps("Processing your query...") + '\n'
        time.sleep(0.3)
        
        # 1. Analyze the query with the monitor
        if query_monitor:
            try:
                yield json.dumps("Analyzing your question...") + '\n'
                
                # Use the correct method based on the implementation
                if hasattr(query_monitor, 'analyze_query'):
                    query_analysis = query_monitor.analyze_query(message)
                    query_type = query_analysis.get('query_type', 'general')
                elif hasattr(query_monitor, 'determine_inquiry_type'):
                    inquiry_result = query_monitor.determine_inquiry_type(message)
                    query_type = inquiry_result.get('inquiry_type', 'GENERAL').lower()
                    domain = inquiry_result.get('primary_domain', 'OTHER').lower()
                    
                    # Map inquiry types to our internal types
                    if query_type == "employment_risk":
                        query_type = "employment"
                
                logger.info(f"Query type detected: {query_type}")
                if domain:
                    logger.info(f"Domain detected: {domain}")
            except Exception as e:
                logger.error(f"Error analyzing query: {str(e)}")
        
        # Enhanced detection for employment questions (same as non-streaming)
        employment_keywords = ['job', 'employ', 'work', 'career', 'unemployment', 'labor', 'worker', 'workforce', 'occupation']
        socioeconomic_keywords = ['socioeconomic', 'economic', 'inequality', 'social', 'financial']
        
        if not query_type or query_type == "general":
            message_lower = message.lower()
            if any(keyword in message_lower for keyword in employment_keywords):
                query_type = "employment"
                domain = "socioeconomic"
                logger.info("Enhanced detection found employment related query")
            elif any(keyword in message_lower for keyword in socioeconomic_keywords):
                query_type = "socioeconomic"
                domain = "socioeconomic"
                logger.info("Enhanced detection found socioeconomic related query")
        
        if vector_store:
            try:
                yield json.dumps("Searching repository for relevant information...") + '\n'
                
                # Enhanced search for employment/socioeconomic queries (same as non-streaming)
                if query_type in ["employment", "socioeconomic"]:
                    enhanced_query = message
                    if query_type == "employment":
                        enhanced_query += " employment job work career labor workforce inequality"
                    elif query_type == "socioeconomic":
                        enhanced_query += " socioeconomic economic social inequality"
                    
                    docs = vector_store.get_relevant_documents(enhanced_query, k=8)
                    logger.info(f"Retrieved {len(docs)} documents using enhanced employment search")
                    
                    # Filter to prioritize employment-related domains
                    employment_docs = []
                    other_docs = []
                    
                    for doc in docs:
                        doc_domain = doc.metadata.get('domain', '').lower()
                        doc_subdomain = doc.metadata.get('subdomain', '').lower()
                        doc_specific_domain = doc.metadata.get('specific_domain', '').lower()
                        
                        is_employment_related = any(keyword in doc_domain + doc_subdomain + doc_specific_domain 
                                                  for keyword in ['employ', 'job', 'work', 'labor', 'socioeconomic', 'economic', 'inequality'])
                        
                        if is_employment_related:
                            employment_docs.append(doc)
                        else:
                            other_docs.append(doc)
                    
                    docs = employment_docs[:6] + other_docs[:2]
                    logger.info(f"Filtered to {len(employment_docs)} employment-specific documents and {min(2, len(other_docs))} general documents")
                else:
                    docs = vector_store.get_relevant_documents(message, k=5)
                    logger.info(f"Retrieved {len(docs)} relevant documents using standard search")
                
                # Store in request_docs for later citation enhancement
                request_docs = docs.copy()
                
                if docs:
                    if query_type in ["employment", "socioeconomic"]:
                        employment_count = len([doc for doc in docs if any(keyword in doc.metadata.get('domain', '').lower() + doc.metadata.get('subdomain', '').lower() + doc.metadata.get('specific_domain', '').lower() 
                                                                          for keyword in ['employ', 'job', 'work', 'labor', 'socioeconomic', 'economic', 'inequality'])])
                        if employment_count > 0:
                            yield json.dumps(f"Found {employment_count} employment-specific documents and {len(docs) - employment_count} additional relevant documents.") + '\n'
                        else:
                            yield json.dumps(f"Found {len(docs)} relevant documents in the repository.") + '\n'
                    else:
                        yield json.dumps(f"Found {len(docs)} relevant documents in the repository.") + '\n'
                else:
                    yield json.dumps("No specific documents found. Using general knowledge.") + '\n'
            except Exception as e:
                logger.error(f"Error retrieving documents from vector store: {str(e)}")
                
                # Try to use the simple vector store as fallback
                yield json.dumps("Trying alternative search method...") + '\n'
                
                # First try to use the already loaded simple_vector_store if available
                if 'simple_vector_store' in globals() and simple_vector_store is not None:
                    try:
                        logger.info("Using already loaded simple vector store as fallback...")
                        docs = simple_vector_store.get_relevant_documents(message, k=5)
                        # Store in request_docs for later citation enhancement
                        request_docs = docs.copy()
                        
                        logger.info(f"Retrieved {len(docs)} documents from simple vector store")
                        
                        if docs:
                            yield json.dumps(f"Found {len(docs)} relevant documents using alternative search.") + '\n'
                        else:
                            yield json.dumps("No specific documents found. Using general knowledge.") + '\n'
                    except Exception as loaded_err:
                        logger.error(f"Error using loaded simple vector store: {str(loaded_err)}")
                        # Continue to pickle fallback
                
                # Try to load from pickle as a second fallback
                if not docs:
                    try:
                        simple_store_path = os.path.join(current_dir, 'simple_store.pkl')
                        if os.path.exists(simple_store_path):
                            logger.info("Loading simple vector store from pickle file as fallback...")
                            import pickle
                            with open(simple_store_path, 'rb') as f:
                                simple_store = pickle.load(f)
                            
                            docs = simple_store.get_relevant_documents(message, k=5)
                            # Store in request_docs for later citation enhancement
                            request_docs = docs.copy()
                            
                            logger.info(f"Retrieved {len(docs)} documents from simple vector store")
                            
                            if docs:
                                yield json.dumps(f"Found {len(docs)} relevant documents using alternative search.") + '\n'
                            else:
                                yield json.dumps("No specific documents found. Using general knowledge.") + '\n'
                        else:
                            logger.error("No simple_store.pkl file found for fallback")
                            yield json.dumps("Search failed. Using general knowledge.") + '\n'
                    except Exception as simple_err:
                        logger.error(f"Error using simple vector store: {str(simple_err)}")
                        yield json.dumps("Search failed. Using general knowledge.") + '\n'
        
        # 3. Format context from documents
        context = ""
        if docs:
            try:
                # Check if we're using the standard vector store or the simple store
                if 'simple_store' in locals():
                    # Use the simple store's formatting method
                    context = simple_store.format_context_from_docs(docs)
                else:
                    # Use the standard vector store formatting
                    # Use special formatting for employment-related queries
                    if query_type == "employment" or domain == "socioeconomic":
                        if hasattr(vector_store, 'format_context_for_domain'):
                            context = vector_store.format_context_for_domain(docs, "employment")
                        else:
                            context = vector_store.format_context_from_docs(docs)
                    else:
                        context = vector_store.format_context_from_docs(docs)
                
                logger.info(f"Formatted context with {len(context)} characters")
            except Exception as e:
                logger.error(f"Error formatting context: {str(e)}")
                
                # Fallback formatting if all else fails
                try:
                    context = "INFORMATION FROM THE AI RISK REPOSITORY:\n\n"
                    for i, doc in enumerate(docs, 1):
                        context += f"SECTION {i}:\n{doc.page_content}\n\n"
                except Exception as fallback_err:
                    logger.error(f"Even fallback formatting failed: {str(fallback_err)}")
        
        # 4. Generate response using Gemini with retrieved context
        complete_response = ""
        
        if gemini_model:
            try:
                yield json.dumps("Generating response...") + '\n'
                
                # Prepare history for the model (if needed)
                history = conversations.get(conversation_id, [])[-5:]  # Last 5 messages
                
                # Convert our history format to the model's expected format if needed
                model_history = []
                for msg in history:
                    if msg['role'] == 'user':
                        model_history.append({"role": "user", "parts": [{"text": msg['content']}]})
                    elif msg['role'] == 'assistant':
                        model_history.append({"role": "model", "parts": [{"text": msg['content']}]})
                
                # Prepare the prompt with context
                if context:
                    prompt = f"""You are an AI assistant that helps users understand AI risks based on information from the MIT AI Risk Repository. 
Answer based on the retrieved context when possible. If the context doesn't contain relevant information, say so honestly.

When referring to sources, use 'SECTION X' or 'Document X' format which will be replaced with proper citations later.

Context: {context}

Question: {message}"""
                else:
                    prompt = f"""You are an AI assistant that helps users understand AI risks based on the MIT AI Risk Repository. 
Answer the following question about AI risk: {message}

When referring to sources, use 'SECTION X' or 'Document X' format which will be replaced with proper citations later."""
                
                # Use the correct streaming method based on implementation
                if hasattr(gemini_model, 'generate_stream'):
                    # Use generate_stream if available
                    for chunk in gemini_model.generate_stream(prompt, model_history):
                        yield json.dumps(chunk) + '\n'
                        complete_response += chunk
                        time.sleep(0.05)
                elif hasattr(gemini_model, 'generate_response'):
                    # Use generate_response with stream=True
                    response_generator = gemini_model.generate_response(prompt, stream=True)
                    
                    # Check if it returned a string (non-streaming response) or an iterable (streaming)
                    if isinstance(response_generator, str):
                        # Single response returned
                        yield json.dumps(response_generator) + '\n'
                        complete_response = response_generator
                    else:
                        try:
                            # Try to iterate through the response
                            for chunk in response_generator:
                                text_chunk = chunk if isinstance(chunk, str) else chunk.text if hasattr(chunk, 'text') else str(chunk)
                                yield json.dumps(text_chunk) + '\n'
                                complete_response += text_chunk
                                time.sleep(0.05)
                        except Exception as stream_err:
                            logger.error(f"Error streaming response: {str(stream_err)}")
                            # Fall back to non-streaming response
                            response = gemini_model.generate_response(prompt, stream=False)
                            yield json.dumps(response) + '\n'
                            complete_response = response
                else:
                    # No streaming method available, use regular generate
                    response = gemini_model.generate_response(prompt)
                    
                    # Break up the response into chunks for smoother display
                    words = response.split()
                    for i in range(0, len(words), 5):  # Send 5 words at a time
                        chunk = ' '.join(words[i:i+5]) + ' '
                        yield json.dumps(chunk) + '\n'
                        time.sleep(0.1)
                    
                    complete_response = response
                
                # If no content was generated, provide a fallback
                if not complete_response:
                    fallback = "I'm sorry, I couldn't generate a detailed response. The AI Risk Repository covers various risk domains including discrimination, privacy, misinformation, malicious use, human-computer interaction, socioeconomic impacts, and system safety."
                    yield json.dumps(fallback) + '\n'
                    complete_response = fallback
            except Exception as e:
                logger.error(f"Error generating response: {str(e)}")
                error_message = f"I encountered an error while generating a response: {str(e)}"
                yield json.dumps(error_message) + '\n'
                complete_response = error_message
        else:
            # Fallback if Gemini model is not available - use raw document content
            if docs:
                # More detailed response with multiple documents
                yield json.dumps(f"I found {len(docs)} relevant documents in the AI Risk Repository.") + '\n'
                time.sleep(0.3)
                
                for i, doc in enumerate(docs[:3]):  # Show up to 3 documents
                    # Format the document content
                    citation = format_document_citation(doc)
                    content = f"\nDocument {i+1}:\n{doc.page_content[:800]}..." + f"\n\n{citation}"
                    
                    # Split content into chunks for a more natural streaming experience
                    words = content.split()
                    for j in range(0, len(words), 5):  # Send 5 words at a time
                        chunk = ' '.join(words[j:j+5]) + ' '
                        yield json.dumps(chunk) + '\n'
                        time.sleep(0.05)
                    
                    if i < len(docs[:3]) - 1:
                        yield json.dumps("\n\n") + '\n'
                        time.sleep(0.3)
                    
                    # Add to complete response
                    complete_response += content
            else:
                fallback = "I'm sorry, but I couldn't find specific information in the AI Risk Repository for your query. The repository covers risks related to discrimination, privacy, misinformation, malicious use, human-computer interaction, socioeconomic impacts, and system safety."
                yield json.dumps(fallback) + '\n'
                complete_response = fallback
        
        # Enhance the response with clickable citations before saving to history
        enhanced_response = enhance_response_with_citations(complete_response, request_docs)
        
        # Add the complete response to conversation history
        conversations[conversation_id].append({"role": "assistant", "content": enhanced_response})
    
    # Set proper headers for SSE
    return Response(
        stream_with_context(generate()), 
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no',
            'Content-Type': 'text/event-stream'
        }
    )

# Route to manually trigger document ingestion
@app.route('/api/ingest', methods=['POST'])
def ingest_documents():
    """Manually trigger document ingestion"""
    if not vector_store:
        return jsonify({'error': 'Vector store not initialized'}), 500
    
    success = vector_store.ingest_documents()
    
    if success:
        return jsonify({'status': 'Documents ingested successfully'})
    else:
        return jsonify({'error': 'Failed to ingest documents'}), 500

#------------------------------------------------------
# Routes for serving the frontend files
#------------------------------------------------------

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_frontend(path):
    """Serve the React app (GitHub frontend)"""
    logger.info(f"Serving path: {path}")
    
    # Special route for source snippets
    if path.startswith('snippet/'):
        snippet_id = path.split('/')[-1]
        return get_snippet_page(snippet_id)
    
    try:
        if path and os.path.exists(os.path.join(frontend_path, path)):
            logger.info(f"Serving file: {path}")
            return send_from_directory(frontend_path, path)
        elif path and '.' in path:  # File with extension that doesn't exist
            logger.warning(f"File not found: {path}")
            return f"File not found: {path}", 404
        else:
            # Try to serve index.html for all other routes (SPA routing)
            logger.info(f"Serving index.html for path: {path}")
            if os.path.exists(os.path.join(frontend_path, 'index.html')):
                return send_from_directory(frontend_path, 'index.html')
            else:
                return "Frontend not found. Please run setup.sh first to build the frontend.", 404
    except Exception as e:
        logger.error(f"Error serving file: {str(e)}")
        return f"Error: {str(e)}", 500

#------------------------------------------------------
# Main entry point
#------------------------------------------------------

if __name__ == '__main__':
    # Get port from environment variables or use default port 8090
    port = int(os.environ.get('PORT', 8090))
    
    # Function to check if port is available
    def is_port_available(port):
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        available = False
        try:
            sock.bind(('0.0.0.0', port))
            available = True
        except:
            pass
        finally:
            sock.close()
        return available
    
    # Try to use specified port, fall back to alternatives if needed
    if not is_port_available(port):
        print(f"Warning: Port {port} is already in use")
        
        # Try alternative ports
        for alt_port in [8090, 8080, 8000, 3000]:
            if is_port_available(alt_port):
                print(f"Using alternative port: {alt_port}")
                port = alt_port
                break
        
        if not is_port_available(port):
            print(f"Warning: Port {port} is still unavailable. The server may fail to start.")
    
    print("\n=========================================")
    print(" AIRI CHATBOT ADAPTER")
    print("=========================================")
    print(f"Starting server at http://localhost:{port}")
    print(f"Serving frontend from: {frontend_path}")
    print(f"Using repository path: {repository_path}")
    print("\nBackend components status:")
    print(f"- Vector Store: {'Active' if vector_store else 'Disabled'}")
    print(f"- Query Monitor: {'Active' if query_monitor else 'Disabled'}")
    print(f"- Gemini Model: {'Active' if gemini_model else 'Disabled'}")
    print("=========================================\n")
    
    # Run the Flask application
    app.run(host='0.0.0.0', port=port, debug=True)