"""
Chat service that orchestrates the entire conversation flow.
"""
from typing import List, Dict, Any, Tuple, Optional
from langchain.docstore.document import Document

from ..models.gemini import GeminiModel
from ..storage.vector_store import VectorStore
from ..query.processor import QueryProcessor
from .citation_service import CitationService
from ...config.logging import get_logger
from ...config.settings import settings

logger = get_logger(__name__)

class ChatService:
    """Main service for handling chat interactions."""
    
    def __init__(self, 
                 gemini_model: Optional[GeminiModel] = None,
                 vector_store: Optional[VectorStore] = None,
                 query_monitor: Optional[Any] = None):
        """
        Initialize the chat service.
        
        Args:
            gemini_model: Gemini model instance
            vector_store: Vector store instance
            query_monitor: Query monitor for advanced analysis
        """
        self.gemini_model = gemini_model
        self.vector_store = vector_store
        self.query_processor = QueryProcessor(query_monitor)
        self.citation_service = CitationService()
        
        # Conversation storage (in production, use a proper database)
        self.conversations: Dict[str, List[Dict[str, str]]] = {}
    
    def process_query(self, message: str, conversation_id: str) -> Tuple[str, List[Document]]:
        """
        Process a user query and generate a response.
        
        Args:
            message: User message
            conversation_id: Conversation identifier
            
        Returns:
            Tuple of (response_text, retrieved_documents)
        """
        try:
            # 1. Analyze the query
            query_type, domain = self.query_processor.analyze_query(message)
            
            # 2. Retrieve relevant documents
            docs = self._retrieve_documents(message, query_type)
            
            # 3. Format context
            context = self._format_context(docs, query_type)
            
            # 4. Generate response
            response = self._generate_response(message, query_type, context, conversation_id)
            
            # 5. Enhance with citations
            enhanced_response = self.citation_service.enhance_response_with_citations(response, docs)
            
            # 6. Update conversation history
            self._update_conversation_history(conversation_id, message, enhanced_response)
            
            return enhanced_response, docs
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            error_response = f"I encountered an error while processing your question: {str(e)}"
            return error_response, []
    
    def _retrieve_documents(self, message: str, query_type: str) -> List[Document]:
        """Retrieve relevant documents based on query type."""
        if not self.vector_store:
            logger.warning("No vector store available")
            return []
        
        try:
            if query_type in ["employment", "socioeconomic"]:
                # Enhanced search for employment/socioeconomic queries
                enhanced_query = self.query_processor.enhance_query(message, query_type)
                docs = self.vector_store.get_relevant_documents(enhanced_query, k=settings.EMPLOYMENT_DOCS_RETRIEVED)
                
                # Filter and prioritize employment-related documents
                docs = self.query_processor.filter_documents_by_relevance(docs, query_type)
                
                logger.info(f"Retrieved {len(docs)} documents using enhanced employment search")
            else:
                # Standard retrieval for other queries
                docs = self.vector_store.get_relevant_documents(message, k=settings.DEFAULT_DOCS_RETRIEVED)
                logger.info(f"Retrieved {len(docs)} documents using standard search")
            
            return docs
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            return []
    
    def _format_context(self, docs: List[Document], query_type: str) -> str:
        """Format retrieved documents into context string."""
        if not docs:
            return ""
        
        try:
            if self.vector_store:
                return self.vector_store.format_context_from_docs(docs)
            else:
                # Fallback formatting
                context = "INFORMATION FROM THE AI RISK REPOSITORY:\\n\\n"
                for i, doc in enumerate(docs, 1):
                    context += f"SECTION {i}:\\n{doc.page_content}\\n\\n"
                return context
                
        except Exception as e:
            logger.error(f"Error formatting context: {str(e)}")
            return ""
    
    def _generate_response(self, message: str, query_type: str, context: str, conversation_id: str) -> str:
        """Generate response using the AI model."""
        if not self.gemini_model:
            logger.warning("No Gemini model available")
            return self._create_fallback_response(context, message)
        
        try:
            # Prepare conversation history
            history = self._get_conversation_history(conversation_id)
            
            # Generate enhanced prompt
            prompt = self.query_processor.generate_prompt(message, query_type, context)
            
            # Generate response
            response = self.gemini_model.generate(prompt, history)
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"I encountered an error while generating a response: {str(e)}"
    
    def _create_fallback_response(self, context: str, message: str) -> str:
        """Create a fallback response when AI model is not available."""
        if context:
            return f"Based on the AI Risk Repository, here's what I found:\\n\\n{context[:1000]}..."
        else:
            return "I'm sorry, but I couldn't find specific information in the AI Risk Repository for your query. The repository covers risks related to discrimination, privacy, misinformation, malicious use, human-computer interaction, socioeconomic impacts, and system safety."
    
    def _get_conversation_history(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Get conversation history for the model."""
        if conversation_id not in self.conversations:
            return []
        
        # Get last N messages and convert to model format
        history = self.conversations[conversation_id][-settings.MAX_CONVERSATION_HISTORY:]
        model_history = []
        
        for msg in history:
            if msg['role'] == 'user':
                model_history.append({"role": "user", "parts": [{"text": msg['content']}]})
            elif msg['role'] == 'assistant':
                model_history.append({"role": "model", "parts": [{"text": msg['content']}]})
        
        return model_history
    
    def _update_conversation_history(self, conversation_id: str, message: str, response: str) -> None:
        """Update conversation history."""
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = []
        
        # Add user message and assistant response
        self.conversations[conversation_id].extend([
            {"role": "user", "content": message},
            {"role": "assistant", "content": response}
        ])
        
        # Keep only recent messages to avoid memory issues
        max_messages = settings.MAX_CONVERSATION_HISTORY * 2  # *2 for user+assistant pairs
        if len(self.conversations[conversation_id]) > max_messages:
            self.conversations[conversation_id] = self.conversations[conversation_id][-max_messages:]
    
    def reset_conversation(self, conversation_id: str) -> None:
        """Reset conversation history."""
        if conversation_id in self.conversations:
            self.conversations[conversation_id] = []
        
        # Also reset the model if it supports it
        if self.gemini_model and hasattr(self.gemini_model, 'reset_conversation'):
            self.gemini_model.reset_conversation()