"""
Chat service that orchestrates the entire conversation flow.
"""
from typing import List, Dict, Any, Tuple, Optional
from langchain.docstore.document import Document

from ..models.gemini import GeminiModel
from ..storage.vector_store import VectorStore
from ..query.processor import QueryProcessor
from .citation_service import CitationService
from ..validation.response_validator import validation_chain
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
        Process a user query with intent classification and pre-filtering.
        
        Args:
            message: User message
            conversation_id: Conversation identifier
            
        Returns:
            Tuple of (response_text, retrieved_documents)
        """
        try:
            # 1. Intent classification (Phase 2.1) - Using working version from copy folder
            from ...core.query.intent_classifier import intent_classifier
            intent_result = intent_classifier.classify_intent(message)
            
            # 2. Handle non-repository queries immediately
            if not intent_result.should_process:
                logger.info(f"Query filtered by intent classifier: {intent_result.category.value} (confidence: {intent_result.confidence:.2f})")
                
                if intent_result.suggested_response:
                    response = intent_result.suggested_response
                else:
                    response = "I can help you understand AI risks from the MIT AI Risk Repository. Try asking about employment impacts, safety concerns, privacy issues, or algorithmic bias."
                
                # Update conversation history even for filtered queries
                self._update_conversation_history(conversation_id, message, response)
                return response, []
            
            # 3. Process repository-related queries
            logger.info(f"Processing repository query (intent confidence: {intent_result.confidence:.2f})")
            
            # 4. Query refinement check (Phase 2.2) - handle over-broad queries
            from ...core.query.refinement import query_refiner
            refinement_result = query_refiner.analyze_query(message)
            
            # 5. Handle over-broad queries with suggestions (less aggressive)
            if refinement_result.needs_refinement and refinement_result.complexity.value == 'very_broad':
                logger.info(f"Query is very broad and needs refinement: {refinement_result.complexity.value}")
                
                # Use auto-refined query if available
                if refinement_result.refined_query:
                    logger.info(f"Using auto-refined query: {refinement_result.refined_query}")
                    message = refinement_result.refined_query
                elif refinement_result.suggestions:
                    # Only block very_broad queries with suggestions, let broad queries proceed
                    suggestion_response = query_refiner.format_suggestions_response(refinement_result)
                    self._update_conversation_history(conversation_id, message, suggestion_response)
                    return suggestion_response, []
            elif refinement_result.needs_refinement and refinement_result.complexity.value == 'broad':
                # For broad queries, use auto-refined query if available, but don't block with suggestions
                if refinement_result.refined_query:
                    logger.info(f"Using auto-refined query for broad query: {refinement_result.refined_query}")
                    message = refinement_result.refined_query
                # Let broad queries proceed to retrieval even if they have suggestions
            
            # 6. Analyze the query
            query_type, domain = self.query_processor.analyze_query(message)
            
            # 7. Retrieve relevant documents
            docs = self._retrieve_documents(message, query_type)
            
            # 8. Format context
            context = self._format_context(docs, query_type)
            
            # 9. Generate response
            response = self._generate_response(message, query_type, context, conversation_id, docs)
            
            # 10. Enhance with citations
            enhanced_response = self.citation_service.enhance_response_with_citations(response, docs)
            
            # 11. Self-validation chain for quality assurance
            validated_response, validation_results = validation_chain.validate_and_improve(
                response=enhanced_response,
                query=message,
                documents=docs,
                domain=domain
            )
            
            # Log validation results
            logger.info(f"Response validation: {validation_results.overall_result.value} "
                       f"(score: {validation_results.overall_score:.2f})")
            
            if validation_results.overall_score < 0.6:
                logger.warning(f"Low quality response detected. Recommendations: {validation_results.recommendations}")
            
            # 12. Update conversation history
            self._update_conversation_history(conversation_id, message, validated_response)
            
            return validated_response, docs
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            error_response = f"I encountered an error while processing your question: {str(e)}"
            return error_response, []
    
    def _retrieve_documents(self, message: str, query_type: str) -> List[Document]:
        """Retrieve relevant documents with relevance threshold filtering."""
        if not self.vector_store:
            logger.warning("No vector store available")
            return []
        
        try:
            # Detect domain for threshold-based filtering
            from ...config.domains import domain_classifier
            domain = domain_classifier.classify_domain(message)
            
            if query_type in ["employment", "socioeconomic"]:
                # Enhanced search for employment/socioeconomic queries
                enhanced_query = self.query_processor.enhance_query(message, query_type)
                docs = self.vector_store.get_relevant_documents(
                    enhanced_query, 
                    k=settings.DOMAIN_DOCS_RETRIEVED, 
                    domain="socioeconomic"
                )
                
                # Filter and prioritize employment-related documents
                docs = self.query_processor.filter_documents_by_relevance(docs, query_type)
                
                logger.info(f"Retrieved {len(docs)} documents using enhanced {domain} search")
            else:
                # Standard retrieval with relevance threshold
                docs = self.vector_store.get_relevant_documents(
                    message, 
                    k=settings.DEFAULT_DOCS_RETRIEVED, 
                    domain=domain
                )
                
                # If no docs above threshold, this is likely out-of-scope
                if not docs:
                    logger.info(f"No relevant documents found above threshold for query: {message[:50]}...")
                    return []
                
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
    
    def _generate_response(self, message: str, query_type: str, context: str, conversation_id: str, docs: List[Document] = None) -> str:
        """Generate response using the AI model."""
        if not self.gemini_model:
            logger.warning("No Gemini model available")
            return self._create_fallback_response(context, message)
        
        try:
            # Prepare conversation history
            history = self._get_conversation_history(conversation_id)
            
            # Generate enhanced prompt with session awareness and RID information
            prompt = self.query_processor.generate_prompt(message, query_type, context, conversation_id, docs)
            
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