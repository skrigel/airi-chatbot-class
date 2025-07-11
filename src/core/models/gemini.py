"""
Gemini model implementation with automatic quota fallback.
"""
import google.generativeai as genai
from typing import List, Dict, Any, Optional, Iterator
import mimetypes

from .base import BaseModel
from .gemini_pool import GeminiModelPool
from ...config.logging import get_logger
from ...config.settings import settings

logger = get_logger(__name__)

class GeminiModel(BaseModel):
    """Gemini AI model implementation with automatic quota fallback."""
    
    def __init__(self, api_key: str, model_name: str = None, use_fallback: bool = True):
        """
        Initialize the Gemini model.
        
        Args:
            api_key: Gemini API key
            model_name: Model name to use (if None, uses model chain)
            use_fallback: Whether to use multi-model fallback
        """
        self.api_key = api_key
        self.use_fallback = use_fallback
        
        if use_fallback:
            # Use the model pool for automatic fallback
            self.model_pool = GeminiModelPool(api_key)
            self.model_name = self.model_pool.model_name
            logger.info(f"Initialized Gemini model with fallback chain: {settings.GEMINI_MODEL_CHAIN}")
        else:
            # Use single model (legacy mode)
            self.model_name = model_name or settings.GEMINI_MODEL_NAME
            super().__init__(api_key, self.model_name)
            
            # Configure MIME types
            mimetypes.add_type('text/plain', '.txt')
            
            # Initialize client
            genai.configure(api_key=api_key)
            self.client = genai
            
            # Safety settings
            self.safety_settings = []
            self.model_pool = None
            
            logger.info(f"Initialized Gemini single model: {self.model_name}")
    
    def generate(self, prompt: str, history: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        Generate a response to the given prompt.
        
        Args:
            prompt: Input prompt
            history: Conversation history
            
        Returns:
            Generated response text
        """
        if self.model_pool:
            # Use model pool with automatic fallback
            return self.model_pool.generate(prompt, history)
        else:
            # Use single model (legacy mode)
            try:
                model = genai.GenerativeModel(model_name=self.model_name)
                
                if history:
                    chat = model.start_chat(history=history)
                    response = chat.send_message(prompt)
                else:
                    response = model.generate_content(prompt)
                
                return response.text if hasattr(response, 'text') else str(response)
                
            except Exception as e:
                logger.error(f"Error generating response: {str(e)}")
                return f"I encountered an error while generating a response: {str(e)}"
    
    def generate_stream(self, prompt: str, history: Optional[List[Dict[str, Any]]] = None) -> Iterator[str]:
        """
        Generate a streaming response to the given prompt.
        
        Args:
            prompt: Input prompt
            history: Conversation history
            
        Yields:
            Response chunks
        """
        if self.model_pool:
            # Use model pool with automatic fallback
            yield from self.model_pool.generate_stream(prompt, history)
        else:
            # Use single model (legacy mode)
            try:
                model = genai.GenerativeModel(model_name=self.model_name)
                
                if history:
                    chat = model.start_chat(history=history)
                    response = chat.send_message(prompt, stream=True)
                else:
                    response = model.generate_content(prompt, stream=True)
                
                for chunk in response:
                    if hasattr(chunk, 'text') and chunk.text:
                        yield chunk.text
                        
            except Exception as e:
                logger.error(f"Error generating streaming response: {str(e)}")
                yield f"I encountered an error while generating a response: {str(e)}"
    
    def generate_response(self, prompt: str, stream: bool = False, history: Optional[List[Dict[str, Any]]] = None):
        """
        Legacy method for backward compatibility.
        
        Args:
            prompt: Input prompt
            stream: Whether to stream the response
            history: Conversation history
            
        Returns:
            Response text or generator
        """
        if self.model_pool:
            # Use model pool
            return self.model_pool.generate_response(prompt, stream, history)
        else:
            # Use single model (legacy mode)
            if stream:
                return self.generate_stream(prompt, history)
            else:
                return self.generate(prompt, history)
    
    def get_embedding(self, text: str) -> Optional[List[float]]:
        """
        Get text embedding using Gemini's embedding model.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector or None if failed
        """
        if self.model_pool:
            # Use model pool
            return self.model_pool.get_embedding(text)
        else:
            # Use single model (legacy mode)
            try:
                response = genai.embed_content(
                    model=settings.EMBEDDING_MODEL_NAME,
                    content=text,
                    task_type="semantic_similarity"
                )
                return response['embedding']
            except Exception as e:
                logger.warning(f"Failed to get embedding: {str(e)}")
                return None
    
    def reset_conversation(self) -> None:
        """Reset conversation history."""
        # This is handled at the application level, not model level
        pass
    
    def get_status(self) -> Dict[str, Any]:
        """Get status information about the model or model pool."""
        if self.model_pool:
            return self.model_pool.get_status()
        else:
            return {
                "current_model": self.model_name,
                "model_chain": [self.model_name],
                "failed_models": {},
                "available_models": [self.model_name],
                "mode": "single_model"
            }