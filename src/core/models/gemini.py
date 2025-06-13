"""
Gemini model implementation.
"""
import google.generativeai as genai
from typing import List, Dict, Any, Optional, Iterator
import mimetypes

from .base import BaseModel
from ...config.logging import get_logger

logger = get_logger(__name__)

class GeminiModel(BaseModel):
    """Gemini AI model implementation."""
    
    def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash"):
        """
        Initialize the Gemini model.
        
        Args:
            api_key: Gemini API key
            model_name: Model name to use
        """
        super().__init__(api_key, model_name)
        
        # Configure MIME types
        mimetypes.add_type('text/plain', '.txt')
        
        # Initialize client
        genai.configure(api_key=api_key)
        self.client = genai
        
        # Safety settings
        self.safety_settings = []
        
        logger.info(f"Initialized Gemini model: {model_name}")
    
    def generate(self, prompt: str, history: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        Generate a response to the given prompt.
        
        Args:
            prompt: Input prompt
            history: Conversation history
            
        Returns:
            Generated response text
        """
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
        if stream:
            return self.generate_stream(prompt, history)
        else:
            return self.generate(prompt, history)
    
    def reset_conversation(self) -> None:
        """Reset conversation history."""
        # This is handled at the application level, not model level
        pass