"""
Gemini model pool with automatic quota fallback.
"""
import time
import google.generativeai as genai
from typing import List, Dict, Any, Optional, Iterator
import mimetypes

from .base import BaseModel
from ...config.logging import get_logger
from ...config.settings import settings

logger = get_logger(__name__)

class GeminiModelPool(BaseModel):
    """Gemini AI model pool with automatic quota fallback."""
    
    def __init__(self, api_key: str, model_chain: Optional[List[str]] = None):
        """
        Initialize the Gemini model pool.
        
        Args:
            api_key: Gemini API key
            model_chain: List of model names to try in order
        """
        self.api_key = api_key
        self.model_chain = model_chain or settings.GEMINI_MODEL_CHAIN
        self.current_model_index = 0
        
        # Track failed models and their cooldown times
        self.failed_models = {}  # model_name -> failure_time
        
        # Configure MIME types
        mimetypes.add_type('text/plain', '.txt')
        
        # Initialize client
        genai.configure(api_key=api_key)
        self.client = genai
        
        # Safety settings
        self.safety_settings = []
        
        logger.info(f"Initialized Gemini model pool with chain: {self.model_chain}")
    
    @property
    def model_name(self) -> str:
        """Get current active model name."""
        return self.model_chain[self.current_model_index]
    
    def _is_quota_error(self, error_str: str) -> bool:
        """Check if error is a quota/rate limit error."""
        error_lower = error_str.lower()
        quota_indicators = [
            "429", "quota", "rate limit", "exceeded", "too many requests",
            "quota_metric", "GenerateRequestsPerDayPerProjectPerModel-FreeTier"
        ]
        return any(indicator in error_lower for indicator in quota_indicators)
    
    def _is_model_available(self, model_name: str) -> bool:
        """Check if a model is available (not in cooldown)."""
        if model_name not in self.failed_models:
            return True
        
        failure_time = self.failed_models[model_name]
        cooldown_expired = time.time() - failure_time > settings.MODEL_COOLDOWN_TIME
        
        if cooldown_expired:
            # Remove from failed models list
            del self.failed_models[model_name]
            logger.info(f"Model {model_name} cooldown expired, back in rotation")
            return True
        
        return False
    
    def _mark_model_failed(self, model_name: str):
        """Mark a model as failed and put it in cooldown."""
        self.failed_models[model_name] = time.time()
        logger.warning(f"Model {model_name} marked as failed, cooldown until {time.ctime(time.time() + settings.MODEL_COOLDOWN_TIME)}")
    
    def _get_next_available_model(self) -> Optional[str]:
        """Get the next available model in the chain."""
        for i, model_name in enumerate(self.model_chain):
            if self._is_model_available(model_name):
                self.current_model_index = i
                return model_name
        
        # If all models are in cooldown, use the primary one anyway
        logger.warning("All models in cooldown, falling back to primary model")
        self.current_model_index = 0
        return self.model_chain[0]
    
    def _generate_with_model(self, model_name: str, prompt: str, history: Optional[List[Dict[str, Any]]] = None) -> str:
        """Generate response with a specific model."""
        try:
            # Get model settings
            model_settings = settings.MODEL_SETTINGS.get(model_name, {})
            
            # Add thinking mode if supported
            if model_settings.get("supports_thinking", False):
                # For thinking models, we can add thinking instructions
                enhanced_prompt = f"<thinking>\nLet me think about this query step by step to provide the most accurate response from the AI Risk Repository.\n</thinking>\n\n{prompt}"
            else:
                enhanced_prompt = prompt
            
            # Create model with generation config
            generation_config = genai.GenerationConfig(
                max_output_tokens=model_settings.get("max_tokens", 8192),
                temperature=model_settings.get("temperature", 0.1)
            )
            
            model = genai.GenerativeModel(
                model_name=model_name,
                generation_config=generation_config,
                safety_settings=self.safety_settings
            )
            
            if history:
                chat = model.start_chat(history=history)
                response = chat.send_message(enhanced_prompt)
            else:
                response = model.generate_content(enhanced_prompt)
            
            logger.info(f"Successfully generated response using model: {model_name}")
            
            # Handle different response formats
            if hasattr(response, 'text'):
                return response.text
            elif hasattr(response, 'candidates') and response.candidates:
                # Handle multi-part responses
                candidate = response.candidates[0]
                if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                    text_parts = []
                    for part in candidate.content.parts:
                        if hasattr(part, 'text'):
                            text_parts.append(part.text)
                    return ''.join(text_parts)
            
            return str(response)
            
        except Exception as e:
            error_str = str(e)
            logger.error(f"Error generating response with {model_name}: {error_str}")
            
            # Check if it's a quota error
            if self._is_quota_error(error_str):
                logger.warning(f"Quota exceeded for {model_name}, marking for cooldown")
                self._mark_model_failed(model_name)
                raise QuotaExceededError(f"Quota exceeded for {model_name}: {error_str}")
            else:
                # For non-quota errors, re-raise as-is
                raise e
    
    def generate(self, prompt: str, history: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        Generate a response with automatic model fallback.
        
        Args:
            prompt: Input prompt
            history: Conversation history
            
        Returns:
            Generated response text
        """
        last_error = None
        
        # Try each model in the chain
        for attempt in range(len(self.model_chain)):
            current_model = self._get_next_available_model()
            if not current_model:
                break
                
            try:
                return self._generate_with_model(current_model, prompt, history)
                
            except QuotaExceededError as e:
                last_error = e
                logger.info(f"Quota exceeded for {current_model}, trying next model...")
                continue
                
            except Exception as e:
                last_error = e
                logger.error(f"Non-quota error with {current_model}: {str(e)}")
                # For non-quota errors, try next model but with retries
                retry_count = 0
                while retry_count < settings.MAX_RETRIES_PER_MODEL:
                    try:
                        time.sleep(settings.MODEL_RETRY_DELAY)
                        return self._generate_with_model(current_model, prompt, history)
                    except Exception as retry_error:
                        retry_count += 1
                        last_error = retry_error
                        logger.warning(f"Retry {retry_count} failed for {current_model}: {str(retry_error)}")
                
                # Mark model as failed if all retries exhausted
                self._mark_model_failed(current_model)
                continue
        
        # If all models failed, return error message
        error_msg = f"All models in chain failed. Last error: {str(last_error)}"
        logger.error(error_msg)
        return f"I encountered an error while generating a response: {error_msg}"
    
    def generate_stream(self, prompt: str, history: Optional[List[Dict[str, Any]]] = None) -> Iterator[str]:
        """
        Generate a streaming response with automatic model fallback.
        
        Args:
            prompt: Input prompt
            history: Conversation history
            
        Yields:
            Response chunks
        """
        current_model = self._get_next_available_model()
        if not current_model:
            yield "I encountered an error: No models available"
            return
            
        try:
            # Get model settings
            model_settings = settings.MODEL_SETTINGS.get(current_model, {})
            
            # Add thinking mode if supported
            if model_settings.get("supports_thinking", False):
                enhanced_prompt = f"<thinking>\nLet me think about this query step by step to provide the most accurate response from the AI Risk Repository.\n</thinking>\n\n{prompt}"
            else:
                enhanced_prompt = prompt
            
            # Create model with generation config
            generation_config = genai.GenerationConfig(
                max_output_tokens=model_settings.get("max_tokens", 8192),
                temperature=model_settings.get("temperature", 0.1)
            )
            
            model = genai.GenerativeModel(
                model_name=current_model,
                generation_config=generation_config,
                safety_settings=self.safety_settings
            )
            
            if history:
                chat = model.start_chat(history=history)
                response = chat.send_message(enhanced_prompt, stream=True)
            else:
                response = model.generate_content(enhanced_prompt, stream=True)
            
            for chunk in response:
                if hasattr(chunk, 'text') and chunk.text:
                    yield chunk.text
                elif hasattr(chunk, 'candidates') and chunk.candidates:
                    # Handle multi-part streaming responses
                    candidate = chunk.candidates[0]
                    if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                        for part in candidate.content.parts:
                            if hasattr(part, 'text') and part.text:
                                yield part.text
                    
        except Exception as e:
            error_str = str(e)
            logger.error(f"Error generating streaming response with {current_model}: {error_str}")
            
            if self._is_quota_error(error_str):
                self._mark_model_failed(current_model)
                yield f"I encountered a quota error with {current_model}. Please try again."
            else:
                yield f"I encountered an error while generating a response: {error_str}"
    
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
    
    def get_embedding(self, text: str) -> Optional[List[float]]:
        """
        Get text embedding using Gemini's embedding model.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector or None if failed
        """
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
        """Get status information about the model pool."""
        return {
            "current_model": self.model_name,
            "model_chain": self.model_chain,
            "failed_models": {
                model: {
                    "failed_at": time.ctime(failure_time),
                    "cooldown_remaining": max(0, settings.MODEL_COOLDOWN_TIME - (time.time() - failure_time))
                }
                for model, failure_time in self.failed_models.items()
            },
            "available_models": [m for m in self.model_chain if self._is_model_available(m)]
        }

class QuotaExceededError(Exception):
    """Custom exception for quota exceeded errors."""
    pass