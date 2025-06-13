"""
Base classes for AI models.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Iterator

class BaseModel(ABC):
    """Base class for AI models."""
    
    def __init__(self, api_key: str, model_name: str):
        self.api_key = api_key
        self.model_name = model_name
    
    @abstractmethod
    def generate(self, prompt: str, history: Optional[List[Dict[str, Any]]] = None) -> str:
        """Generate a response to the given prompt."""
        pass
    
    @abstractmethod
    def generate_stream(self, prompt: str, history: Optional[List[Dict[str, Any]]] = None) -> Iterator[str]:
        """Generate a streaming response to the given prompt."""
        pass

class BaseEmbeddingModel(ABC):
    """Base class for embedding models."""
    
    def __init__(self, api_key: str, model_name: str):
        self.api_key = api_key
        self.model_name = model_name
    
    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Generate embeddings for a query."""
        pass
    
    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple documents."""
        pass