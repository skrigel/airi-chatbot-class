"""
Vector store implementation with hybrid search capabilities.
"""
import os
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter

from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.docstore.document import Document
from langchain_core.retrievers import BaseRetriever
from langchain_community.retrievers import BM25Retriever

from .document_processor import DocumentProcessor
from ...config.logging import get_logger
from ...config.settings import settings

logger = get_logger(__name__)

class HybridRetriever(BaseRetriever):
    """Combines vector similarity and keyword-based search for better results."""
    
    def __init__(
        self,
        vector_retriever: BaseRetriever,
        keyword_retriever: BaseRetriever,
        vector_weight: float = 0.7,
        rerank_top_k: int = 10
    ):
        super().__init__()
        self.vector_retriever = vector_retriever
        self.keyword_retriever = keyword_retriever
        self.vector_weight = max(0.0, min(1.0, vector_weight))
        self.keyword_weight = 1.0 - self.vector_weight
        self.rerank_top_k = rerank_top_k
        
    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        """Get relevant documents using hybrid search."""
        # Get docs from both retrievers
        vector_docs = self.vector_retriever.get_relevant_documents(query)
        keyword_docs = self.keyword_retriever.get_relevant_documents(query)
        
        # Track unique docs and scores
        unique_docs = {}
        
        # Add vector docs with position-based scoring
        for i, doc in enumerate(vector_docs):
            doc_id = self._get_doc_id(doc)
            score = self.vector_weight * (1.0 - (i / max(1, len(vector_docs))))
            unique_docs[doc_id] = {"doc": doc, "score": score}
        
        # Add keyword docs, combine scores for overlapping docs
        for i, doc in enumerate(keyword_docs):
            doc_id = self._get_doc_id(doc)
            score = self.keyword_weight * (1.0 - (i / max(1, len(keyword_docs))))
            
            if doc_id in unique_docs:
                unique_docs[doc_id]["score"] += score
            else:
                unique_docs[doc_id] = {"doc": doc, "score": score}
        
        # Sort by score and return top results
        sorted_docs = sorted(unique_docs.values(), key=lambda x: x["score"], reverse=True)
        return [item["doc"] for item in sorted_docs[:self.rerank_top_k]]
    
    def _get_doc_id(self, doc: Document) -> str:
        """Generate unique ID for document."""
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "")
        row = doc.metadata.get("row", "")
        content_hash = hash(doc.page_content[:100]) % 10000
        return f"{source}_{page}_{row}_{content_hash}"

class VectorStore:
    """Vector store for document embeddings and retrieval."""
    
    def __init__(self, 
                 embedding_provider: str = "google", 
                 api_key: Optional[str] = None,
                 repository_path: Optional[str] = None,
                 persist_directory: Optional[str] = None,
                 use_hybrid_search: bool = True):
        """
        Initialize the vector store.
        
        Args:
            embedding_provider: Provider for embeddings
            api_key: API key for the embedding provider
            repository_path: Path to the directory containing documents
            persist_directory: Directory to persist the vector store
            use_hybrid_search: Whether to use hybrid search
        """
        self.embedding_provider = embedding_provider
        self.api_key = api_key or settings.GEMINI_API_KEY
        self.repository_path = repository_path or str(settings.INFO_FILES_DIR)
        self.persist_directory = persist_directory or str(settings.CHROMA_DB_DIR)
        self.use_hybrid_search = use_hybrid_search
        
        # Initialize embeddings
        self.embeddings = self._initialize_embeddings()
        
        # Cache for query results
        self.query_cache = {}
        self.cache_expiry = settings.QUERY_CACHE_EXPIRY
        
        # Text splitters
        self.default_text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.DEFAULT_CHUNK_SIZE,
            chunk_overlap=settings.DEFAULT_CHUNK_OVERLAP,
            length_function=len,
        )
        
        self.risk_entry_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.RISK_ENTRY_CHUNK_SIZE,
            chunk_overlap=settings.RISK_ENTRY_CHUNK_OVERLAP,
            length_function=len,
        )
        
        # Initialize components
        self.vector_store = None
        self.keyword_retriever = None
        self.hybrid_retriever = None
        self.structured_data = []
        self.all_documents = []
        
        # Document processor
        self.document_processor = DocumentProcessor()
        
        # Ensure directories exist
        settings.ensure_directories()
    
    def initialize(self) -> bool:
        """
        Initialize vector store with robust error handling.
        
        This method handles both fresh setup and existing store loading
        with a unified approach that doesn't need fallbacks.
        """
        try:
            logger.info(f"Initializing vector store - repository: {self.repository_path}, persist: {self.persist_directory}")
            
            # Check if repository path exists
            if not self.repository_path or not os.path.exists(self.repository_path):
                logger.error(f"Repository path {self.repository_path} does not exist")
                return False
            
            logger.info(f"Repository path exists with {len(os.listdir(self.repository_path))} items")
            
            # Determine initialization path
            if os.path.exists(self.persist_directory) and os.listdir(self.persist_directory):
                logger.info(f"Existing vector store found at {self.persist_directory} - loading")
                result = self.load_existing_store()
                logger.info(f"Load existing store result: {result}")
                return result
            else:
                logger.info(f"No existing vector store found at {self.persist_directory} - creating new one")
                result = self.ingest_documents()
                logger.info(f"Ingest documents result: {result}")
                return result
                
        except Exception as e:
            logger.error(f"Error during vector store initialization: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def _initialize_embeddings(self):
        """Initialize embeddings based on the provider."""
        if self.embedding_provider.lower() == "google":
            try:
                embeddings = GoogleGenerativeAIEmbeddings(
                    model=settings.EMBEDDING_MODEL_NAME,
                    google_api_key=self.api_key,
                    task_type="retrieval_query"
                )
                # Test the embeddings
                test_embedding = embeddings.embed_query("test query for embeddings")
                if test_embedding and len(test_embedding) > 0:
                    logger.info("Embeddings test successful")
                    return embeddings
                else:
                    logger.error("Embeddings test failed - returned empty embedding")
                    raise ValueError("Empty embedding result")
            except Exception as e:
                logger.error(f"Error initializing Google embeddings: {str(e)}")
                raise
        else:
            raise ValueError(f"Unsupported embedding provider: {self.embedding_provider}")
    
    def ingest_documents(self) -> bool:
        """Ingest documents from the repository path into the vector store."""
        if not self.repository_path or not os.path.exists(self.repository_path):
            logger.error(f"Repository path {self.repository_path} does not exist")
            return False
        
        try:
            logger.info(f"Loading documents from {self.repository_path}")
            
            self.all_documents = []
            base_path = Path(self.repository_path)
            
            files_processed = 0
            files_failed = 0
            
            # Process files in the repository
            for file_path in base_path.iterdir():
                if file_path.is_file() and not file_path.name.startswith('.'):
                    try:
                        docs = self._process_file(file_path)
                        if docs:
                            self.all_documents.extend(docs)
                            files_processed += 1
                            logger.info(f"Successfully processed {file_path.name}: {len(docs)} documents")
                        else:
                            logger.warning(f"No documents extracted from {file_path.name}")
                    except Exception as e:
                        logger.error(f"Error processing file {file_path.name}: {str(e)}")
                        files_failed += 1
            
            logger.info(f"Document processing summary: {files_processed} processed, {files_failed} with errors")
            logger.info(f"Loaded {len(self.all_documents)} total documents")
            
            if not self.all_documents:
                logger.error("No documents were loaded")
                return False
            
            # Split documents into chunks
            all_splits = []
            for doc in self.all_documents:
                if 'risk_entry' in doc.metadata.get('file_type', ''):
                    splits = self.risk_entry_splitter.split_documents([doc])
                else:
                    splits = self.default_text_splitter.split_documents([doc])
                all_splits.extend(splits)
            
            logger.info(f"Split into {len(all_splits)} chunks")
            
            # Create vector store
            if os.path.exists(self.persist_directory):
                # Load existing vector store
                try:
                    logger.info(f"Loading existing vector store from {self.persist_directory}")
                    self.vector_store = Chroma(
                        persist_directory=self.persist_directory,
                        embedding_function=self.embeddings
                    )
                    # Add new documents
                    self.vector_store.add_documents(all_splits)
                except Exception as e:
                    logger.warning(f"Error loading existing vector store: {str(e)}")
                    # Create new one
                    logger.info(f"Creating new vector store in {self.persist_directory}")
                    self.vector_store = Chroma.from_documents(
                        documents=all_splits,
                        embedding=self.embeddings,
                        persist_directory=self.persist_directory
                    )
            else:
                logger.info(f"Creating new vector store in {self.persist_directory}")
                self.vector_store = Chroma.from_documents(
                    documents=all_splits,
                    embedding=self.embeddings,
                    persist_directory=self.persist_directory
                )
            
            logger.info("Vector store created and persisted successfully")
            
            # Ensure complete initialization
            success = self._ensure_complete_initialization()
            if not success:
                logger.error("Vector store created but initialization incomplete")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error during document ingestion: {str(e)}")
            return False
    
    def _ensure_complete_initialization(self) -> bool:
        """Ensure vector store is completely initialized with all components."""
        try:
            # Check if vector store exists
            if not self.vector_store:
                logger.error("Vector store not initialized")
                return False
            
            # Test vector store basic functionality
            try:
                test_results = self.vector_store.similarity_search("test query", k=1)
                logger.info(f"Vector store test successful - found {len(test_results)} results")
            except Exception as e:
                logger.error(f"Vector store test failed: {str(e)}")
                return False
            
            # Set up hybrid retriever if enabled and not already set up
            if self.use_hybrid_search and not self.hybrid_retriever:
                try:
                    logger.info("Setting up hybrid retriever")
                    
                    # Get all documents from the vector store to rebuild BM25
                    all_docs = self.vector_store.get()
                    if not all_docs or not all_docs['documents']:
                        logger.warning("No documents found in vector store - disabling hybrid search")
                        self.use_hybrid_search = False
                        return True
                    
                    # Reconstruct Document objects for BM25
                    documents = []
                    for i, text in enumerate(all_docs['documents']):
                        metadata = all_docs['metadatas'][i] if all_docs['metadatas'] else {}
                        doc = Document(page_content=text, metadata=metadata)
                        documents.append(doc)
                    
                    logger.info(f"Building BM25 retriever with {len(documents)} documents")
                    self.keyword_retriever = BM25Retriever.from_documents(documents)
                    self.keyword_retriever.k = 10
                    
                    # Create hybrid retriever
                    self.hybrid_retriever = HybridRetriever(
                        vector_retriever=self.vector_store.as_retriever(),
                        keyword_retriever=self.keyword_retriever
                    )
                    
                    logger.info("Hybrid retriever initialized successfully")
                    
                except Exception as e:
                    logger.error(f"Error setting up hybrid retriever: {str(e)}")
                    logger.info("Falling back to vector-only search")
                    self.use_hybrid_search = False
            
            return True
            
        except Exception as e:
            logger.error(f"Error ensuring complete initialization: {str(e)}")
            return False
    
    def load_existing_store(self) -> bool:
        """Load existing vector store and ensure complete initialization."""
        try:
            if not os.path.exists(self.persist_directory):
                logger.error(f"Vector store directory {self.persist_directory} does not exist")
                return False
            
            # Load existing vector store
            logger.info(f"Loading existing vector store from {self.persist_directory}")
            self.vector_store = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
            
            # Ensure complete initialization
            success = self._ensure_complete_initialization()
            if success:
                logger.info("Vector store loaded and fully initialized")
            else:
                logger.error("Vector store loaded but initialization incomplete")
            
            return success
            
        except Exception as e:
            logger.error(f"Error loading existing vector store: {str(e)}")
            return False
    
    def _process_file(self, file_path: Path) -> List[Document]:
        """Process a single file."""
        file_extension = file_path.suffix.lower()
        
        if file_extension in ['.xlsx', '.xls']:
            return self.document_processor.process_excel_file(file_path)
        elif file_extension == '.txt':
            return self._process_text_file(file_path)
        else:
            logger.warning(f"Unsupported file type: {file_extension}")
            return []
    
    def _process_text_file(self, file_path: Path) -> List[Document]:
        """Process text files."""
        try:
            loader = TextLoader(str(file_path), encoding='utf-8')
            documents = loader.load()
            
            # Add metadata
            for doc in documents:
                doc.metadata.update({
                    "file_type": "text",
                    "title": file_path.name
                })
            
            return documents
        except Exception as e:
            logger.error(f"Error processing text file {file_path}: {str(e)}")
            return []
    
    def get_relevant_documents(self, query: str, k: int = 5) -> List[Document]:
        """Get relevant documents for a query."""
        # Check cache
        cache_key = f"{query}_{k}"
        if cache_key in self.query_cache:
            cached_time, cached_docs = self.query_cache[cache_key]
            if time.time() - cached_time < self.cache_expiry:
                return cached_docs
        
        try:
            if self.use_hybrid_search and self.hybrid_retriever:
                docs = self.hybrid_retriever.get_relevant_documents(query)[:k]
            elif self.vector_store:
                docs = self.vector_store.similarity_search(query, k=k)
            else:
                logger.error("No retriever available")
                return []
            
            # Cache the results
            self.query_cache[cache_key] = (time.time(), docs)
            
            # Add domain-specific documents for employment queries
            if any(keyword in query.lower() for keyword in ['employ', 'job', 'work', 'labor']):
                employment_docs = self._get_employment_specific_docs()
                docs.extend(employment_docs)
                logger.info(f"Added {len(employment_docs)} domain-specific documents for employment")
            
            return docs[:k]  # Ensure we don't exceed k
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            return []
    
    def _get_employment_specific_docs(self) -> List[Document]:
        """Get employment-specific documents."""
        if not self.vector_store:
            return []
        
        try:
            # Search for employment-related content
            employment_queries = [
                "6.2 Increased inequality and decline in employment quality",
                "6.3 Economic and cultural devaluation of human effort",
                "socioeconomic environmental AI risk"
            ]
            
            docs = []
            for query in employment_queries:
                search_docs = self.vector_store.similarity_search(query, k=2)
                docs.extend(search_docs)
            
            # Remove duplicates
            seen_content = set()
            unique_docs = []
            for doc in docs:
                content_hash = hash(doc.page_content[:100])
                if content_hash not in seen_content:
                    seen_content.add(content_hash)
                    unique_docs.append(doc)
            
            return unique_docs[:2]  # Return top 2 unique employment docs
            
        except Exception as e:
            logger.error(f"Error getting employment-specific docs: {str(e)}")
            return []
    
    def format_context_from_docs(self, docs: List[Document]) -> str:
        """Format documents into context string."""
        if not docs:
            return ""
        
        context = "INFORMATION FROM THE AI RISK REPOSITORY:\\n\\n"
        for i, doc in enumerate(docs, 1):
            # Add document metadata if available
            title = doc.metadata.get('title', f'Document {i}')
            domain = doc.metadata.get('domain', '')
            
            context += f"SECTION {i}"
            if domain and domain != 'Unspecified':
                context += f" (Domain: {domain})"
            context += f":\\n{doc.page_content}\\n\\n"
        
        return context