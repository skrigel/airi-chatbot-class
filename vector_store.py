import os
import logging
import json
import csv
import io
import re
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from collections import Counter

from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader, DirectoryLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.docstore.document import Document
from langchain_core.retrievers import BaseRetriever
from langchain_community.retrievers import BM25Retriever

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HybridRetriever(BaseRetriever):
    """Combines vector similarity and keyword-based search for better results"""
    
    def __init__(
        self,
        vector_retriever: BaseRetriever,
        keyword_retriever: BaseRetriever,
        vector_weight: float = 0.7,
        rerank_top_k: int = 10
    ):
        # Tried different weights, 0.7 seems to work okay but might need tweaking
        # vector_retriever does semantic matching
        # keyword_retriever catches exact terms that might be missed
        super().__init__()
        self.vector_retriever = vector_retriever
        self.keyword_retriever = keyword_retriever
        self.vector_weight = max(0.0, min(1.0, vector_weight))  # Clamp between 0 and 1
        self.keyword_weight = 1.0 - self.vector_weight
        self.rerank_top_k = rerank_top_k
        
    def _get_relevant_documents(
        self, query: str, *, run_manager = None
    ) -> List[Document]:
        # Get docs from both retrievers
        vector_docs = self.vector_retriever.get_relevant_documents(query)
        keyword_docs = self.keyword_retriever.get_relevant_documents(query)
        
        # Track unique docs and scores - dict seems faster than a list with lookups
        unique_docs = {}
        
        # Add vector docs with position-based scoring
        for i, doc in enumerate(vector_docs):
            doc_id = self._get_doc_id(doc)
            # First doc gets full weight, last gets ~0
            score = self.vector_weight * (1.0 - (i / max(1, len(vector_docs))))
            unique_docs[doc_id] = {"doc": doc, "score": score}
        
        # Add keyword docs, combine scores for overlapping docs
        for i, doc in enumerate(keyword_docs):
            doc_id = self._get_doc_id(doc)
            score = self.keyword_weight * (1.0 - (i / max(1, len(keyword_docs))))
            
            # If already found by vector search, add scores
            if doc_id in unique_docs:
                unique_docs[doc_id]["score"] += score
            else:
                unique_docs[doc_id] = {"doc": doc, "score": score}
        
        # Sort by score and grab top results
        sorted_docs = sorted(unique_docs.values(), key=lambda x: x["score"], reverse=True)
        return [item["doc"] for item in sorted_docs[:self.rerank_top_k]]
    
    def _get_doc_id(self, doc: Document) -> str:
        # Need unique IDs based on metadata + content since same source might appear multiple times
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "")
        row = doc.metadata.get("row", "")
        
        # Quick mini-hash from first 100 chars - probably good enough
        content_hash = hash(doc.page_content[:100]) % 10000
        
        return f"{source}_{page}_{row}_{content_hash}"


class VectorStore:
    """Vector store for document embeddings and retrieval."""
    
    def __init__(self, 
                 embedding_provider: str = "google", 
                 api_key: Optional[str] = None,
                 repository_path: Optional[str] = None,
                 persist_directory: str = "chroma_db",
                 use_hybrid_search: bool = True):
        """
        Initialize the vector store.
        
        Args:
            embedding_provider: Provider for embeddings, 'google' or 'openai'
            api_key: API key for the embedding provider
            repository_path: Path to the directory containing documents
            persist_directory: Directory to persist the vector store
            use_hybrid_search: Whether to use hybrid search (vector+keyword)
        """
        self.embedding_provider = embedding_provider
        self.api_key = api_key
        self.repository_path = repository_path
        self.persist_directory = persist_directory
        self.use_hybrid_search = use_hybrid_search
        self.embeddings = self._initialize_embeddings()
        
        # Cache for query results
        self.query_cache = {}
        self.cache_expiry = 60 * 15  # 15 minutes
        
        # Different text splitters for different content types
        self.default_text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        # Splitter specifically for structured risk entries
        self.risk_entry_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,  # Larger chunks to keep risk entries together
            chunk_overlap=300,
            length_function=len,
        )
        
        self.vector_store = None
        self.keyword_retriever = None
        self.hybrid_retriever = None
        self.structured_data = []  # Store structured data for direct access
        self.all_documents = []  # Store all documents for BM25 indexing
        
    def _initialize_embeddings(self):
        """Initialize embeddings based on the provider."""
        if self.embedding_provider.lower() == "google":
            return GoogleGenerativeAIEmbeddings(
                model="embedding-001",
                google_api_key=self.api_key,
                task_type="retrieval_query"
            )
        else:
            raise ValueError(f"Unsupported embedding provider: {self.embedding_provider}")
    
    def _detect_file_format(self, file_path: Path) -> str:
        """
        Detect the format of a file based on content and extension.
        
        Args:
            file_path: Path to the file
            
        Returns:
            str: 'text', 'csv', 'json', or 'unknown'
        """
        extension = file_path.suffix.lower()
        
        if extension in ['.txt']:
            # Check if it looks like tabular data
            with open(file_path, 'r', encoding='utf-8') as f:
                first_lines = [next(f) for _ in range(5) if f]
                if any(line.count('\t') > 3 for line in first_lines):
                    return 'tabular'
            return 'text'
        elif extension in ['.csv', '.tsv']:
            return 'csv'
        elif extension in ['.json']:
            return 'json'
        else:
            # Try to guess based on content
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read(1024)  # Read first 1KB
                    if content.startswith('{') or content.startswith('['):
                        return 'json'
                    elif content.count(',') > 5 or content.count('\t') > 5:
                        return 'tabular'
                    else:
                        return 'text'
            except:
                return 'unknown'
    
    def _process_tabular_content(self, file_path: Path) -> List[Document]:
        """
        Process tabular content (CSV/TSV or tab-delimited text files).
        
        Args:
            file_path: Path to the file
            
        Returns:
            List of Document objects
        """
        documents = []
        delimiter = '\t' if str(file_path).endswith('.tsv') else ','
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                # Check if first line might be headers
                first_line = f.readline().strip()
                has_headers = not any(c.isdigit() for c in first_line.split(delimiter)[0])
                
                # Reset file pointer
                f.seek(0)
                
                # Parse as CSV
                reader = csv.reader(f, delimiter=delimiter)
                headers = next(reader) if has_headers else None
                
                # Process rows
                for i, row in enumerate(reader):
                    if not row or all(not cell.strip() for cell in row):
                        continue  # Skip empty rows
                    
                    # Create formatted text from row
                    if headers:
                        # Format with headers
                        entry_text = "Risk Entry:\n"
                        for j, value in enumerate(row):
                            if j < len(headers) and value.strip():
                                entry_text += f"{headers[j]}: {value}\n"
                    else:
                        # Format without headers
                        entry_text = "Risk Entry " + str(i) + ":\n"
                        for value in row:
                            if value.strip():
                                entry_text += f"{value}\n"
                    
                    # Create document
                    doc = Document(
                        page_content=entry_text,
                        metadata={
                            "source": str(file_path),
                            "row": i + (1 if has_headers else 0),
                            "file_type": "tabular"
                        }
                    )
                    documents.append(doc)
                    
                    # Also store structured data
                    if headers:
                        entry_data = {headers[j]: row[j] for j in range(min(len(headers), len(row)))}
                        entry_data["_source"] = str(file_path)
                        entry_data["_row"] = i + 1
                        self.structured_data.append(entry_data)
        
        except Exception as e:
            logger.error(f"Error processing tabular file {file_path}: {str(e)}")
        
        return documents
    
    def _process_risk_entries(self, file_path: Path) -> List[Document]:
        """
        Process files that contain MIT Risk Repository entries.
        
        Args:
            file_path: Path to the file
            
        Returns:
            List of Document objects
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check if this file looks like it contains risk entries
            risk_indicators = ["Risk category", "Risk subcategory", "Socioeconomic", 
                              "Increased inequality", "employment quality"]
            
            is_risk_file = any(indicator in content for indicator in risk_indicators)
            
            if not is_risk_file:
                # Process as regular text file
                return [Document(
                    page_content=content,
                    metadata={"source": str(file_path), "file_type": "text"}
                )]
            
            # Process as risk entries file
            documents = []
            
            # Split into entries - this is a simplified approach
            # For real implementation, would need more sophisticated parsing based on the actual format
            if '\t' in content or ',' in content:
                # Looks like tabular data, process as such
                return self._process_tabular_content(file_path)
            else:
                # Simple text splitting for entries separated by newlines
                entries = content.split('\n\n')
                for i, entry in enumerate(entries):
                    if entry.strip():
                        doc = Document(
                            page_content=entry,
                            metadata={
                                "source": str(file_path),
                                "entry_index": i,
                                "file_type": "risk_entry"
                            }
                        )
                        documents.append(doc)
            
            return documents
                
        except Exception as e:
            logger.error(f"Error processing risk entries file {file_path}: {str(e)}")
            return []
    
    def ingest_documents(self):
        """Ingest documents from the repository path into the vector store."""
        if not self.repository_path or not os.path.exists(self.repository_path):
            logger.error(f"Repository path {self.repository_path} does not exist")
            return False
        
        try:
            # Process all files in the repository
            logger.info(f"Loading documents from {self.repository_path}")
            
            self.all_documents = []
            base_path = Path(self.repository_path)
            
            for file_path in base_path.glob("**/*"):
                if not file_path.is_file() or file_path.name.startswith('.'):
                    continue
                
                logger.info(f"Processing file: {file_path}")
                
                # Detect file format
                format_type = self._detect_file_format(file_path)
                
                if format_type == 'tabular':
                    # Process as tabular data
                    docs = self._process_tabular_content(file_path)
                    self.all_documents.extend(docs)
                elif format_type == 'text':
                    # Check if it might contain risk entries
                    docs = self._process_risk_entries(file_path)
                    self.all_documents.extend(docs)
                else:
                    # Process as regular text
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        # Add document title/filename as metadata
                        filename = os.path.basename(file_path)
                        title = filename.replace('.txt', '').replace('_', ' ').replace('-', ' ')
                        
                        doc = Document(
                            page_content=content,
                            metadata={
                                "source": str(file_path), 
                                "file_type": format_type,
                                "title": title,
                                "filename": filename
                            }
                        )
                        self.all_documents.append(doc)
                    except Exception as e:
                        logger.error(f"Error reading file {file_path}: {e}")
            
            logger.info(f"Loaded {len(self.all_documents)} documents")
            
            # Process the documents for both vector and keyword search
            self._process_documents_for_retrieval()
            
            return True
            
        except Exception as e:
            logger.error(f"Error ingesting documents: {str(e)}")
            return False
            
    def _process_documents_for_retrieval(self):
        """Process documents for both vector and keyword retrieval."""
        if not self.all_documents:
            logger.warning("No documents to process")
            return
            
        try:
            # Split documents into chunks
            texts = self.default_text_splitter.split_documents(self.all_documents)
            logger.info(f"Split into {len(texts)} chunks")
            
            # 1. Create or update vector store
            if os.path.exists(self.persist_directory):
                logger.info(f"Loading existing vector store from {self.persist_directory}")
                self.vector_store = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings
                )
                # Add documents to existing store
                self.vector_store.add_documents(texts)
            else:
                logger.info(f"Creating new vector store in {self.persist_directory}")
                self.vector_store = Chroma.from_documents(
                    documents=texts,
                    embedding=self.embeddings,
                    persist_directory=self.persist_directory
                )
            
            # Persist the vector store
            self.vector_store.persist()
            logger.info("Vector store created and persisted successfully")
            
            # 2. Initialize BM25 retriever for keyword search
            if self.use_hybrid_search:
                logger.info("Initializing BM25 retriever for keyword search")
                self.keyword_retriever = BM25Retriever.from_documents(texts)
                self.keyword_retriever.k = 10  # Return top 10 results
                
                # 3. Create hybrid retriever
                logger.info("Creating hybrid retriever")
                self.hybrid_retriever = HybridRetriever(
                    vector_retriever=self.vector_store.as_retriever(search_kwargs={"k": 10}),
                    keyword_retriever=self.keyword_retriever,
                    vector_weight=0.7,  # Favor semantic search but consider keywords
                    rerank_top_k=10
                )
        except Exception as e:
            logger.error(f"Error processing documents for retrieval: {str(e)}")
            raise
    
    def get_relevant_documents(self, query: str, k: int = 5) -> List[Document]:
        """Find the most relevant docs for a query"""
        # Check if we've seen this query before (save some API calls)
        cache_key = f"{query}_{k}"
        current_time = time.time()
        
        # Return cached results if we have them and they're not too old
        if cache_key in self.query_cache:
            cache_time, docs = self.query_cache[cache_key]
            if current_time - cache_time < self.cache_expiry:
                logger.info(f"Using cached results for query: {query}")
                return docs
        
        # Initialize stores if needed
        if not self.vector_store and os.path.exists(self.persist_directory):
            logger.info(f"Loading existing vector store from {self.persist_directory}")
            self.vector_store = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
            
            # If we loaded vector store but hybrid search is enabled, recreate the hybrid retriever
            if self.use_hybrid_search and self.all_documents:
                # Split documents for BM25
                texts = self.default_text_splitter.split_documents(self.all_documents)
                
                # Initialize keyword retriever
                self.keyword_retriever = BM25Retriever.from_documents(texts)
                self.keyword_retriever.k = 10
                
                # Create hybrid retriever
                self.hybrid_retriever = HybridRetriever(
                    vector_retriever=self.vector_store.as_retriever(search_kwargs={"k": 10}),
                    keyword_retriever=self.keyword_retriever,
                    vector_weight=0.7,
                    rerank_top_k=10
                )
        
        if not self.vector_store:
            logger.error("Vector store not initialized. Call ingest_documents first.")
            return []
        
        try:
            # Process the query - normalize, expand, etc.
            processed_query = self._preprocess_query(query)
            
            # Get documents based on search strategy
            if self.use_hybrid_search and self.hybrid_retriever:
                # Use hybrid search
                logger.info(f"Using hybrid search for query: {processed_query}")
                docs = self.hybrid_retriever.get_relevant_documents(processed_query)
                # Limit to k results
                docs = docs[:k]
            else:
                # Fall back to vector search only
                logger.info(f"Using vector search for query: {processed_query}")
                docs = self.vector_store.similarity_search(processed_query, k=k)
            
            # For employment-related queries, also check structured data directly
            if self._is_employment_related(query) and self.structured_data:
                employment_docs = self._get_employment_related_docs()
                
                # Combine and deduplicate results
                existing_sources = {(doc.metadata.get('source', ''), doc.metadata.get('row', '')) for doc in docs}
                for doc in employment_docs:
                    source_key = (doc.metadata.get('source', ''), doc.metadata.get('row', ''))
                    if source_key not in existing_sources and len(docs) < k * 2:
                        docs.append(doc)
                        existing_sources.add(source_key)
            
            # Apply citations/references to documents
            docs = self._add_citations_to_docs(docs)
            
            # Cache the results
            self.query_cache[cache_key] = (current_time, docs)
            
            return docs
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            return []
    
    def _preprocess_query(self, query: str) -> str:
        """Make queries better for retrieval"""
        # Clean it up
        processed_query = query.strip()
        
        # Add some important terms that might help retrieval
        key_terms = self._extract_key_terms(query)
        
        # Short queries need help - expand them
        if len(processed_query.split()) < 5 and key_terms:
            processed_query = f"{processed_query} {' '.join(key_terms)}"
        
        # Special handling for questions - grab the focus
        if processed_query.endswith('?'):
            question_words = ['what', 'which', 'who', 'where', 'when', 'how', 'why']
            words = processed_query.lower().split()
            
            # Quick check if it starts with a question word
            if words and words[0] in question_words:
                focus_terms = []
                
                # Capitalized words often indicate the subject
                # Could do fancy NLP here but this seems good enough
                cap_words = re.findall(r'\b[A-Z][a-zA-Z]+\b', processed_query)
                quotes = re.findall(r'"([^"]+)"', processed_query)
                
                focus_terms.extend(cap_words)
                focus_terms.extend(quotes)
                
                # Add any focus terms we found
                if focus_terms:
                    processed_query = f"{processed_query} {' '.join(focus_terms)}"
        
        return processed_query
        
    def _add_citations_to_docs(self, docs: List[Document]) -> List[Document]:
        """Add source info to docs so we can tell users where stuff came from"""
        for doc in docs:
            # Get some identifiers from metadata
            source = doc.metadata.get('source', 'Unknown source')
            title = doc.metadata.get('title', os.path.basename(source))
            
            # Simple citation format
            citation = f"Source: {title}"
            
            # Add entry number for structured data
            if 'row' in doc.metadata:
                citation += f" (Entry {doc.metadata['row']})"
                
            doc.metadata['citation'] = citation
            
        return docs
    
    def _is_employment_related(self, query: str) -> bool:
        """Check if a query is related to employment or labor issues."""
        employment_terms = [
            "job", "jobs", "employment", "unemployment", "labor", "labour", 
            "work", "worker", "workers", "workforce", "workplace", "displacement",
            "inequality", "income", "wage", "salary", "automation", "economic",
            "AI taking jobs", "job loss", "replaced by AI", "AI automation"
        ]
        
        lower_query = query.lower()
        return any(term in lower_query for term in employment_terms)
    
    def _get_employment_related_docs(self) -> List[Document]:
        """Get documents specifically related to employment from structured data."""
        docs = []
        
        for entry in self.structured_data:
            # Check if employment-related based on category or content
            is_employment = False
            
            # Check category fields
            category_fields = ["Risk category", "Risk subcategory", "Category", "Sub-domain"]
            for field in category_fields:
                if field in entry and entry[field] and isinstance(entry[field], str):
                    value = entry[field].lower()
                    if any(term in value for term in ["inequality", "employment", "economic", "labor", "job"]):
                        is_employment = True
                        break
            
            # Check description fields
            if not is_employment:
                desc_fields = ["Description", "Additional ev."]
                for field in desc_fields:
                    if field in entry and entry[field] and isinstance(entry[field], str):
                        value = entry[field].lower()
                        if any(term in value for term in ["job", "employment", "unemployment", "labor", 
                                                         "work", "worker", "workforce", "displacement"]):
                            is_employment = True
                            break
            
            if is_employment:
                # Format entry as a document
                content = "Employment-Related Risk Entry:\n"
                for key, value in entry.items():
                    if key.startswith('_'):  # Skip metadata fields
                        continue
                    if value and isinstance(value, str):
                        content += f"{key}: {value}\n"
                
                doc = Document(
                    page_content=content,
                    metadata={
                        "source": entry.get("_source", "structured_data"),
                        "row": entry.get("_row", 0),
                        "file_type": "employment_data"
                    }
                )
                docs.append(doc)
        
        return docs
    
    def _extract_key_terms(self, query: str) -> List[str]:
        """Extract key terms from a query to enhance search."""
        # This could be improved with NLP techniques
        important_terms = set()
        
        # Important keywords for risk analysis
        risk_terms = [
            "risk", "danger", "threat", "hazard", "vulnerability", 
            "impact", "harm", "damage", "consequence", "effect",
            "mitigation", "prevention", "control", "management",
            "AI", "artificial intelligence", "machine learning", "algorithm",
            "ethics", "ethical", "governance", "regulation", "policy",
            "employment", "job", "labor", "automation", "displacement",
            "inequality", "bias", "discrimination", "fairness",
            "privacy", "security", "safety", "reliability", "robustness",
            "transparency", "explainability", "accountability"
        ]
        
        # Check for risk terms in query
        query_lower = query.lower()
        for term in risk_terms:
            if term.lower() in query_lower:
                important_terms.add(term)
        
        return list(important_terms)
    
    def format_context_from_docs(self, docs: List[Document]) -> str:
        """Format docs into a nice context string with citations"""
        if not docs:
            return ""
        
        context_parts = []
        context_parts.append("RELEVANT INFORMATION FROM THE MIT AI RISK REPOSITORY:\n")
        
        # Group docs by source - seems to read better this way
        docs_by_source = {}
        for doc in docs:
            source = doc.metadata.get('source', 'Unknown')
            if source not in docs_by_source:
                docs_by_source[source] = []
            docs_by_source[source].append(doc)
        
        # Go through each source and format its docs
        section_num = 1
        for source, source_docs in docs_by_source.items():
            filename = os.path.basename(source)
            title = filename.replace('.txt', '').replace('_', ' ').replace('-', ' ')
            
            # Handle different types of entries
            if any(doc.metadata.get('file_type') in ['tabular', 'risk_entry', 'employment_data'] for doc in source_docs):
                # These are database entries
                context_parts.append(f"SECTION {section_num}: Risk Repository Entries")
                section_num += 1
                
                for i, doc in enumerate(source_docs, 1):
                    entry_content = doc.page_content.strip()
                    citation = doc.metadata.get('citation', f"Source: {filename}")
                    context_parts.append(f"Entry {i}:\n{entry_content}\n[{citation}]\n")
            else:
                # Regular text chunks
                context_parts.append(f"SECTION {section_num}: {title}")
                section_num += 1
                
                for i, doc in enumerate(source_docs):
                    content = doc.page_content.strip()
                    citation = doc.metadata.get('citation', f"Source: {filename}")
                    
                    # Try to use first line as a title if it's not too long
                    lines = content.split('\n')
                    chunk_title = ""
                    if lines and lines[0].strip() and len(lines[0]) < 80:
                        chunk_title = f": {lines[0].strip()}"
                    
                    context_parts.append(f"Excerpt {i+1}{chunk_title}\n{content}\n[{citation}]\n")
        
        # Tell the model how to use the data
        context_parts.append(
            "NOTE: Cite relevant sections when answering the question"
        )
        
        return "\n\n".join(context_parts)
    
    def format_context_for_domain(self, docs: List[Document], domain: str) -> str:
        """Special formatting for specific domains like employment"""
        if not docs:
            return ""
        
        # Handle employment questions specially - lots of these in the repo
        if domain.lower() == "employment":
            context_parts = [
                f"RELEVANT INFO ON AI & EMPLOYMENT/LABOR MARKETS:",
                "Entries from MIT AI Risk Repository on employment effects:"
            ]
            
            # Sort so employment-specific docs come first
            emp_docs = []
            other_docs = []
            
            # This is hacky but seems to work ok - could use better heuristics
            for doc in docs:
                if ("employment" in doc.page_content.lower() or 
                    "jobs" in doc.page_content.lower() or 
                    "labor" in doc.page_content.lower() or
                    "inequality" in doc.page_content.lower()):
                    emp_docs.append(doc)
                else:
                    other_docs.append(doc)
            
            # List most relevant docs first
            all_docs = emp_docs + other_docs
            
            # Format each doc
            for i, doc in enumerate(all_docs, 1):
                content = doc.page_content.strip()
                citation = doc.metadata.get('citation', f"Source: {os.path.basename(doc.metadata.get('source', 'Unknown'))}")
                context_parts.append(f"Entry {i}:\n{content}\n[{citation}]")
            
            # Hint for the model
            context_parts.append(
                "Focus on concrete employment impacts from these entries"
            )
            
            return "\n\n".join(context_parts)
        else:
            # Default formatting for other domains
            return self.format_context_from_docs(docs)