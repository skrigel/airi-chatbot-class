import os
import logging
import json
import csv
import io
import re
import time
import pandas as pd
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
            try:
                embeddings = GoogleGenerativeAIEmbeddings(
                    model="models/embedding-001",  # Updated model name format
                    google_api_key=self.api_key,
                    task_type="retrieval_query"
                )
                # Test the embeddings with a simple query to verify they work
                test_embedding = embeddings.embed_query("test query for embeddings")
                if test_embedding and len(test_embedding) > 0:
                    logger.info("Embeddings test successful")
                    return embeddings
                else:
                    logger.error("Embeddings test failed - returned empty embedding")
                    raise ValueError("Empty embedding result")
            except Exception as e:
                logger.error(f"Error initializing Google embeddings: {str(e)}")
                # Try the alternative embedding format
                try:
                    logger.info("Trying alternative embedding model format...")
                    embeddings = GoogleGenerativeAIEmbeddings(
                        model="embedding-001",  # Original model name format
                        google_api_key=self.api_key,
                        task_type="retrieval_query"
                    )
                    test_embedding = embeddings.embed_query("test query for embeddings")
                    if test_embedding and len(test_embedding) > 0:
                        logger.info("Alternative embeddings test successful")
                        return embeddings
                    else:
                        raise ValueError("Alternative embedding format also failed")
                except Exception as alt_err:
                    logger.error(f"Alternative embedding also failed: {str(alt_err)}")
                    raise
        else:
            raise ValueError(f"Unsupported embedding provider: {self.embedding_provider}")
    
    def _detect_file_format(self, file_path: Path) -> str:
        """
        Detect the format of a file based on content and extension.
        
        Args:
            file_path: Path to the file
            
        Returns:
            str: 'text', 'csv', 'json', 'excel', or 'unknown'
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
        elif extension in ['.xlsx', '.xls']:
            return 'excel'
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
                # Binary file - could be excel
                if file_path.name.endswith(('.xlsx', '.xls')):
                    return 'excel'
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
                        try:
                            entry_data = {headers[j]: row[j] for j in range(min(len(headers), len(row)))}
                            entry_data["_source"] = str(file_path)
                            entry_data["_row"] = i + 1
                            self.structured_data.append(entry_data)
                        except Exception as struct_err:
                            logger.warning(f"Could not create structured data for row {i}: {str(struct_err)}")
        
        except Exception as e:
            logger.error(f"Error processing tabular file {file_path}: {str(e)}")
            
            # Fallback: Try to create a single document with the file content
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    file_content = f.read()
                
                # If we have content, create a simple document with the raw text
                if file_content.strip():
                    logger.info(f"Creating fallback document for tabular file {file_path}")
                    fallback_text = f"Content from tabular file: {os.path.basename(file_path)}\n\n"
                    
                    # Truncate if the content is too large
                    if len(file_content) > 10000:
                        fallback_text += file_content[:10000] + "\n\n[Content truncated due to size]"
                    else:
                        fallback_text += file_content
                    
                    doc = Document(
                        page_content=fallback_text,
                        metadata={
                            "source": str(file_path),
                            "file_type": "tabular_fallback",
                            "title": f"Tabular Data: {os.path.basename(file_path)}"
                        }
                    )
                    documents.append(doc)
            except Exception as fallback_err:
                logger.error(f"Fallback processing for tabular file failed: {str(fallback_err)}")
        
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
                    metadata={
                        "source": str(file_path), 
                        "file_type": "text",
                        "title": os.path.basename(file_path)
                    }
                )]
            
            # Process as risk entries file
            documents = []
            
            # Split into entries - this is a simplified approach
            # For real implementation, would need more sophisticated parsing based on the actual format
            if '\t' in content or ',' in content:
                # Looks like tabular data, process as such
                return self._process_tabular_content(file_path)
            else:
                try:
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
                    
                    # If we successfully created documents, return them
                    if documents:
                        return documents
                    
                    # If we didn't get any valid entries, fall back to whole file approach
                    logger.warning(f"No valid entries found in risk entries file {file_path}, using fallback")
                except Exception as parsing_err:
                    logger.warning(f"Error parsing entries in {file_path}: {str(parsing_err)}, using fallback")
                
                # Fallback: use whole file as one document
                return [Document(
                    page_content=f"Risk Repository Content from {os.path.basename(file_path)}:\n\n{content}",
                    metadata={
                        "source": str(file_path),
                        "file_type": "risk_entry_fallback",
                        "title": f"Risk Repository: {os.path.basename(file_path)}"
                    }
                )]
            
        except Exception as e:
            logger.error(f"Error processing risk entries file {file_path}: {str(e)}")
            
            # Final fallback - create a placeholder document
            try:
                filename = os.path.basename(file_path)
                fallback_text = f"AI Risk Repository File: {filename}\n\n"
                fallback_text += "This file contains AI risk information but could not be processed.\n"
                fallback_text += "It may contain risk categories, descriptions, and other relevant data."
                
                return [Document(
                    page_content=fallback_text,
                    metadata={
                        "source": str(file_path),
                        "file_type": "fallback",
                        "title": f"AI Risk Info: {filename}"
                    }
                )]
            except Exception:
                # If even that fails, return empty list
                return []
    
    def _process_excel_content(self, file_path: Path) -> List[Document]:
        """
        Process Excel files with robust handling of the MIT AI Risk Repository structure.
        
        Args:
            file_path: Path to the Excel file
            
        Returns:
            List of Document objects
        """
        documents = []
        try:
            # Load the Excel file
            excel_data = pd.read_excel(file_path, sheet_name=None)
            logger.info(f"Found {len(excel_data)} sheets in Excel file {file_path}")
            
            # Process each sheet with specialized handling
            for sheet_name, df in excel_data.items():
                logger.info(f"Processing sheet: {sheet_name} with {len(df)} rows and {len(df.columns)} columns")
                
                # Skip empty dataframes
                if df.empty:
                    logger.info(f"Skipping empty sheet: {sheet_name}")
                    continue
                
                # Special handling for the main AI Risk Database sheet
                if 'AI Risk Database v3' in sheet_name:
                    documents.extend(self._process_ai_risk_database_sheet(df, sheet_name, file_path))
                elif 'Domain Taxonomy' in sheet_name:
                    documents.extend(self._process_domain_taxonomy_sheet(df, sheet_name, file_path))
                elif 'Causal Taxonomy' in sheet_name:
                    documents.extend(self._process_causal_taxonomy_sheet(df, sheet_name, file_path))
                else:
                    # Generic processing for other sheets
                    documents.extend(self._process_generic_sheet(df, sheet_name, file_path))
        
        except Exception as e:
            logger.error(f"Error processing Excel file {file_path}: {str(e)}")
            # Create a fallback document
            try:
                fallback_doc = Document(
                    page_content=f"Excel file: {os.path.basename(file_path)}\n\nError processing file: {str(e)}",
                    metadata={
                        "source": str(file_path),
                        "file_type": "excel_error",
                        "title": f"Error processing {os.path.basename(file_path)}"
                    }
                )
                documents.append(fallback_doc)
            except Exception as fallback_err:
                logger.error(f"Even fallback processing failed: {str(fallback_err)}")
        
        logger.info(f"Created {len(documents)} documents from Excel file {file_path}")
        return documents
    
    def _process_ai_risk_database_sheet(self, df: pd.DataFrame, sheet_name: str, file_path: Path) -> List[Document]:
        """
        Process the main AI Risk Database sheet with proper column detection.
        """
        documents = []
        
        # The AI Risk Database has header row at position 2
        try:
            # Load with proper header
            df_proper = pd.read_excel(file_path, sheet_name=sheet_name, header=2)
            logger.info(f"Loaded AI Risk Database with proper headers: {list(df_proper.columns)}")
            
            # Clean up the dataframe
            df_proper = df_proper.dropna(how='all')
            
            # Key columns for the AI Risk Database
            key_columns = {
                'title': 'Title',
                'domain': 'Domain', 
                'subdomain': 'Sub-domain',
                'risk_category': 'Risk category',
                'risk_subcategory': 'Risk subcategory', 
                'description': 'Description',
                'entity': 'Entity',
                'intent': 'Intent',
                'timing': 'Timing'
            }
            
            # Group documents by domain for better retrieval
            domain_groups = {}
            individual_docs = []
            
            for index, row in df_proper.iterrows():
                # Skip rows with no meaningful content
                if pd.isna(row.get('Title')) and pd.isna(row.get('Description')):
                    continue
                
                # Extract key information
                title = str(row.get('Title', '')).strip() if pd.notna(row.get('Title')) else ''
                domain = str(row.get('Domain', '')).strip() if pd.notna(row.get('Domain')) else 'Unspecified'
                subdomain = str(row.get('Sub-domain', '')).strip() if pd.notna(row.get('Sub-domain')) else ''
                risk_category = str(row.get('Risk category', '')).strip() if pd.notna(row.get('Risk category')) else ''
                description = str(row.get('Description', '')).strip() if pd.notna(row.get('Description')) else ''
                
                # Create comprehensive content for this risk entry
                content_parts = []
                
                if title:
                    content_parts.append(f"Title: {title}")
                
                if domain and domain != 'Unspecified':
                    content_parts.append(f"Domain: {domain}")
                    
                if subdomain:
                    content_parts.append(f"Sub-domain: {subdomain}")
                    
                if risk_category:
                    content_parts.append(f"Risk Category: {risk_category}")
                    
                if row.get('Risk subcategory') and pd.notna(row.get('Risk subcategory')):
                    content_parts.append(f"Risk Subcategory: {str(row['Risk subcategory']).strip()}")
                    
                if description:
                    content_parts.append(f"Description: {description}")
                    
                # Add additional evidence if available
                if row.get('Additional ev.') and pd.notna(row.get('Additional ev.')):
                    additional_ev = str(row['Additional ev.']).strip()
                    if additional_ev:
                        content_parts.append(f"Additional Evidence: {additional_ev}")
                
                # Add other relevant fields
                for field in ['Entity', 'Intent', 'Timing']:
                    if row.get(field) and pd.notna(row.get(field)):
                        value = str(row[field]).strip()
                        if value:
                            content_parts.append(f"{field}: {value}")
                
                if not content_parts:
                    continue  # Skip empty entries
                
                entry_content = "\n".join(content_parts)
                
                # Determine the most specific domain for categorization
                specific_domain = subdomain if subdomain else domain
                
                # Create individual document
                doc = Document(
                    page_content=entry_content,
                    metadata={
                        "source": str(file_path),
                        "sheet": sheet_name,
                        "row": index,
                        "file_type": "ai_risk_entry",
                        "title": title if title else f"Risk Entry {index}",
                        "domain": domain,
                        "subdomain": subdomain,
                        "risk_category": risk_category,
                        "specific_domain": specific_domain
                    }
                )
                individual_docs.append(doc)
                
                # Group by specific domain for aggregated documents
                if specific_domain and specific_domain != 'Unspecified':
                    if specific_domain not in domain_groups:
                        domain_groups[specific_domain] = []
                    domain_groups[specific_domain].append({
                        'content': entry_content,
                        'title': title,
                        'index': index
                    })
            
            # Add individual documents
            documents.extend(individual_docs)
            logger.info(f"Created {len(individual_docs)} individual risk entries")
            
            # Create domain-aggregated documents for better retrieval
            for domain_name, entries in domain_groups.items():
                if len(entries) >= 3:  # Only create aggregated docs for domains with multiple entries
                    # Create a comprehensive domain document
                    domain_content = f"AI Risk Domain: {domain_name}\n\n"
                    domain_content += f"This domain contains {len(entries)} risk entries from the AI Risk Repository:\n\n"
                    
                    # Include first 10 entries in full, then summarize the rest
                    for i, entry in enumerate(entries[:10]):
                        domain_content += f"Risk Entry {i+1}:\n{entry['content']}\n\n"
                    
                    if len(entries) > 10:
                        domain_content += f"Additional {len(entries) - 10} entries in this domain include:\n"
                        for entry in entries[10:]:
                            if entry['title']:
                                domain_content += f"- {entry['title']}\n"
                    
                    # Create domain aggregated document
                    domain_doc = Document(
                        page_content=domain_content,
                        metadata={
                            "source": str(file_path),
                            "sheet": sheet_name,
                            "file_type": "ai_risk_domain_summary",
                            "title": f"AI Risk Domain: {domain_name}",
                            "domain": domain_name,
                            "entry_count": len(entries),
                            "specific_domain": domain_name
                        }
                    )
                    documents.append(domain_doc)
            
            logger.info(f"Created {len(domain_groups)} domain summary documents")
            
        except Exception as e:
            logger.error(f"Error processing AI Risk Database sheet: {str(e)}")
            # Fallback to generic processing
            documents.extend(self._process_generic_sheet(df, sheet_name, file_path))
        
        return documents
    
    def _process_domain_taxonomy_sheet(self, df: pd.DataFrame, sheet_name: str, file_path: Path) -> List[Document]:
        """
        Process Domain Taxonomy sheets which contain domain definitions.
        """
        documents = []
        
        try:
            # Look for domain/subdomain definitions in the sheet
            for index, row in df.iterrows():
                # Look for rows that define domains or subdomains
                row_text = ' '.join([str(val) for val in row.values if pd.notna(val)]).strip()
                
                # Skip header rows or empty rows
                if not row_text or len(row_text) < 50:
                    continue
                
                # Check if this looks like a domain definition
                if any(keyword in row_text.lower() for keyword in ['socioeconomic', 'employment', 'economic', 'environmental', 'domain', 'subdomain']):
                    doc = Document(
                        page_content=f"Domain Taxonomy Entry:\n{row_text}",
                        metadata={
                            "source": str(file_path),
                            "sheet": sheet_name,
                            "row": index,
                            "file_type": "domain_taxonomy",
                            "title": f"Domain Definition from {sheet_name}"
                        }
                    )
                    documents.append(doc)
        
        except Exception as e:
            logger.error(f"Error processing domain taxonomy sheet: {str(e)}")
        
        return documents
    
    def _process_causal_taxonomy_sheet(self, df: pd.DataFrame, sheet_name: str, file_path: Path) -> List[Document]:
        """
        Process Causal Taxonomy sheets.
        """
        documents = []
        
        try:
            # Process causal taxonomy entries
            for index, row in df.iterrows():
                row_text = ' '.join([str(val) for val in row.values if pd.notna(val)]).strip()
                
                if not row_text or len(row_text) < 50:
                    continue
                
                doc = Document(
                    page_content=f"Causal Taxonomy Entry:\n{row_text}",
                    metadata={
                        "source": str(file_path),
                        "sheet": sheet_name,
                        "row": index,
                        "file_type": "causal_taxonomy",
                        "title": f"Causal Factor from {sheet_name}"
                    }
                )
                documents.append(doc)
        
        except Exception as e:
            logger.error(f"Error processing causal taxonomy sheet: {str(e)}")
        
        return documents
    
    def _process_generic_sheet(self, df: pd.DataFrame, sheet_name: str, file_path: Path) -> List[Document]:
        """
        Generic processing for other sheets.
        """
        documents = []
        
        try:
            # Try to detect header row
            header_row = self._detect_header_row(df)
            if header_row > 0:
                df.columns = [str(x).strip() if not pd.isna(x) else f"Column_{i}" 
                             for i, x in enumerate(df.iloc[header_row])]
                df = df.iloc[header_row+1:].reset_index(drop=True)
            
            df = df.dropna(how='all')
            
            # Create a single document for the sheet if it has substantial content
            if len(df) > 0:
                try:
                    if len(df) > 50:
                        sheet_content = df.head(50).to_string(index=False, na_rep="")
                        sheet_content += f"\n\n[Additional {len(df) - 50} rows not shown]"
                    else:
                        sheet_content = df.to_string(index=False, na_rep="")
                    
                    doc = Document(
                        page_content=f"Content from sheet {sheet_name}:\n\n{sheet_content}",
                        metadata={
                            "source": str(file_path),
                            "sheet": sheet_name,
                            "file_type": "excel_sheet",
                            "title": f"Sheet: {sheet_name}"
                        }
                    )
                    documents.append(doc)
                    
                except Exception as format_err:
                    logger.warning(f"Could not format sheet {sheet_name}: {str(format_err)}")
        
        except Exception as e:
            logger.error(f"Error in generic sheet processing: {str(e)}")
        
        return documents
            
    def _detect_header_row(self, df):
        """
        Detect the most likely header row in a DataFrame.
        Returns the index of the header row, or 0 if none detected.
        """
        # Common header terms to look for
        header_terms = ['risk', 'domain', 'category', 'description', 'source', 
                       'hazard', 'impact', 'type', 'id', 'reference']
        
        # Check first few rows for header-like characteristics
        for i in range(min(10, len(df))):
            row = df.iloc[i]
            
            # Check if many cells in this row contain header terms
            header_term_count = sum(1 for cell in row if isinstance(cell, str) and 
                                   any(term in cell.lower() for term in header_terms))
            
            # If multiple header terms found, likely a header row
            if header_term_count >= 2:
                return i
            
            # Check if row has many string values (not numeric) and not many NaNs
            string_ratio = sum(1 for cell in row if isinstance(cell, str)) / len(row)
            nan_ratio = sum(1 for cell in row if pd.isna(cell)) / len(row)
            
            if string_ratio > 0.7 and nan_ratio < 0.3:
                return i
        
        # Default to row 0 if no clear header detected
        return 0
    
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
            
            files_processed = 0
            files_failed = 0
            
            # Check for repository configuration file that may specify priority files
            priority_excel_file = None
            config_path = os.path.join(self.repository_path, ".repository_config")
            if os.path.exists(config_path):
                try:
                    with open(config_path, 'r') as config_file:
                        for line in config_file:
                            if line.startswith('excel_priority='):
                                priority_excel_file = line.strip().split('=', 1)[1]
                                logger.info(f"Found priority Excel file in config: {priority_excel_file}")
                                break
                except Exception as config_err:
                    logger.error(f"Error reading repository config: {str(config_err)}")
            
            # If we have a priority Excel file, process it first
            if priority_excel_file:
                priority_path = os.path.join(self.repository_path, priority_excel_file)
                if os.path.exists(priority_path):
                    logger.info(f"Processing priority Excel file first: {priority_path}")
                    try:
                        # Process as Excel file
                        docs = self._process_excel_content(Path(priority_path))
                        if docs:
                            self.all_documents.extend(docs)
                            files_processed += 1
                            logger.info(f"Successfully processed priority file: {priority_excel_file} - {len(docs)} documents created")
                    except Exception as priority_err:
                        logger.error(f"Error processing priority file {priority_path}: {str(priority_err)}")
                        files_failed += 1
            
            # Process all other files
            for file_path in base_path.glob("**/*"):
                if not file_path.is_file() or file_path.name.startswith('.'):
                    continue
                
                # Skip if this is our priority file that was already processed
                if priority_excel_file and file_path.name == priority_excel_file:
                    logger.info(f"Skipping already processed priority file: {file_path}")
                    continue
                
                logger.info(f"Processing file: {file_path}")
                
                try:
                    # Give priority to Excel files that might be the AI Risk Repository
                    if file_path.suffix.lower() in ['.xlsx', '.xls'] and ('risk' in file_path.name.lower() or 'repository' in file_path.name.lower()):
                        logger.info(f"Found potential AI Risk Repository Excel file: {file_path}")
                        format_type = 'excel'
                    else:
                        # Detect file format
                        format_type = self._detect_file_format(file_path)
                    
                    # Process based on detected format type
                    if format_type == 'tabular':
                        # Process as tabular data
                        docs = self._process_tabular_content(file_path)
                        self.all_documents.extend(docs)
                    elif format_type == 'excel':
                        # Process as Excel file
                        docs = self._process_excel_content(file_path)
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
                        except Exception as read_err:
                            logger.error(f"Error reading file {file_path}: {read_err}")
                            
                            # Add a minimal document as fallback
                            fallback_doc = Document(
                                page_content=f"File: {os.path.basename(file_path)} (could not be processed)",
                                metadata={
                                    "source": str(file_path),
                                    "file_type": "fallback",
                                    "title": os.path.basename(file_path),
                                    "error": str(read_err)
                                }
                            )
                            self.all_documents.append(fallback_doc)
                            files_failed += 1
                    
                    files_processed += 1
                
                except Exception as process_err:
                    logger.error(f"Failed to process file {file_path}: {str(process_err)}")
                    files_failed += 1
                    
                    # Create a minimal fallback document
                    try:
                        fallback_doc = Document(
                            page_content=f"AI Risk Repository file: {os.path.basename(file_path)}\n\nThis file could not be processed due to an error.",
                            metadata={
                                "source": str(file_path),
                                "file_type": "error_fallback",
                                "title": os.path.basename(file_path)
                            }
                        )
                        self.all_documents.append(fallback_doc)
                    except Exception:
                        # If even the fallback fails, log and continue
                        logger.critical(f"Fallback document creation failed for {file_path}")
            
            # Summary logging
            logger.info(f"Document processing summary: {files_processed} processed, {files_failed} with errors")
            logger.info(f"Loaded {len(self.all_documents)} total documents")
            
            # If we have no documents, create a default document
            if not self.all_documents:
                logger.warning("No documents were successfully loaded. Creating a default document.")
                default_doc = Document(
                    page_content="MIT AI Risk Repository Information\n\nThe repository contains information about various AI risks across different domains including discrimination, privacy, misinformation, malicious use, human-computer interaction, socioeconomic impacts, and system safety.",
                    metadata={
                        "source": "default",
                        "file_type": "default",
                        "title": "MIT AI Risk Repository Overview"
                    }
                )
                self.all_documents.append(default_doc)
            
            # Process the documents for both vector and keyword search
            try:
                self._process_documents_for_retrieval()
            except Exception as retrieval_err:
                logger.error(f"Error processing documents for retrieval: {str(retrieval_err)}")
                # We'll still return True since we loaded documents, even if the retrieval processing failed
            
            return True
            
        except Exception as e:
            logger.error(f"Error ingesting documents: {str(e)}")
            
            # Create a minimal default document as fallback if everything else fails
            try:
                if not self.all_documents:
                    default_doc = Document(
                        page_content="MIT AI Risk Repository Information\n\nThe repository contains information about various AI risks across different domains including discrimination, privacy, misinformation, malicious use, human-computer interaction, socioeconomic impacts, and system safety.",
                        metadata={
                            "source": "default",
                            "file_type": "default",
                            "title": "MIT AI Risk Repository Overview"
                        }
                    )
                    self.all_documents.append(default_doc)
                    
                    # Try to process for retrieval
                    try:
                        self._process_documents_for_retrieval()
                        return True
                    except Exception:
                        # If even this fails, we have to return False
                        return False
            except Exception:
                pass
                
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
            try:
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
            except Exception as load_err:
                logger.error(f"Error loading vector store: {str(load_err)}")
                # Continue without vector store - will use fallback
        
        if not self.vector_store:
            logger.error("Vector store not initialized. Using fallback documents.")
            return self._get_fallback_documents(query)
        
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
            
            # Detect domain for domain-specific handling
            domain = self._detect_domain_in_query(query)
            if domain and self.structured_data:
                logger.info(f"Detected domain: {domain}, fetching domain-specific documents")
                domain_docs = self._get_domain_related_docs(domain)
                
                # Combine and deduplicate results
                existing_sources = {(doc.metadata.get('source', ''), doc.metadata.get('row', '')) for doc in docs}
                for doc in domain_docs:
                    source_key = (doc.metadata.get('source', ''), doc.metadata.get('row', ''))
                    if source_key not in existing_sources and len(docs) < k * 2:
                        docs.append(doc)
                        existing_sources.add(source_key)
                        
                logger.info(f"Added {len(domain_docs)} domain-specific documents for {domain}")
            
            # If we got results, apply citations and return
            if docs:
                # Apply citations/references to documents
                docs = self._add_citations_to_docs(docs)
                
                # Cache the results
                self.query_cache[cache_key] = (current_time, docs)
                
                return docs
            else:
                logger.warning("No documents found in vector store. Using fallback documents.")
                return self._get_fallback_documents(query)
                
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            return self._get_fallback_documents(query)
    
    def _get_fallback_documents(self, query: str) -> List[Document]:
        """
        Get fallback documents when vector search fails.
        Creates general information documents based on the query topic.
        """
        logger.info("Using fallback documents for query")
        fallback_docs = []
        
        # Detect domain in query for relevant fallback
        domain = self._detect_domain_in_query(query)
        
        # Start with a general repository document
        general_doc = Document(
            page_content=(
                "MIT AI Risk Repository Information\n\n"
                "The MIT AI Risk Repository is a comprehensive database containing over 1600 AI risks "
                "categorized across different domains including discrimination, privacy, misinformation, "
                "malicious use, human-computer interaction, socioeconomic impacts, and system safety."
            ),
            metadata={
                "source": "fallback",
                "file_type": "default",
                "title": "MIT AI Risk Repository Overview",
                "citation": "MIT AI Risk Repository"
            }
        )
        fallback_docs.append(general_doc)
        
        # Add domain-specific fallback if detected
        if domain:
            domain_info = {
                "discrimination": (
                    "AI Discrimination & Fairness Risks\n\n"
                    "The repository tracks risks related to bias, unfairness, and discrimination in AI systems. "
                    "Common issues include demographic biases in facial recognition, hiring algorithms that "
                    "discriminate against certain groups, and language models that perpetuate stereotypes. "
                    "Key concerns include representational harm, allocation harm, quality-of-service disparities, "
                    "and denigration."
                ),
                "privacy": (
                    "AI Privacy & Security Risks\n\n"
                    "The repository documents privacy violations related to AI systems, including data breaches, "
                    "surveillance concerns, model inversion attacks, membership inference attacks, and data poisoning. "
                    "Key issues include unauthorized data collection, insecure storage practices, and insufficient "
                    "anonymization techniques."
                ),
                "misinformation": (
                    "AI Misinformation Risks\n\n"
                    "The repository tracks AI's role in generating and spreading misinformation. This includes "
                    "deepfakes, synthetic media generation, automated propaganda, and AI systems that can generate "
                    "convincing but false content. Social harms include decreased trust in institutions, election "
                    "interference, and public health misinformation."
                ),
                "malicious_use": (
                    "AI Misuse & Malicious Application Risks\n\n"
                    "The repository catalogs how AI can be weaponized or misused, including cybercrime automation, "
                    "targeted phishing attacks, malware optimization, physical attacks using autonomous systems, "
                    "and fraud enhancement. These risks cover both intentional and unintentional harmful applications."
                ),
                "human_computer": (
                    "Human-Computer Interaction Risks\n\n"
                    "The repository documents issues related to how humans interact with AI systems. This includes "
                    "psychological manipulation, addiction mechanisms, over-reliance on automation, lack of transparency "
                    "in decision-making, and user interface issues that obscure AI limitations."
                ),
                "employment": (
                    "AI Employment & Economic Impacts\n\n"
                    "The repository tracks how AI affects labor markets and economic systems. Key concerns include "
                    "job displacement, changing skill requirements, wage depression, wealth concentration, and "
                    "economic inequality. Both short-term disruption and long-term structural changes are considered."
                ),
                "safety": (
                    "AI System Safety & Technical Limitations\n\n"
                    "The repository documents technical failures and safety issues, including reliability problems, "
                    "robustness failures, alignment challenges, control issues, and value specification challenges. "
                    "This domain encompasses both current AI system limitations and longer-term safety concerns."
                )
            }
            
            if domain in domain_info:
                domain_doc = Document(
                    page_content=domain_info[domain],
                    metadata={
                        "source": "fallback",
                        "file_type": "domain_fallback",
                        "title": f"{domain.capitalize()} Risk Information",
                        "domain": domain,
                        "citation": f"MIT AI Risk Repository - {domain.capitalize()} Domain"
                    }
                )
                fallback_docs.append(domain_doc)
        
        # Add structured data examples if available
        if hasattr(self, 'structured_data') and self.structured_data:
            # Try to get some representative examples (up to 3)
            sample_entries = []
            for i, entry in enumerate(self.structured_data):
                if i > 10:  # Only look at first few entries
                    break
                    
                # Format as a document
                entry_text = "Example Risk Entry:\n"
                for key, value in entry.items():
                    if not key.startswith('_') and value and isinstance(value, str):
                        entry_text += f"{key}: {value}\n"
                
                sample_entry = Document(
                    page_content=entry_text,
                    metadata={
                        "source": entry.get("_source", "unknown"),
                        "file_type": "structured_sample",
                        "title": "Example Risk Entry",
                        "citation": "Example from MIT AI Risk Repository"
                    }
                )
                sample_entries.append(sample_entry)
                
                if len(sample_entries) >= 3:
                    break
            
            # Add found samples
            fallback_docs.extend(sample_entries[:1])  # Only add 1 sample to avoid confusion
        
        # Apply citations
        fallback_docs = self._add_citations_to_docs(fallback_docs)
        
        return fallback_docs
    
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
    
    def _detect_domain_in_query(self, query: str) -> str:
        """
        Detect which domain a query likely refers to.
        Returns the domain name or None if no specific domain detected.
        """
        # Define domains and their keywords
        domains = {
            "discrimination": ["bias", "discrim", "fairness", "unfair", "prejudice", "equity"],
            "privacy": ["privacy", "data protection", "surveillance", "personal information", "anonymity", "confidential"],
            "misinformation": ["misinformation", "disinformation", "fake", "false", "misleading", "propaganda"],
            "malicious_use": ["malicious", "misuse", "weaponiz", "attack", "adversarial", "harmful", "criminal"],
            "human_computer": ["interface", "interaction", "human-computer", "usability", "design", "user experience"],
            "employment": ["job", "employ", "unemploy", "labor", "labour", "work", "worker", "workforce", 
                         "displacement", "income", "wage", "salary", "automation", "economic inequality"],
            "safety": ["safety", "accident", "harm", "dangerous", "physical", "injury", "death", "catastrophic"]
        }
        
        # Check query against each domain
        lower_query = query.lower()
        matched_domains = []
        
        for domain, keywords in domains.items():
            match_count = sum(1 for keyword in keywords if keyword in lower_query)
            if match_count > 0:
                matched_domains.append((domain, match_count))
        
        # Sort by number of matches (highest first)
        matched_domains.sort(key=lambda x: x[1], reverse=True)
        
        # Return the best match if any
        if matched_domains:
            return matched_domains[0][0]
            
        # No specific domain detected
        return None
    
    def _get_domain_related_docs(self, domain: str) -> List[Document]:
        """
        Get documents related to a specific domain from structured data.
        """
        docs = []
        domain_keywords = {
            "discrimination": ["bias", "discrimination", "fairness", "unfair", "prejudice", "equity"],
            "privacy": ["privacy", "data protection", "surveillance", "personal data", "confidential"],
            "misinformation": ["misinformation", "disinformation", "fake", "false", "misleading"],
            "malicious_use": ["malicious", "misuse", "weaponization", "attack", "adversarial", "harmful"],
            "human_computer": ["interface", "interaction", "human-computer", "usability", "design"],
            "employment": ["job", "employment", "unemployment", "labor", "work", "economic", "inequality"],
            "safety": ["safety", "accident", "harm", "dangerous", "physical", "injury"]
        }
        
        # Get keywords for the requested domain
        keywords = domain_keywords.get(domain, [])
        if not keywords:
            return []
            
        # Find entries with matching domain or relevant content
        for entry in self.structured_data:
            is_relevant = False
            
            # First check if the entry has an explicit domain field
            if "_domain" in entry and isinstance(entry["_domain"], str):
                domain_value = entry["_domain"].lower()
                # Check if any domain keyword matches
                if any(keyword in domain_value for keyword in keywords):
                    is_relevant = True
            
            # If not found by domain field, check for keywords in content fields
            if not is_relevant:
                # Check all non-metadata fields for keywords
                for key, value in entry.items():
                    if key.startswith('_'):  # Skip metadata fields
                        continue
                    if value and isinstance(value, str):
                        value_lower = value.lower()
                        if any(keyword in value_lower for keyword in keywords):
                            is_relevant = True
                            break
            
            if is_relevant:
                # Format entry as a document
                content = f"{domain.capitalize()}-Related Risk Entry:\n"
                for key, value in entry.items():
                    if key.startswith('_'):  # Skip metadata fields
                        continue
                    if value and isinstance(value, str):
                        content += f"{key}: {value}\n"
                
                doc = Document(
                    page_content=content,
                    metadata={
                        "source": entry.get("_source", "structured_data"),
                        "sheet": entry.get("_sheet", "unknown"),
                        "row": entry.get("_row", 0),
                        "domain": entry.get("_domain", domain),
                        "category": entry.get("_category", "Unspecified"),
                        "file_type": f"{domain}_data"
                    }
                )
                docs.append(doc)
        
        return docs
        
    def _is_employment_related(self, query: str) -> bool:
        """Check if a query is related to employment or labor issues."""
        detected_domain = self._detect_domain_in_query(query)
        return detected_domain == "employment"
    
    def _get_employment_related_docs(self) -> List[Document]:
        """Get documents specifically related to employment from structured data."""
        return self._get_domain_related_docs("employment")
    
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