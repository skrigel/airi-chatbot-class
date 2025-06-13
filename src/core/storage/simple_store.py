"""
A simplified vector store implementation that uses keyword search instead of embeddings.
This is a workaround for the embedding model error.
"""

import os
import logging
import json
import re
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
from langchain.docstore.document import Document

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleVectorStore:
    """
    A simple vector store that uses keyword matching instead of embeddings.
    This is a temporary solution until the embedding issue is fixed.
    """
    
    def __init__(self, repository_path=None):
        """
        Initialize the simple vector store.
        
        Args:
            repository_path (str): Path to the directory containing documents
        """
        self.repository_path = repository_path
        self.documents = []
        
        # If repository path is provided, ingest documents
        if repository_path:
            self.ingest_documents()
    
    def ingest_documents(self):
        """Ingest documents from the repository path."""
        if not self.repository_path or not os.path.exists(self.repository_path):
            logger.error(f"Repository path {self.repository_path} does not exist")
            return False
        
        try:
            # Process all files in the repository
            logger.info(f"Loading documents from {self.repository_path}")
            
            self.documents = []
            base_path = Path(self.repository_path)
            
            # Priority for Excel files
            excel_files = []
            text_files = []
            
            # Collect all files
            for file_path in base_path.glob("**/*"):
                if not file_path.is_file() or file_path.name.startswith('.'):
                    continue
                
                if file_path.suffix.lower() in ['.xlsx', '.xls']:
                    excel_files.append(file_path)
                else:
                    text_files.append(file_path)
            
            # Process Excel files first
            for file_path in excel_files:
                logger.info(f"Processing Excel file: {file_path}")
                try:
                    # Try to read Excel file
                    try:
                        excel_data = pd.read_excel(file_path, sheet_name=None)
                        logger.info(f"Successfully read Excel file with {len(excel_data)} sheets")
                        
                        # Process each sheet
                        for sheet_name, df in excel_data.items():
                            if df.empty:
                                continue
                                
                            # Process each row
                            for i, row in df.iterrows():
                                # Create a document for this row
                                content = f"Excel Row from {sheet_name}:\n"
                                for col in df.columns:
                                    if not pd.isna(row[col]) and str(row[col]).strip():
                                        content += f"{col}: {row[col]}\n"
                                
                                doc = Document(
                                    page_content=content,
                                    metadata={
                                        "source": str(file_path),
                                        "sheet": sheet_name,
                                        "row": i,
                                        "file_type": "excel"
                                    }
                                )
                                self.documents.append(doc)
                            
                    except Exception as excel_err:
                        logger.error(f"Error reading Excel file {file_path}: {str(excel_err)}")
                except Exception as e:
                    logger.error(f"Error processing file {file_path}: {str(e)}")
            
            # Process text files
            for file_path in text_files:
                logger.info(f"Processing text file: {file_path}")
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Split into manageable chunks
                    chunks = self._split_text(content)
                    
                    for i, chunk in enumerate(chunks):
                        doc = Document(
                            page_content=chunk,
                            metadata={
                                "source": str(file_path),
                                "chunk": i,
                                "file_type": "text"
                            }
                        )
                        self.documents.append(doc)
                        
                except Exception as e:
                    logger.error(f"Error processing file {file_path}: {str(e)}")
            
            logger.info(f"Loaded {len(self.documents)} total documents")
            return True
            
        except Exception as e:
            logger.error(f"Error ingesting documents: {str(e)}")
            return False
    
    def _split_text(self, text, chunk_size=1000, overlap=100):
        """Split text into chunks with overlap."""
        chunks = []
        
        # If text is shorter than chunk size, return as is
        if len(text) <= chunk_size:
            return [text]
        
        # Split by paragraphs first
        paragraphs = text.split('\n\n')
        
        current_chunk = ""
        for para in paragraphs:
            # If adding this paragraph would exceed chunk size
            if len(current_chunk) + len(para) > chunk_size and current_chunk:
                chunks.append(current_chunk)
                # Keep overlap from previous chunk
                overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else ""
                current_chunk = overlap_text + para + "\n\n"
            else:
                current_chunk += para + "\n\n"
        
        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def get_relevant_documents(self, query, k=5):
        """Find the most relevant documents for a query using keyword matching."""
        if not self.documents:
            logger.warning("No documents in the store")
            return []
        
        # Extract keywords from the query
        keywords = self._extract_keywords(query)
        
        # Score documents based on keyword matches
        scored_docs = []
        for doc in self.documents:
            score = self._score_document(doc, keywords, query)
            scored_docs.append((score, doc))
        
        # Sort by score (highest first) and take top k
        scored_docs.sort(reverse=True)
        return [doc for _, doc in scored_docs[:k]]
    
    def _extract_keywords(self, query):
        """Extract keywords from a query with expansion for common AI risk topics."""
        # Remove common stop words
        stop_words = {'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 
                     'to', 'of', 'and', 'or', 'in', 'on', 'at', 'for', 'with', 'by', 'about',
                     'against', 'between', 'into', 'through', 'during', 'before', 'after',
                     'above', 'below', 'from', 'up', 'down', 'what', 'which', 'who', 'whom',
                     'this', 'that', 'these', 'those', 'how', 'when', 'where', 'why', 'have', 'has', 'had',
                     'do', 'does', 'did', 'can', 'could', 'will', 'would', 'shall', 'should', 'may', 'might'}
        
        # Topic-based synonym/expansion map for key AI risk areas
        expansion_map = {
            # Employment/job-related terms
            "job": ["employment", "work", "labor", "workforce", "occupation", "career"],
            "employment": ["job", "work", "labor", "workforce", "occupation", "career"],
            "work": ["job", "employment", "labor", "workforce"],
            "jobs": ["employment", "work", "labor", "workforce", "occupation", "career"],
            "worker": ["employee", "laborer", "workforce"],
            "workers": ["employees", "laborers", "workforce"],
            "unemployment": ["joblessness", "job loss", "layoff"],
            
            # Harm/dangerous content
            "harmful": ["toxic", "dangerous", "unsafe", "inappropriate", "offensive"],
            "toxic": ["harmful", "dangerous", "unsafe", "inappropriate", "offensive"],
            "unsafe": ["harmful", "dangerous", "toxic", "inappropriate"],
            "offensive": ["harmful", "toxic", "inappropriate", "objectionable"],
            "inappropriate": ["harmful", "offensive", "objectionable", "improper"],
            
            # Language-related terms
            "language": ["text", "content", "speech", "output", "generation"],
            "content": ["language", "text", "material", "information"],
            "text": ["language", "content", "writing"],
            "generation": ["creation", "production", "output"],
            
            # Bias/discrimination
            "bias": ["discrimination", "prejudice", "unfairness", "partiality"],
            "discrimination": ["bias", "prejudice", "unfairness", "inequality"],
            "fairness": ["equality", "impartiality", "unbiased", "equitable"],
            
            # Privacy/security
            "privacy": ["confidentiality", "data protection", "surveillance"],
            "security": ["protection", "safety", "safeguarding"],
            
            # Misinformation
            "misinformation": ["fake news", "disinformation", "false information", "propaganda"],
            "fake": ["false", "misleading", "deceptive", "untrue"]
        }
        
        # Basic phrase detection for common multi-word concepts
        common_phrases = {
            "job loss": ["employment impact", "workforce displacement", "automation effects", "labor displacement"],
            "harmful language": ["toxic text", "offensive content", "inappropriate output", "unsafe content"],
            "language model": ["llm", "large language model", "text generation model"],
            "harmful content": ["toxic content", "offensive material", "inappropriate information"],
            "ai bias": ["algorithmic bias", "model bias", "machine learning bias"],
            "ai safety": ["artificial intelligence safety", "model safety"],
            "data privacy": ["information privacy", "personal data protection"]
        }
        
        query_lower = query.lower()
        expanded_keywords = []
        
        # Check query for phrases
        for phrase, alternatives in common_phrases.items():
            if phrase in query_lower:
                expanded_keywords.append(phrase)
                # Add alternative phrasings
                expanded_keywords.extend(alternatives)
        
        # Extract base keywords
        words = re.findall(r'\b\w+\b', query_lower)
        base_keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Add base keywords to expanded list
        expanded_keywords.extend(base_keywords)
        
        # Add synonyms and related terms
        for word in base_keywords:
            if word in expansion_map:
                expanded_keywords.extend(expansion_map[word])
        
        # Special handling for common problem cases
        
        # Special case 1: Job loss and employment effects
        if any(term in query_lower for term in ["job", "employ", "work", "labor", "career"]):
            expanded_keywords.extend(["job", "employment", "work", "labor", "workforce", "occupation", 
                                     "automation", "replacement", "displacement", "economy"])
            
        # Special case 2: Harmful language generation
        if ("harmful" in query_lower or "toxic" in query_lower or "offensive" in query_lower) and \
           ("language" in query_lower or "content" in query_lower or "text" in query_lower or "generation" in query_lower):
            expanded_keywords.extend(["harmful", "toxic", "dangerous", "unsafe", "inappropriate", "offensive",
                                    "language", "content", "text", "generation", "output", "speech"])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_keywords = []
        for keyword in expanded_keywords:
            if keyword not in seen:
                seen.add(keyword)
                unique_keywords.append(keyword)
        
        logger.info(f"Expanded keywords: {unique_keywords} from query: '{query}'")
        return unique_keywords
    
    def _score_document(self, doc, keywords, original_query):
        """Score a document based on keyword matches with improved scoring logic."""
        content = doc.page_content.lower()
        score = 0
        
        # Exact query match (highest score)
        if original_query.lower() in content:
            score += 100
            logger.info(f"Exact query match: +100 points for document {doc.metadata.get('source', 'unknown')}")
        
        # Check for matches of multi-word phrases from the original query (high value)
        query_tokens = original_query.lower().split()
        if len(query_tokens) >= 2:
            # Look for bigrams and trigrams from the query
            for i in range(len(query_tokens) - 1):
                bigram = f"{query_tokens[i]} {query_tokens[i+1]}"
                if bigram in content:
                    score += 40
                    logger.info(f"Bigram match '{bigram}': +40 points")
            
            # Try trigrams if available
            if len(query_tokens) >= 3:
                for i in range(len(query_tokens) - 2):
                    trigram = f"{query_tokens[i]} {query_tokens[i+1]} {query_tokens[i+2]}"
                    if trigram in content:
                        score += 60
                        logger.info(f"Trigram match '{trigram}': +60 points")
        
        # Special case handling for job/employment questions
        if any(term in original_query.lower() for term in ["job", "employment", "work", "labor", "career"]):
            # Check for employment-related content
            employment_terms = ["job", "employment", "work", "labor", "workforce", "career", 
                               "economy", "economic", "automation", "worker", "occupation"]
            
            employment_matches = sum(1 for term in employment_terms if term in content)
            if employment_matches >= 3:
                score += 35
                logger.info(f"Employment relevance: +35 points ({employment_matches} employment terms)")
        
        # Special case handling for harmful language/content questions
        if ("harmful" in original_query.lower() or "toxic" in original_query.lower()) and \
           any(term in original_query.lower() for term in ["language", "content", "text", "generation"]):
            # Check for harmful language related content
            harmful_terms = ["harmful", "toxic", "offensive", "inappropriate", "language",
                            "content", "generation", "output", "text", "safety"]
            
            harmful_matches = sum(1 for term in harmful_terms if term in content)
            if harmful_matches >= 3:
                score += 35
                logger.info(f"Harmful content relevance: +35 points ({harmful_matches} harmful terms)")
        
        # Group keywords by importance
        primary_keywords = []
        secondary_keywords = []
        
        # Define importance tiers based on proximity to core concepts
        primary_concepts = ["job", "work", "employ", "labor", "harmful", "toxic", "language", 
                           "risk", "safety", "bias", "discriminat", "privacy", "misinformation"]
        
        # Categorize keywords into importance tiers
        for keyword in keywords:
            if any(concept in keyword for concept in primary_concepts):
                primary_keywords.append(keyword)
            else:
                secondary_keywords.append(keyword)
        
        # Score primary keywords (higher weight)
        for keyword in primary_keywords:
            count = content.count(keyword)
            if count > 0:
                # More matches = higher score with emphasis on important terms
                points = count * 8
                score += points
                
                # Bonus for matches in the first 200 chars (likely more important)
                if keyword in content[:200]:
                    score += 5
                
                # Additional bonus for matches in title or section headers
                lines = content.split('\n')
                for line in lines[:10]:  # Check early lines that might be headers
                    if len(line) < 100 and keyword in line.lower():
                        score += 3
                        break
        
        # Score secondary keywords (lower weight)
        for keyword in secondary_keywords:
            count = content.count(keyword)
            if count > 0:
                # More matches = higher score
                score += count * 3
                
                # Smaller bonus for matches in the first 200 chars
                if keyword in content[:200]:
                    score += 2
        
        # Content type bonuses
        
        # Bonus for Excel files with specific sheets (more structured content)
        if doc.metadata.get('file_type') == 'excel':
            score += 2
            
            # Extra points for row content that has risk-related columns
            if 'risk' in content.lower() or 'hazard' in content.lower():
                score += 5
        
        # Bonus for documents that focus on AI risks
        if 'risk' in content.lower() and ('ai' in content.lower() or 'artificial intelligence' in content.lower()):
            score += 10
            logger.info(f"AI risk focus: +10 points")
        
        # Penalize too-short documents that might be fragments
        if len(content) < 100:
            score *= 0.5
            logger.info(f"Short document penalty: score halved")
        
        logger.info(f"Final score: {score} for document {doc.metadata.get('source', 'unknown')}")
        return score
    
    def format_context_from_docs(self, docs):
        """Format docs into a context string."""
        if not docs:
            return ""
        
        context_parts = []
        context_parts.append("RELEVANT INFORMATION FROM THE MIT AI RISK REPOSITORY:\n")
        
        # Group docs by source
        docs_by_source = {}
        for doc in docs:
            source = doc.metadata.get('source', 'Unknown')
            if source not in docs_by_source:
                docs_by_source[source] = []
            docs_by_source[source].append(doc)
        
        # Format each source
        section_num = 1
        for source, source_docs in docs_by_source.items():
            filename = os.path.basename(source)
            
            if 'excel' in source.lower() or source_docs[0].metadata.get('file_type') == 'excel':
                # Excel entries
                context_parts.append(f"SECTION {section_num}: Repository Excel Entries")
                section_num += 1
                
                for i, doc in enumerate(source_docs, 1):
                    sheet = doc.metadata.get('sheet', 'Unknown Sheet')
                    row = doc.metadata.get('row', 'Unknown Row')
                    context_parts.append(f"Entry {i}:\n{doc.page_content}\n[Source: {filename}, Sheet: {sheet}, Row: {row}]\n")
            else:
                # Text chunks
                context_parts.append(f"SECTION {section_num}: {filename}")
                section_num += 1
                
                for i, doc in enumerate(source_docs):
                    chunk = doc.metadata.get('chunk', i)
                    context_parts.append(f"Excerpt {i+1}:\n{doc.page_content}\n[Source: {filename}, Excerpt: {chunk}]\n")
        
        return "\n\n".join(context_parts)

# Simple usage example for testing
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python simple_vector_store.py <repository_path>")
        sys.exit(1)
    
    repo_path = sys.argv[1]
    store = SimpleVectorStore(repo_path)
    
    print(f"Loaded {len(store.documents)} documents")
    
    # Test query
    test_query = "What are risks of AI in employment?"
    docs = store.get_relevant_documents(test_query, k=3)
    
    print(f"\nTop results for query: '{test_query}'")
    for i, doc in enumerate(docs, 1):
        print(f"\nDocument {i}:")
        print("-" * 40)
        if len(doc.page_content) > 300:
            print(doc.page_content[:300] + "...")
        else:
            print(doc.page_content)
        print("-" * 40)
        print(f"Source: {doc.metadata.get('source', 'unknown')}")
    
    # Format context
    context = store.format_context_from_docs(docs)
    print("\nExample context formatting:")
    print(context[:500] + "..." if len(context) > 500 else context)