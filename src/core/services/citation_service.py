"""
Citation service for handling document references and snippets.
"""
import os
import hashlib
import re
from typing import List
from langchain.docstore.document import Document

from ...config.logging import get_logger
from ...config.settings import settings

logger = get_logger(__name__)

class CitationService:
    """Handles citation generation and snippet management."""
    
    def __init__(self):
        self.snippets_dir = settings.DOC_SNIPPETS_DIR
        # Ensure snippets directory exists
        self.snippets_dir.mkdir(parents=True, exist_ok=True)
    
    def enhance_response_with_citations(self, response: str, docs: List[Document]) -> str:
        """
        Add clickable citations to the response text.
        
        Args:
            response: Generated response text
            docs: Source documents
            
        Returns:
            Response with enhanced citations
        """
        if not docs:
            return response
        
        # Create mapping of docs to their citations
        doc_citations = {}
        for i, doc in enumerate(docs):
            citation = self._format_document_citation(doc)
            
            # Map various reference patterns to citations
            doc_citations[f"SECTION {i+1}"] = citation
            doc_citations[f"Source {i+1}"] = citation
            doc_citations[f"Document {i+1}"] = citation
            doc_citations[f"Entry {i+1}"] = citation
            
            # Also add by filename if available
            if 'source' in doc.metadata:
                filename = os.path.basename(doc.metadata['source'])
                doc_citations[filename] = citation
                
                # Add the mangled version that actually appears in responses
                mangled_name = self._clean_for_filename(filename).replace('-', ' ').title()
                doc_citations[mangled_name] = citation
        
        # Replace section references with clickable citations
        enhanced_response = response
        logger.info(f"Processing citations for {len(doc_citations)} documents")
        logger.info(f"Available patterns: {list(doc_citations.keys())}")
        logger.info(f"Response snippet: {response[:200]}...")
        
        for pattern, citation in doc_citations.items():
            # Check if pattern exists in response
            if pattern in enhanced_response:
                logger.info(f"✓ Replacing pattern '{pattern}' with citation")
                enhanced_response = enhanced_response.replace(f"[{pattern}]", citation)
                enhanced_response = enhanced_response.replace(pattern, citation)
            else:
                logger.debug(f"✗ Pattern '{pattern}' not found in response")
        
        logger.info(f"Citation enhancement complete. Changed: {response != enhanced_response}")
        return enhanced_response
    
    def _format_document_citation(self, doc: Document) -> str:
        """Format a citation for a document based on its type and metadata."""
        if not doc or not hasattr(doc, 'metadata'):
            return "[Unknown Source]"
        
        # Get basic metadata
        file_type = doc.metadata.get('file_type', '')
        source = doc.metadata.get('source', 'Unknown')
        
        # Create a unique ID for this document reference
        doc_id = hashlib.md5(f"{source}_{file_type}".encode()).hexdigest()[:8]
        
        # Save snippet for reference
        self._save_document_snippet(doc, doc_id)
        
        # Create appropriate citation based on file type
        if 'excel' in file_type.lower():
            return self._format_excel_citation(doc, doc_id)
        elif doc.metadata.get('citation'):
            # Use existing citation if available but add link
            citation = doc.metadata.get('citation')
            return f"[{citation}](/snippet/{doc_id})"
        else:
            # Create a clean generic citation
            filename = os.path.basename(source)
            if 'AI_Risk' in filename or 'ai_risk' in filename.lower():
                citation_text = "AI Risk Repository Document"
            else:
                # Clean filename for display
                clean_name = filename.replace('_', ' ').replace('-', ' ')
                if len(clean_name) > 30:
                    clean_name = clean_name[:30] + "..."
                citation_text = clean_name
            
            return f"[{citation_text}](/snippet/{doc_id})"
    
    def _format_excel_citation(self, doc: Document, doc_id: str) -> str:
        """Format a citation specifically for Excel files with sheet and row info."""
        sheet_name = doc.metadata.get('sheet', 'Unknown Sheet')
        row = doc.metadata.get('row', None)
        
        # Create clean, readable citation text
        if 'AI Risk Database' in sheet_name:
            if row is not None:
                citation_text = f"AI Risk Repository, Row {row}"
            else:
                citation_text = "AI Risk Repository"
        else:
            if row is not None:
                citation_text = f"MIT AI Repository, {sheet_name}, Row {row}"
            else:
                citation_text = f"MIT AI Repository, {sheet_name}"
        
        return f"[{citation_text}](/snippet/{doc_id})"
    
    def _save_document_snippet(self, doc: Document, doc_id: str) -> None:
        """Save document snippet for later reference."""
        snippet_path = self.snippets_dir / f"doc_{doc_id}.txt"
        
        try:
            with open(snippet_path, 'w') as f:
                f.write(f"Source: {doc.metadata.get('source', 'Unknown')}\\n")
                
                # Add all metadata
                for key, value in doc.metadata.items():
                    if key not in ['source', 'page_content']:
                        f.write(f"{key}: {value}\\n")
                
                f.write(f"\\nContent:\\n{doc.page_content}")
        except Exception as e:
            logger.error(f"Error saving document snippet: {str(e)}")
    
    def _clean_for_filename(self, text: str, max_length: int = 50) -> str:
        """Clean a string to make it suitable for use as a filename."""
        # Remove invalid filename characters
        clean = re.sub(r'[\\\\/*?:"<>|]', '', text)
        # Replace spaces and underscores with hyphens
        clean = re.sub(r'[\\s_]+', '-', clean)
        # Truncate to reasonable length
        return clean[:max_length]
    
    def get_snippet_content(self, snippet_id: str) -> str:
        """Get snippet content by ID."""
        snippet_path = self.snippets_dir / f"doc_{snippet_id}.txt"
        
        if snippet_path.exists():
            try:
                with open(snippet_path, 'r') as f:
                    return f.read()
            except Exception as e:
                logger.error(f"Error reading snippet: {str(e)}")
                return f"Error reading snippet: {str(e)}"
        else:
            return "Snippet not found"