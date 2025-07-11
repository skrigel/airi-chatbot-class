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
    """Handles citation generation and snippet management with RID-based citations."""
    
    def __init__(self):
        self.snippets_dir = settings.DOC_SNIPPETS_DIR
        # Ensure snippets directory exists
        self.snippets_dir.mkdir(parents=True, exist_ok=True)
        
        # RID to citation mapping for deterministic citation replacement
        self.rid_citation_map = {}
    
    def enhance_response_with_citations(self, response: str, docs: List[Document]) -> str:
        """
        Add RID-based citations to the response text with inline highlighting.
        
        Args:
            response: Generated response text
            docs: Source documents
            
        Returns:
            Response with enhanced RID-based citations and highlighting
        """
        if not docs:
            return response
        
        # Build RID citation mapping
        self.rid_citation_map = {}
        for doc in docs:
            rid = doc.metadata.get('rid', None)
            if rid:
                citation = self._format_rid_citation(doc)
                self.rid_citation_map[rid] = citation
                
                # Save snippet for the RID
                self._save_rid_snippet(doc, rid)
        
        # Replace RID placeholders and legacy patterns
        enhanced_response = self._replace_rid_citations(response, docs)
        
        # Add inline highlighting for supported claims
        enhanced_response = self._add_inline_highlighting(enhanced_response, docs)
        
        logger.info(f"RID citation enhancement complete. RIDs processed: {len(self.rid_citation_map)}")
        return enhanced_response
    
    def _replace_rid_citations(self, response: str, docs: List[Document]) -> str:
        """Replace RID placeholders and legacy section references with proper citations."""
        enhanced_response = response
        
        # First, replace any RID-##### patterns with proper citations
        import re
        rid_pattern = r'RID-(\d{5})'
        
        def replace_rid(match):
            rid = match.group(0)
            return self.rid_citation_map.get(rid, rid)  # Keep original if not found
        
        enhanced_response = re.sub(rid_pattern, replace_rid, enhanced_response)
        
        # Handle legacy patterns (SECTION X, Document X) for backward compatibility
        for i, doc in enumerate(docs, 1):
            rid = doc.metadata.get('rid')
            if not rid:
                continue
                
            citation = self.rid_citation_map.get(rid, f"[Source {i}]")
            
            # Replace various legacy patterns
            patterns = [
                f"SECTION {i}",
                f"Source {i}",
                f"Document {i}",
                f"Entry {i}"
            ]
            
            for pattern in patterns:
                if pattern in enhanced_response:
                    enhanced_response = enhanced_response.replace(pattern, citation)
                    logger.info(f"✓ Replaced legacy pattern '{pattern}' with RID citation")
        
        # If no citations were added through pattern replacement, append them
        if self.rid_citation_map and not any(rid in enhanced_response for rid in self.rid_citation_map.keys()):
            logger.info("No citation patterns found - appending citations to response")
            citations_list = []
            for rid, citation in self.rid_citation_map.items():
                citations_list.append(f"{rid}: {citation}")
            
            if citations_list:
                enhanced_response += "\n\n**Sources:**\n" + "\n".join(f"• {cite}" for cite in citations_list)
                logger.info(f"✓ Appended {len(citations_list)} citations to response")
        
        return enhanced_response
    
    def _add_inline_highlighting(self, response: str, docs: List[Document]) -> str:
        """Add bold highlighting to phrases that directly reference source content."""
        # For now, implement basic highlighting
        # Future enhancement: use fuzzy matching to find exact phrases from sources
        return response  # Placeholder for inline highlighting feature
    
    def _format_rid_citation(self, doc: Document) -> str:
        """Format a citation using the document's RID."""
        rid = doc.metadata.get('rid', 'RID-UNKNOWN')
        file_type = doc.metadata.get('file_type', '')
        
        # Create human-readable citation text based on document type
        if 'ai_risk_entry' in file_type:
            title = doc.metadata.get('title', 'AI Risk Entry')
            domain = doc.metadata.get('domain', '')
            if domain and domain != 'Unspecified':
                citation_text = f"{title} (Domain: {domain})"
            else:
                citation_text = title
        elif 'domain_summary' in file_type:
            domain = doc.metadata.get('domain', 'Unknown Domain')
            citation_text = f"AI Risk Domain: {domain}"
        elif 'excel' in file_type:
            sheet = doc.metadata.get('sheet', 'Unknown Sheet')
            row = doc.metadata.get('row', '')
            if row:
                citation_text = f"MIT AI Repository, {sheet}, Row {row}"
            else:
                citation_text = f"MIT AI Repository, {sheet}"
        else:
            # Generic citation
            source = doc.metadata.get('source', 'Unknown Source')
            filename = os.path.basename(source)
            if 'AI_Risk' in filename:
                citation_text = "AI Risk Repository Document"
            else:
                citation_text = filename.replace('_', ' ').replace('-', ' ')[:30]
        
        return f"[{citation_text}](/snippet/{rid})"
    
    def _save_rid_snippet(self, doc: Document, rid: str) -> None:
        """Save document snippet using RID for easy retrieval."""
        snippet_path = self.snippets_dir / f"{rid}.txt"
        
        try:
            with open(snippet_path, 'w', encoding='utf-8') as f:
                f.write(f"Repository ID: {rid}\n")
                f.write(f"Source: {doc.metadata.get('source', 'Unknown')}\n")
                
                # Add all metadata
                for key, value in doc.metadata.items():
                    if key not in ['source', 'page_content', 'content_hash']:
                        f.write(f"{key}: {value}\n")
                
                f.write(f"\nContent:\n{doc.page_content}")
        except Exception as e:
            logger.error(f"Error saving RID snippet for {rid}: {str(e)}")
    
    def get_snippet_by_rid(self, rid: str) -> str:
        """Get snippet content by RID."""
        snippet_path = self.snippets_dir / f"{rid}.txt"
        
        if snippet_path.exists():
            try:
                with open(snippet_path, 'r', encoding='utf-8') as f:
                    return f.read()
            except Exception as e:
                logger.error(f"Error reading RID snippet {rid}: {str(e)}")
                return f"Error reading snippet {rid}: {str(e)}"
        else:
            return f"Snippet for {rid} not found"
    
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