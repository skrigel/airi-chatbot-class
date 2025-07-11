"""
Query processing and enhancement for domain-specific searches.
"""
from typing import List, Dict, Any, Tuple, Optional
from langchain.docstore.document import Document

from ...config.logging import get_logger

logger = get_logger(__name__)

class QueryProcessor:
    """Handles query analysis and processing."""
    
    def __init__(self, query_monitor=None):
        self.query_monitor = query_monitor
        
        # Import domain classifier for generic domain detection
        from ...config.domains import domain_classifier
        from ...config.prompts import prompt_manager
        self.domain_classifier = domain_classifier
        self.prompt_manager = prompt_manager
    
    def analyze_query(self, message: str) -> Tuple[str, Optional[str]]:
        """
        Analyze query to determine type and domain.
        
        Args:
            message: User query
            
        Returns:
            Tuple of (query_type, domain)
        """
        query_type = "general"
        domain = None
        
        # Try advanced analysis first
        if self.query_monitor:
            try:
                if hasattr(self.query_monitor, 'analyze_query'):
                    query_analysis = self.query_monitor.analyze_query(message)
                    query_type = query_analysis.get('query_type', 'general')
                elif hasattr(self.query_monitor, 'determine_inquiry_type'):
                    inquiry_result = self.query_monitor.determine_inquiry_type(message)
                    query_type = inquiry_result.get('inquiry_type', 'GENERAL').lower()
                    domain = inquiry_result.get('primary_domain', 'OTHER').lower()
                    
                    # Map inquiry types to our internal types
                    if query_type == "employment_risk":
                        query_type = "employment"
                
                logger.info(f"Query type detected: {query_type}")
                if domain:
                    logger.info(f"Domain detected: {domain}")
            except Exception as e:
                logger.error(f"Error analyzing query: {str(e)}")
        
        # Enhanced domain-based detection using generic classifier
        if not domain or domain == "other":
            domain = self.domain_classifier.classify_domain(message)
            if domain != "other":
                query_type = domain
                logger.info(f"Enhanced detection found {domain} related query")
        
        return query_type, domain
    
    def enhance_query(self, message: str, query_type: str) -> str:
        """
        Enhance query with additional keywords for better retrieval.
        
        Args:
            message: Original query
            query_type: Detected query type
            
        Returns:
            Enhanced query string
        """
        enhanced_query = message
        
        # Use generic domain keywords for enhancement
        if query_type != "general":
            domain_keywords = self.domain_classifier.get_domain_keywords(query_type)
            if domain_keywords:
                enhanced_query += " " + " ".join(domain_keywords[:6])  # Limit to first 6 keywords
        
        return enhanced_query
    
    def filter_documents_by_relevance(self, docs: List[Document], query_type: str) -> List[Document]:
        """
        Filter and prioritize documents based on query type.
        
        Args:
            docs: Retrieved documents
            query_type: Query type
            
        Returns:
            Filtered and prioritized documents
        """
        # Generic domain-based document filtering
        if query_type == "general" or not self.domain_classifier.is_domain_enabled(query_type):
            return docs
        
        domain_docs = []
        other_docs = []
        
        # Get domain keywords for matching
        domain_keywords = self.domain_classifier.get_domain_keywords(query_type)
        
        for doc in docs:
            doc_domain = doc.metadata.get('domain', '').lower()
            doc_subdomain = doc.metadata.get('subdomain', '').lower()
            doc_specific_domain = doc.metadata.get('specific_domain', '').lower()
            
            # Check if this document is domain-related
            is_domain_related = any(
                keyword in doc_domain + doc_subdomain + doc_specific_domain 
                for keyword in domain_keywords
            )
            
            if is_domain_related:
                domain_docs.append(doc)
            else:
                other_docs.append(doc)
        
        # Prioritize domain docs, but include some others for context
        filtered_docs = domain_docs[:6] + other_docs[:2]
        
        logger.info(f"Filtered to {len(domain_docs)} {query_type}-specific documents and {min(2, len(other_docs))} general documents")
        
        return filtered_docs
    
    def generate_prompt(self, message: str, query_type: str, context: str, session_id: str = "default", docs: List[Document] = None) -> str:
        """
        Generate enhanced prompt using the new prompt management system.

        Args:
            message: User query
            query_type: Detected query type  
            context: Retrieved context
            session_id: Session ID for intro tracking
            docs: Retrieved documents for RID extraction
            
        Returns:
            Enhanced prompt with brevity rules and domain-specific guidance
        """
        # Detect domain from query type or use domain classifier
        domain = query_type if query_type in ['socioeconomic', 'safety', 'privacy', 'bias', 'governance', 'technical'] else 'general'
        if domain == 'general':
            # Try to detect domain using classifier
            detected_domain = self.domain_classifier.classify_domain(message)
            if detected_domain != 'other':
                domain = detected_domain
        
        # Extract available RIDs from documents
        available_rids = []
        if docs:
            for doc in docs:
                rid = doc.metadata.get('rid')
                if rid and rid not in available_rids:
                    available_rids.append(rid)
        
        # Use the new prompt manager for advanced, context-aware prompts
        return self.prompt_manager.get_prompt(
            query=message,
            domain=domain,
            context=context,
            session_id=session_id,
            query_type=query_type,
            available_rids=available_rids
        )