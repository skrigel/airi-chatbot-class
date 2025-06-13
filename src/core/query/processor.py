"""
Query processing and enhancement for employment-related searches.
"""
from typing import List, Dict, Any, Tuple, Optional
from langchain.docstore.document import Document

from ...config.logging import get_logger

logger = get_logger(__name__)

class QueryProcessor:
    """Handles query analysis and processing."""
    
    def __init__(self, query_monitor=None):
        self.query_monitor = query_monitor
        
        # Keywords for different query types
        self.employment_keywords = [
            'job', 'employ', 'work', 'career', 'unemployment', 
            'labor', 'worker', 'workforce', 'occupation'
        ]
        self.socioeconomic_keywords = [
            'socioeconomic', 'economic', 'inequality', 'social', 'financial'
        ]
    
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
        
        # Enhanced keyword-based detection
        if not query_type or query_type == "general":
            message_lower = message.lower()
            if any(keyword in message_lower for keyword in self.employment_keywords):
                query_type = "employment"
                domain = "socioeconomic"
                logger.info("Enhanced detection found employment related query")
            elif any(keyword in message_lower for keyword in self.socioeconomic_keywords):
                query_type = "socioeconomic"
                domain = "socioeconomic"
                logger.info("Enhanced detection found socioeconomic related query")
        
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
        
        if query_type == "employment":
            enhanced_query += " employment job work career labor workforce inequality"
        elif query_type == "socioeconomic":
            enhanced_query += " socioeconomic economic social inequality"
        
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
        if query_type not in ["employment", "socioeconomic"]:
            return docs
        
        employment_docs = []
        other_docs = []
        
        for doc in docs:
            doc_domain = doc.metadata.get('domain', '').lower()
            doc_subdomain = doc.metadata.get('subdomain', '').lower()
            doc_specific_domain = doc.metadata.get('specific_domain', '').lower()
            
            # Check if this document is employment-related
            is_employment_related = any(
                keyword in doc_domain + doc_subdomain + doc_specific_domain 
                for keyword in ['employ', 'job', 'work', 'labor', 'socioeconomic', 'economic', 'inequality']
            )
            
            if is_employment_related:
                employment_docs.append(doc)
            else:
                other_docs.append(doc)
        
        # Prioritize employment docs, but include some others for context
        filtered_docs = employment_docs[:6] + other_docs[:2]
        
        logger.info(f"Filtered to {len(employment_docs)} employment-specific documents and {min(2, len(other_docs))} general documents")
        
        return filtered_docs
    
    def generate_prompt(self, message: str, query_type: str, context: str) -> str:
        """
        Generate enhanced prompt based on query type.
        
        Args:
            message: User query
            query_type: Detected query type  
            context: Retrieved context
            
        Returns:
            Enhanced prompt
        """
        base_prompt = """You are an AI assistant that helps users understand AI risks based on information from the MIT AI Risk Repository. 
Answer based on the retrieved context when possible. If the context doesn't contain relevant information, say so honestly.

When referring to sources, use 'SECTION X' or 'Document X' format which will be replaced with proper citations later."""
        
        if context:
            if query_type in ["employment", "socioeconomic"]:
                specific_guidance = """

IMPORTANT: This question is about employment, job, or socioeconomic risks from AI. The repository contains specific information about:
- Increased inequality and decline in employment quality (Domain 6.2)
- Economic and cultural devaluation of human effort (Domain 6.3) 
- Socioeconomic and Environmental risks (Domain 6)

Focus your answer on these specific employment-related risks when available in the context."""
                prompt = f"""{base_prompt}{specific_guidance}

Context: {context}

Question: {message}"""
            else:
                prompt = f"""{base_prompt}

Context: {context}

Question: {message}"""
        else:
            if query_type in ["employment", "socioeconomic"]:
                prompt = f"""You are an AI assistant that helps users understand AI risks based on the MIT AI Risk Repository. 
The repository contains information about employment and socioeconomic risks from AI, including:
- Job displacement and automation impacts
- Increased inequality from AI systems
- Decline in employment quality
- Economic impacts on workers

Answer the following question about AI employment/socioeconomic risks: {message}

When referring to sources, use 'SECTION X' or 'Document X' format which will be replaced with proper citations later."""
            else:
                prompt = f"""You are an AI assistant that helps users understand AI risks based on the MIT AI Risk Repository. 
Answer the following question about AI risk: {message}

When referring to sources, use 'SECTION X' or 'Document X' format which will be replaced with proper citations later."""
        
        return prompt