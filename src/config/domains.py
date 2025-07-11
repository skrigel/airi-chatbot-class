"""
Generic domain configuration and classification system.
Replaces hardcoded employment-specific logic with flexible domain definitions.
"""
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

class InquiryType(Enum):
    """Enumeration of inquiry types."""
    GENERAL = "general"
    SPECIFIC_RISK = "specific_risk"
    EMPLOYMENT_RISK = "employment"  # Renamed for consistency
    RECOMMENDATION = "recommendation"
    OUT_OF_SCOPE = "out_of_scope"

@dataclass
class DomainDefinition:
    """Generic definition of a domain with its search configuration."""
    name: str
    keywords: List[str]
    search_queries: List[str]
    enhanced_search: bool = True
    document_limit: int = 2
    search_weight: float = 1.0
    enabled: bool = True

@dataclass 
class DomainConfig:
    """Configuration for domain classification and processing."""
    
    # Domain definitions - completely generic system
    domains: Dict[str, DomainDefinition] = field(default_factory=lambda: {
        'socioeconomic': DomainDefinition(
            name='socioeconomic',
            keywords=['employment', 'job', 'work', 'economic', 'social', 'inequality', 'automation', 'labor', 'wage', 'unemployment', 'workforce', 'displacement'],
            search_queries=[
                "6.2 Increased inequality and decline in employment quality",
                "6.3 Economic and cultural devaluation of human effort", 
                "socioeconomic environmental AI risk",
                "automation impact on employment",
                "AI job displacement",
                "economic inequality artificial intelligence"
            ],
            enhanced_search=True,
            document_limit=3,
            search_weight=1.2
        ),
        'safety': DomainDefinition(
            name='safety',
            keywords=['safety', 'security', 'risk', 'harm', 'danger', 'threat', 'hazard', 'misuse', 'accident', 'injury', 'physical'],
            search_queries=[
                "AI safety risks and mitigation",
                "security threats from AI systems", 
                "potential AI hazards and dangers",
                "physical harm from AI systems",
                "AI safety measures and protocols"
            ],
            enhanced_search=True,
            document_limit=3,
            search_weight=1.1
        ),
        'privacy': DomainDefinition(
            name='privacy',
            keywords=['privacy', 'data', 'surveillance', 'personal', 'information', 'tracking', 'monitoring', 'confidential', 'secret'],
            search_queries=[
                "AI privacy risks and data protection",
                "surveillance and monitoring concerns",
                "personal information security in AI"
            ],
            enhanced_search=True,
            document_limit=2
        ),
        'bias': DomainDefinition(
            name='bias',
            keywords=['bias', 'discrimination', 'unfair', 'prejudice', 'inequality', 'fairness', 'equity', 'representation', 'exclusion'],
            search_queries=[
                "AI bias and discrimination issues",
                "fairness and equity in AI systems",
                "algorithmic bias and representation"
            ],
            enhanced_search=True,
            document_limit=2
        ),
        'governance': DomainDefinition(
            name='governance',
            keywords=['regulation', 'policy', 'oversight', 'governance', 'control', 'compliance', 'law', 'legal', 'ethics', 'accountability'],
            search_queries=[
                "AI governance and policy frameworks",
                "regulatory approaches to AI",
                "legal compliance for AI systems",
                "AI ethics and accountability"
            ],
            enhanced_search=True,
            document_limit=2
        ),
        'technical': DomainDefinition(
            name='technical',
            keywords=['algorithm', 'model', 'technical', 'performance', 'accuracy', 'robustness', 'reliability', 'system'],
            search_queries=[
                "technical AI system performance",
                "algorithm reliability and robustness",
                "AI model accuracy and performance"
            ],
            enhanced_search=False,  # Use standard search for technical queries
            document_limit=2
        )
    })
    
    # Valid inquiry types
    inquiry_types: List[str] = field(default_factory=lambda: [e.value for e in InquiryType])
    
    # Classification thresholds
    confidence_threshold: float = 0.7
    
    # Default classification result
    default_classification: Dict[str, Any] = field(default_factory=lambda: {
        "query_type": InquiryType.GENERAL.value,
        "domain": "other",
        "confidence": 0.5,
        "override_attempt": False
    })

class DomainClassifier:
    """Generic domain classifier that works with any configured domain."""
    
    def __init__(self, config: DomainConfig = None):
        self.config = config or DomainConfig()
    
    def classify_domain(self, query: str) -> str:
        """Classify the domain of a query based on keywords."""
        query_lower = query.lower()
        
        # Count matches for each enabled domain
        domain_scores = {}
        for domain_name, domain_def in self.config.domains.items():
            if not domain_def.enabled:
                continue
                
            score = sum(1 for keyword in domain_def.keywords if keyword in query_lower)
            if score > 0:
                # Apply search weight to score
                domain_scores[domain_name] = score * domain_def.search_weight
        
        # Return the domain with the highest score, or 'other' if no matches
        if domain_scores:
            return max(domain_scores, key=domain_scores.get)
        return 'other'
    
    def get_domain_queries(self, domain: str) -> List[str]:
        """Get search queries for any domain - generic method."""
        domain_def = self.config.domains.get(domain)
        if domain_def and domain_def.enabled:
            return domain_def.search_queries.copy()
        return []
    
    def get_domain_keywords(self, domain: str) -> List[str]:
        """Get keywords for any domain - generic method."""
        domain_def = self.config.domains.get(domain)
        if domain_def and domain_def.enabled:
            return domain_def.keywords.copy()
        return []
    
    def get_domain_config(self, domain: str) -> Optional[DomainDefinition]:
        """Get full domain configuration for any domain."""
        return self.config.domains.get(domain)
    
    def has_enhanced_search(self, domain: str) -> bool:
        """Check if any domain uses enhanced search - generic method."""
        domain_def = self.config.domains.get(domain)
        return domain_def.enhanced_search if domain_def and domain_def.enabled else False
    
    def get_document_limit(self, domain: str) -> int:
        """Get document limit for any domain-specific search."""
        domain_def = self.config.domains.get(domain)
        return domain_def.document_limit if domain_def and domain_def.enabled else 2
    
    def validate_classification_result(self, result: Any) -> bool:
        """Validate that a classification result has the expected structure."""
        if not isinstance(result, dict):
            return False
        
        required_fields = ['query_type', 'domain', 'confidence']
        if not all(field in result for field in required_fields):
            return False
            
        # Validate inquiry type
        if result.get('query_type') not in self.config.inquiry_types:
            return False
            
        return True
    
    def get_all_domains(self) -> List[str]:
        """Get list of all enabled domain names."""
        return [name for name, domain_def in self.config.domains.items() if domain_def.enabled]
    
    def get_default_classification(self) -> Dict[str, Any]:
        """Get default classification result."""
        return self.config.default_classification.copy()
    
    def is_domain_enabled(self, domain: str) -> bool:
        """Check if a domain is enabled."""
        domain_def = self.config.domains.get(domain)
        return domain_def.enabled if domain_def else False
    
    # Backward compatibility methods for gradual migration
    def get_employment_queries(self) -> List[str]:
        """Legacy method - use get_domain_queries('socioeconomic') instead."""
        return self.get_domain_queries('socioeconomic')
    
    def classify_domain_by_keywords(self, text: str) -> str:
        """Legacy method - use classify_domain() instead."""
        return self.classify_domain(text)

# Default domain configuration
DEFAULT_DOMAIN_CONFIG = DomainConfig()

# Global domain classifier instance  
domain_classifier = DomainClassifier(DEFAULT_DOMAIN_CONFIG)