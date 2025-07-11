"""
Query refinement engine for handling over-broad queries and providing suggestions.
Helps users ask better questions and discover relevant content.
"""
import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from ...config.logging import get_logger
from ...config.domains import domain_classifier
from ...config.prompts import prompt_manager

logger = get_logger(__name__)

class QueryComplexity(Enum):
    """Classification of query complexity and specificity."""
    VERY_BROAD = "very_broad"
    BROAD = "broad"
    SPECIFIC = "specific"
    VERY_SPECIFIC = "very_specific"

@dataclass
class RefinementSuggestion:
    """A suggested refinement for a user query."""
    question: str
    domain: str
    reasoning: str
    confidence: float

@dataclass
class RefinementResult:
    """Result of query refinement analysis."""
    original_query: str
    complexity: QueryComplexity
    needs_refinement: bool
    suggestions: List[RefinementSuggestion]
    entity_issues: List[str]
    refined_query: Optional[str] = None

class QueryRefiner:
    """Handles query refinement, entity validation, and suggestion generation."""
    
    def __init__(self):
        self._init_patterns()
        self._init_entity_dictionary()
    
    def _init_patterns(self):
        """Initialize patterns for detecting over-broad queries."""
        
        # Very broad query patterns
        self.very_broad_patterns = [
            r'\bwill ai (kill|destroy|end|eliminate) (us|humanity|everyone|people|the world)\b',
            r'\bis ai (dangerous|bad|harmful|evil)\b',
            r'\bwhat (are|is) ai risks?\b',
            r'\btell me about ai\b',
            r'\bai (impact|effect|influence)\b$',
            r'\bhow does ai work\b',
            r'\bwhat is artificial intelligence\b'
        ]
        
        # Broad query indicators
        self.broad_indicators = [
            'everything', 'anything', 'all about', 'general', 'overview',
            'explain ai', 'tell me', 'what are', 'how does', 'why does'
        ]
        
        # Specific query indicators
        self.specific_indicators = [
            'in my industry', 'for my job', 'in healthcare', 'in finance',
            'specific example', 'particular case', 'exactly how',
            'precise impact', 'detailed analysis'
        ]
        
        # Entity patterns that might not exist in repository
        self.entity_patterns = [
            r'\b(kendall square|harvard|mit campus|boston|cambridge)\b',
            r'\b(google|microsoft|apple|amazon|facebook|meta)\b',
            r'\b(chatgpt|gpt-?\d+|claude|bard)\b',
            r'\b\d{4}\b'  # Years - repository might not have specific year data
        ]
    
    def _init_entity_dictionary(self):
        """Initialize dictionary of entities known to exist in repository."""
        
        # Entities we know exist in the AI Risk Repository
        self.known_entities = {
            'domains': [
                'employment', 'job displacement', 'automation', 'labor',
                'safety', 'security', 'privacy', 'surveillance',
                'bias', 'discrimination', 'fairness', 'equity',
                'governance', 'regulation', 'policy', 'oversight',
                'misinformation', 'manipulation', 'fraud'
            ],
            'industries': [
                'healthcare', 'finance', 'transportation', 'education',
                'manufacturing', 'retail', 'media', 'agriculture'
            ],
            'technologies': [
                'machine learning', 'deep learning', 'neural networks',
                'computer vision', 'natural language processing',
                'autonomous vehicles', 'facial recognition'
            ]
        }
    
    def analyze_query(self, query: str) -> RefinementResult:
        """Analyze a query and determine if it needs refinement."""
        
        complexity = self._assess_complexity(query)
        entity_issues = self._check_entity_availability(query)
        needs_refinement = complexity in [QueryComplexity.VERY_BROAD, QueryComplexity.BROAD] or bool(entity_issues)
        
        suggestions = []
        if needs_refinement:
            suggestions = self._generate_suggestions(query, complexity)
        
        # Generate refined query if possible
        refined_query = self._auto_refine_query(query) if complexity == QueryComplexity.VERY_BROAD else None
        
        return RefinementResult(
            original_query=query,
            complexity=complexity,
            needs_refinement=needs_refinement,
            suggestions=suggestions,
            entity_issues=entity_issues,
            refined_query=refined_query
        )
    
    def _assess_complexity(self, query: str) -> QueryComplexity:
        """Assess the complexity and specificity of a query."""
        query_lower = query.lower()
        
        # Check for very broad patterns
        for pattern in self.very_broad_patterns:
            if re.search(pattern, query_lower):
                return QueryComplexity.VERY_BROAD
        
        # Count broad vs specific indicators
        broad_count = sum(1 for indicator in self.broad_indicators if indicator in query_lower)
        specific_count = sum(1 for indicator in self.specific_indicators if indicator in query_lower)
        
        # Assess based on length and specificity
        word_count = len(query.split())
        
        if broad_count > specific_count and word_count < 8:
            if broad_count >= 2:
                return QueryComplexity.VERY_BROAD
            else:
                return QueryComplexity.BROAD
        elif specific_count > broad_count and word_count > 8:
            return QueryComplexity.VERY_SPECIFIC
        elif word_count > 12 or specific_count > 0:
            return QueryComplexity.SPECIFIC
        else:
            return QueryComplexity.BROAD
    
    def _check_entity_availability(self, query: str) -> List[str]:
        """Check if mentioned entities exist in the repository."""
        issues = []
        query_lower = query.lower()
        
        # Check for entities that likely don't exist in repository
        for pattern in self.entity_patterns:
            matches = re.findall(pattern, query_lower)
            for match in matches:
                # Skip if it's a known entity
                if not any(match in entities for entities in self.known_entities.values()):
                    issues.append(f"'{match}' may not be covered in the repository")
        
        return issues
    
    def _generate_suggestions(self, query: str, complexity: QueryComplexity) -> List[RefinementSuggestion]:
        """Generate refinement suggestions based on query analysis."""
        suggestions = []
        
        # Detect domain from query
        domain = domain_classifier.classify_domain(query)
        
        if complexity == QueryComplexity.VERY_BROAD:
            # For very broad queries, provide specific sub-questions
            if 'kill' in query.lower() or 'destroy' in query.lower() or 'dangerous' in query.lower():
                suggestions.extend([
                    RefinementSuggestion(
                        question="What are the most documented AI safety failures?",
                        domain="safety",
                        reasoning="Focuses on specific documented risks rather than existential speculation",
                        confidence=0.9
                    ),
                    RefinementSuggestion(
                        question="How can AI systems cause unintended harm?",
                        domain="safety", 
                        reasoning="Addresses safety concerns with concrete examples",
                        confidence=0.8
                    ),
                    RefinementSuggestion(
                        question="What safety measures exist for AI systems?",
                        domain="safety",
                        reasoning="Provides actionable information about risk mitigation",
                        confidence=0.7
                    )
                ])
            
            elif 'ai risks' in query.lower() or 'ai impact' in query.lower():
                suggestions.extend([
                    RefinementSuggestion(
                        question="How will AI affect employment and job quality?",
                        domain="socioeconomic",
                        reasoning="Focuses on well-documented employment impacts",
                        confidence=0.9
                    ),
                    RefinementSuggestion(
                        question="What privacy risks do AI systems pose?",
                        domain="privacy",
                        reasoning="Addresses specific privacy and surveillance concerns",
                        confidence=0.8
                    ),
                    RefinementSuggestion(
                        question="How can AI systems exhibit bias and discrimination?",
                        domain="bias",
                        reasoning="Explores algorithmic fairness and equity issues",
                        confidence=0.8
                    )
                ])
        
        elif complexity == QueryComplexity.BROAD:
            # For broad queries, suggest more specific versions within the domain
            if domain != 'other':
                domain_suggestions = prompt_manager.get_follow_up_suggestions(domain)
                for suggestion in domain_suggestions[:3]:  # Limit to 3
                    suggestions.append(RefinementSuggestion(
                        question=suggestion,
                        domain=domain,
                        reasoning=f"More specific question within {domain} domain",
                        confidence=0.7
                    ))
            else:
                # Generic suggestions for unclassified broad queries
                suggestions.extend([
                    RefinementSuggestion(
                        question="What are the economic impacts of AI automation?",
                        domain="socioeconomic",
                        reasoning="Focuses on documented economic effects",
                        confidence=0.6
                    ),
                    RefinementSuggestion(
                        question="How do AI systems violate privacy?",
                        domain="privacy",
                        reasoning="Addresses specific privacy mechanisms",
                        confidence=0.6
                    )
                ])
        
        return suggestions
    
    def _auto_refine_query(self, query: str) -> Optional[str]:
        """Automatically refine very broad queries into more specific ones."""
        query_lower = query.lower()
        
        # Auto-refinement patterns
        refinements = {
            r'will ai kill (us|humanity|everyone)': "What are the most severe documented AI safety risks?",
            r'is ai dangerous': "What types of harm can AI systems cause?",
            r'what are ai risks': "What are the main categories of AI risks in the repository?",
            r'tell me about ai': "What are the key AI risk domains: employment, safety, privacy, or bias?",
            r'how does ai work': "How do AI systems create risks in employment, safety, or privacy?"
        }
        
        for pattern, refinement in refinements.items():
            if re.search(pattern, query_lower):
                return refinement
        
        return None
    
    def format_suggestions_response(self, result: RefinementResult) -> str:
        """Format refinement suggestions into a user-friendly response."""
        if not result.needs_refinement:
            return None
        
        if result.complexity == QueryComplexity.VERY_BROAD:
            response = f'Your question "{result.original_query}" covers a broad area. The repository has specific information on:\n\n'
        else:
            response = f'To give you the best answer, I can help you explore:\n\n'
        
        # Add suggestions
        for i, suggestion in enumerate(result.suggestions[:3], 1):  # Limit to 3
            response += f'{i}. {suggestion.question}\n'
        
        response += '\nWhich aspect interests you most?'
        
        # Add entity warnings if any
        if result.entity_issues:
            response += f'\n\nNote: {result.entity_issues[0]}'
        
        return response
    
    def get_clickable_suggestions(self, result: RefinementResult) -> List[Dict[str, str]]:
        """Get suggestions in a format suitable for UI buttons/links."""
        return [
            {
                'text': suggestion.question,
                'domain': suggestion.domain,
                'confidence': suggestion.confidence
            }
            for suggestion in result.suggestions[:4]  # UI typically shows 3-4 options
        ]

# Global query refiner instance
query_refiner = QueryRefiner()