"""
SCQA (Situation-Complication-Question-Answer) taxonomy for structured content organization.
Enhances document metadata with structured information flow.
"""
import json
import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
from langchain.docstore.document import Document

from ...config.logging import get_logger
from ...config.settings import settings

logger = get_logger(__name__)

class SCQAComponent(Enum):
    """Components of the SCQA framework."""
    SITUATION = "situation"
    COMPLICATION = "complication"  
    QUESTION = "question"
    ANSWER = "answer"

class ContentType(Enum):
    """Types of content structure."""
    RISK_DESCRIPTION = "risk_description"
    IMPACT_ANALYSIS = "impact_analysis"
    MITIGATION_STRATEGY = "mitigation_strategy"
    CASE_STUDY = "case_study"
    DOMAIN_OVERVIEW = "domain_overview"

@dataclass
class SCQAStructure:
    """Structured representation of content using SCQA framework."""
    situation: str = ""
    complication: str = ""
    question: str = ""
    answer: str = ""
    content_type: ContentType = ContentType.RISK_DESCRIPTION
    domain: str = ""
    confidence: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "situation": self.situation,
            "complication": self.complication,
            "question": self.question,
            "answer": self.answer,
            "content_type": self.content_type.value,
            "domain": self.domain,
            "confidence": self.confidence
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SCQAStructure':
        """Create from dictionary."""
        return cls(
            situation=data.get("situation", ""),
            complication=data.get("complication", ""),
            question=data.get("question", ""),
            answer=data.get("answer", ""),
            content_type=ContentType(data.get("content_type", "risk_description")),
            domain=data.get("domain", ""),
            confidence=data.get("confidence", 0.0)
        )

class SCQAAnalyzer:
    """Analyzes content and extracts SCQA structure."""
    
    def __init__(self):
        self._init_patterns()
        self._init_domain_patterns()
    
    def _init_patterns(self):
        """Initialize patterns for detecting SCQA components."""
        
        # Situation indicators (context, background)
        self.situation_patterns = [
            r'in the context of',
            r'given that',
            r'in (.*?) systems?',
            r'when (.*?) is used',
            r'in (.*?) applications?',
            r'background:?',
            r'context:?',
            r'setting:?'
        ]
        
        # Complication indicators (problems, risks, challenges)
        self.complication_patterns = [
            r'however,?',
            r'but,?',
            r'unfortunately,?',
            r'the problem is',
            r'this leads? to',
            r'this causes?',
            r'this results? in',
            r'risk:?',
            r'challenge:?',
            r'issue:?',
            r'concern:?',
            r'may (cause|lead to|result in)',
            r'can (cause|lead to|result in)'
        ]
        
        # Question indicators (what needs to be addressed)
        self.question_patterns = [
            r'how (can|do|does|will|might)',
            r'what (is|are|will|might|can)',
            r'why (does|do|is|are|will)',
            r'when (will|does|do|is|are)',
            r'where (does|do|is|are|will)',
            r'which (.*?)\?',
            r'\?',
            r'whether',
            r'if (.*?) then'
        ]
        
        # Answer indicators (solutions, explanations, outcomes)
        self.answer_patterns = [
            r'therefore,?',
            r'thus,?',
            r'consequently,?',
            r'as a result,?',
            r'this means',
            r'solution:?',
            r'answer:?',
            r'outcome:?',
            r'result:?',
            r'mitigation:?',
            r'to address this',
            r'to solve this'
        ]
    
    def _init_domain_patterns(self):
        """Initialize domain-specific SCQA patterns."""
        
        self.domain_scqa_patterns = {
            'socioeconomic': {
                'situation': [
                    r'in the workforce',
                    r'in employment',
                    r'in (.*?) jobs?',
                    r'in (.*?) industries?',
                    r'workers in'
                ],
                'complication': [
                    r'job displacement',
                    r'unemployment',
                    r'wage reduction',
                    r'economic inequality',
                    r'loss of employment'
                ],
                'answer': [
                    r'retraining',
                    r're-skilling',
                    r'job creation',
                    r'economic support',
                    r'policy intervention'
                ]
            },
            'safety': {
                'situation': [
                    r'in (.*?) systems?',
                    r'when AI controls?',
                    r'in autonomous',
                    r'in critical systems?'
                ],
                'complication': [
                    r'system failure',
                    r'malfunction',
                    r'accident',
                    r'safety risk',
                    r'physical harm'
                ],
                'answer': [
                    r'safety protocols?',
                    r'failsafe',
                    r'redundancy',
                    r'safety testing',
                    r'monitoring'
                ]
            },
            'privacy': {
                'situation': [
                    r'when collecting data',
                    r'in data processing',
                    r'personal information',
                    r'user data'
                ],
                'complication': [
                    r'privacy breach',
                    r'data misuse',
                    r'surveillance',
                    r'unauthorized access',
                    r'privacy violation'
                ],
                'answer': [
                    r'encryption',
                    r'access controls?',
                    r'data protection',
                    r'privacy policies?',
                    r'consent mechanisms?'
                ]
            }
        }
    
    def analyze_document(self, document: Document) -> SCQAStructure:
        """Analyze a document and extract SCQA structure."""
        
        content = document.page_content
        metadata = document.metadata
        domain = metadata.get('domain', 'general')
        
        # Extract SCQA components
        situation = self._extract_situation(content, domain)
        complication = self._extract_complication(content, domain)
        question = self._extract_question(content, domain)
        answer = self._extract_answer(content, domain)
        
        # Determine content type
        content_type = self._classify_content_type(content, metadata)
        
        # Calculate confidence based on how many components we found
        confidence = self._calculate_confidence(situation, complication, question, answer)
        
        return SCQAStructure(
            situation=situation,
            complication=complication,
            question=question,
            answer=answer,
            content_type=content_type,
            domain=domain,
            confidence=confidence
        )
    
    def _extract_situation(self, content: str, domain: str) -> str:
        """Extract situation (context/background) from content."""
        content_lower = content.lower()
        
        # Look for situation patterns
        for pattern in self.situation_patterns:
            match = re.search(pattern, content_lower)
            if match:
                # Extract surrounding context
                start = max(0, match.start() - 50)
                end = min(len(content), match.end() + 100)
                return content[start:end].strip()
        
        # Domain-specific patterns
        if domain in self.domain_scqa_patterns:
            for pattern in self.domain_scqa_patterns[domain]['situation']:
                match = re.search(pattern, content_lower)
                if match:
                    start = max(0, match.start() - 30)
                    end = min(len(content), match.end() + 80)
                    return content[start:end].strip()
        
        # Fallback: use first sentence or metadata
        sentences = content.split('.')
        if sentences:
            return sentences[0].strip()
        
        return ""
    
    def _extract_complication(self, content: str, domain: str) -> str:
        """Extract complication (problem/risk) from content."""
        content_lower = content.lower()
        
        # Look for complication patterns
        for pattern in self.complication_patterns:
            match = re.search(pattern, content_lower)
            if match:
                # Extract surrounding context
                start = max(0, match.start() - 30)
                end = min(len(content), match.end() + 120)
                return content[start:end].strip()
        
        # Domain-specific patterns
        if domain in self.domain_scqa_patterns:
            for pattern in self.domain_scqa_patterns[domain]['complication']:
                if pattern in content_lower:
                    # Find the sentence containing this pattern
                    sentences = content.split('.')
                    for sentence in sentences:
                        if pattern in sentence.lower():
                            return sentence.strip()
        
        # Look for risk-related keywords
        risk_keywords = ['risk', 'problem', 'issue', 'concern', 'threat', 'danger']
        for keyword in risk_keywords:
            if keyword in content_lower:
                # Find sentence with risk keyword
                sentences = content.split('.')
                for sentence in sentences:
                    if keyword in sentence.lower():
                        return sentence.strip()
        
        return ""
    
    def _extract_question(self, content: str, domain: str) -> str:
        """Extract implicit or explicit questions from content."""
        content_lower = content.lower()
        
        # Look for explicit questions
        questions = re.findall(r'[.!]?\s*([A-Z][^.!?]*\?)', content)
        if questions:
            return questions[0].strip()
        
        # Look for question patterns
        for pattern in self.question_patterns:
            match = re.search(pattern, content_lower)
            if match:
                # Extract the sentence containing the question pattern
                sentences = content.split('.')
                for sentence in sentences:
                    if re.search(pattern, sentence.lower()):
                        return sentence.strip() + "?"
        
        # Generate implicit question based on content type and domain
        return self._generate_implicit_question(content, domain)
    
    def _extract_answer(self, content: str, domain: str) -> str:
        """Extract answer/solution from content."""
        content_lower = content.lower()
        
        # Look for answer patterns
        for pattern in self.answer_patterns:
            match = re.search(pattern, content_lower)
            if match:
                # Extract text after the answer indicator
                start = match.end()
                end = min(len(content), start + 150)
                return content[start:end].strip()
        
        # Domain-specific answer patterns
        if domain in self.domain_scqa_patterns:
            for pattern in self.domain_scqa_patterns[domain]['answer']:
                if pattern in content_lower:
                    sentences = content.split('.')
                    for sentence in sentences:
                        if pattern in sentence.lower():
                            return sentence.strip()
        
        # Fallback: use last meaningful sentence
        sentences = [s.strip() for s in content.split('.') if len(s.strip()) > 20]
        if sentences:
            return sentences[-1]
        
        return ""
    
    def _classify_content_type(self, content: str, metadata: Dict[str, Any]) -> ContentType:
        """Classify the type of content."""
        content_lower = content.lower()
        file_type = metadata.get('file_type', '')
        
        # Check for specific content type indicators
        if 'case study' in content_lower or 'example' in content_lower:
            return ContentType.CASE_STUDY
        elif 'mitigation' in content_lower or 'solution' in content_lower:
            return ContentType.MITIGATION_STRATEGY
        elif 'impact' in content_lower or 'effect' in content_lower:
            return ContentType.IMPACT_ANALYSIS
        elif file_type == 'ai_risk_domain_summary':
            return ContentType.DOMAIN_OVERVIEW
        else:
            return ContentType.RISK_DESCRIPTION
    
    def _generate_implicit_question(self, content: str, domain: str) -> str:
        """Generate an implicit question based on content and domain."""
        content_lower = content.lower()
        
        # Domain-specific question generation
        if domain == 'socioeconomic':
            if 'employment' in content_lower or 'job' in content_lower:
                return "How will this affect employment and economic opportunities?"
        elif domain == 'safety':
            if 'risk' in content_lower or 'danger' in content_lower:
                return "What safety measures can prevent this risk?"
        elif domain == 'privacy':
            if 'data' in content_lower or 'information' in content_lower:
                return "How can personal privacy be protected?"
        elif domain == 'bias':
            if 'discrimination' in content_lower or 'bias' in content_lower:
                return "How can algorithmic bias be detected and mitigated?"
        
        # Generic question based on content
        if 'risk' in content_lower:
            return "What are the implications of this risk?"
        elif 'impact' in content_lower:
            return "How significant is this impact?"
        else:
            return "What should be done about this?"
    
    def _calculate_confidence(self, situation: str, complication: str, question: str, answer: str) -> float:
        """Calculate confidence score based on extracted components."""
        components = [situation, complication, question, answer]
        filled_components = sum(1 for comp in components if comp.strip())
        
        # Base confidence on number of filled components
        base_confidence = filled_components / 4.0
        
        # Boost confidence if components are substantial
        substantial_boost = sum(1 for comp in components if len(comp.strip()) > 50) * 0.1
        
        return min(base_confidence + substantial_boost, 1.0)

class SCQATaxonomyManager:
    """Manages SCQA taxonomy for the entire document collection."""
    
    def __init__(self):
        self.analyzer = SCQAAnalyzer()
        self.taxonomy_cache = {}
    
    def enhance_document_with_scqa(self, document: Document) -> Document:
        """Enhance a document with SCQA structure metadata."""
        
        # Check cache first
        doc_id = self._get_document_id(document)
        if doc_id in self.taxonomy_cache:
            scqa_structure = self.taxonomy_cache[doc_id]
        else:
            # Analyze document
            scqa_structure = self.analyzer.analyze_document(document)
            self.taxonomy_cache[doc_id] = scqa_structure
        
        # Add SCQA metadata to document (excluding complex dict for ChromaDB compatibility)
        document.metadata.update({
            # 'scqa_structure': scqa_structure.to_dict(),  # Disabled - ChromaDB doesn't support dict metadata
            'scqa_situation': scqa_structure.situation,
            'scqa_complication': scqa_structure.complication,
            'scqa_question': scqa_structure.question,
            'scqa_answer': scqa_structure.answer,
            'scqa_content_type': scqa_structure.content_type.value,
            'scqa_confidence': scqa_structure.confidence
        })
        
        return document
    
    def _get_document_id(self, document: Document) -> str:
        """Generate unique ID for document caching."""
        # Use RID if available, otherwise create hash-based ID
        if 'rid' in document.metadata:
            return document.metadata['rid']
        else:
            import hashlib
            content_hash = hashlib.md5(document.page_content[:200].encode()).hexdigest()
            return f"doc_{content_hash}"
    
    def get_documents_by_scqa_component(self, 
                                       documents: List[Document], 
                                       component: SCQAComponent,
                                       query: str) -> List[Document]:
        """Filter documents by SCQA component matching a query."""
        matching_docs = []
        query_lower = query.lower()
        
        for doc in documents:
            if 'scqa_structure' not in doc.metadata:
                doc = self.enhance_document_with_scqa(doc)
            
            scqa_data = doc.metadata['scqa_structure']
            component_text = scqa_data.get(component.value, '').lower()
            
            # Check if query matches this component
            if any(word in component_text for word in query_lower.split()):
                matching_docs.append(doc)
        
        return matching_docs
    
    def get_taxonomy_statistics(self, documents: List[Document]) -> Dict[str, Any]:
        """Get statistics about SCQA taxonomy coverage."""
        
        enhanced_docs = [self.enhance_document_with_scqa(doc) for doc in documents]
        
        content_types = {}
        component_coverage = {comp.value: 0 for comp in SCQAComponent}
        avg_confidence = 0
        
        for doc in enhanced_docs:
            scqa_data = doc.metadata['scqa_structure']
            
            # Count content types
            content_type = scqa_data['content_type']
            content_types[content_type] = content_types.get(content_type, 0) + 1
            
            # Count component coverage
            for comp in SCQAComponent:
                if scqa_data.get(comp.value, '').strip():
                    component_coverage[comp.value] += 1
            
            # Sum confidence scores
            avg_confidence += scqa_data['confidence']
        
        if enhanced_docs:
            avg_confidence /= len(enhanced_docs)
        
        return {
            'total_documents': len(enhanced_docs),
            'content_types': content_types,
            'component_coverage': component_coverage,
            'average_confidence': avg_confidence
        }

# Global SCQA taxonomy manager
scqa_manager = SCQATaxonomyManager()