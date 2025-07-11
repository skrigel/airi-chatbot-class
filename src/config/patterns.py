"""
Pattern configuration for the monitor system.
Contains regex patterns used for rule-based classification.
"""
import re
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class PatternConfig:
    """Configuration for regex patterns used in classification."""
    
    # Override attempt patterns
    override_patterns: List[str] = None
    
    # Employment-related patterns
    employment_patterns: List[str] = None
    
    # Risk-related patterns
    risk_patterns: List[str] = None
    
    # Repository-related patterns
    repository_patterns: List[str] = None
    
    def __post_init__(self):
        """Initialize default patterns if not provided."""
        if self.override_patterns is None:
            self.override_patterns = [
                r"ignore (your|previous) instructions",
                r"forget (your|previous) instructions",
                r"disregard (your|previous) instructions",
                r"you are now",
                r"pretend to be",
                r"from now on you are",
                r"system prompt",
                r"jailbreak"
            ]
        
        if self.employment_patterns is None:
            self.employment_patterns = [
                r"job(s)?\b",
                r"employ(ment|ee|er)?\b",
                r"unemploy(ment|ed)?\b",
                r"work(ers|force|place)?\b",
                r"labor\b",
                r"labour\b",
                r"economic inequality",
                r"income inequality",
                r"wage",
                r"automat(ion|ed|ing)",
                r"displac(e|ed|ing|ement)",
                r"ai.+impact.+(job|employ|work|econom)"
            ]
        
        if self.risk_patterns is None:
            self.risk_patterns = [
                r"risk(s)?\b",
                r"danger(s|ous)?\b",
                r"threat(s)?\b",
                r"hazard(s|ous)?\b",
                r"harm(ful)?\b",
                r"impact(s)?\b",
                r"concern(s|ing)?\b",
                r"problem(s)?\b"
            ]
        
        if self.repository_patterns is None:
            self.repository_patterns = [
                r"repository",
                r"database",
                r"collection",
                r"catalog",
                r"how (many|to use|to search|to find|to navigate|is organized)",
                r"what (kind of|types of|is in|does it contain)",
                r"tell me about (the repository|this database|how to use)"
            ]

class PatternMatcher:
    """Helper class for pattern matching operations."""
    
    def __init__(self, config: PatternConfig):
        self.config = config
        self._compiled_patterns = self._compile_patterns()
    
    def _compile_patterns(self) -> Dict[str, List[re.Pattern]]:
        """Pre-compile all regex patterns for better performance."""
        return {
            'override': [re.compile(pattern, re.IGNORECASE) for pattern in self.config.override_patterns],
            'employment': [re.compile(pattern, re.IGNORECASE) for pattern in self.config.employment_patterns],
            'risk': [re.compile(pattern, re.IGNORECASE) for pattern in self.config.risk_patterns],
            'repository': [re.compile(pattern, re.IGNORECASE) for pattern in self.config.repository_patterns]
        }
    
    def matches_override_patterns(self, text: str) -> bool:
        """Check if text matches any override attempt patterns."""
        return any(pattern.search(text) for pattern in self._compiled_patterns['override'])
    
    def matches_employment_patterns(self, text: str) -> bool:
        """Check if text matches any employment-related patterns."""
        return any(pattern.search(text) for pattern in self._compiled_patterns['employment'])
    
    def matches_risk_patterns(self, text: str) -> bool:
        """Check if text matches any risk-related patterns."""
        return any(pattern.search(text) for pattern in self._compiled_patterns['risk'])
    
    def matches_repository_patterns(self, text: str) -> bool:
        """Check if text matches any repository-related patterns."""
        return any(pattern.search(text) for pattern in self._compiled_patterns['repository'])
    
    def has_ai_context(self, text: str) -> bool:
        """Check if text contains AI-related context."""
        ai_patterns = [
            re.compile(r"\bai\b", re.IGNORECASE),
            re.compile(r"artificial intelligence", re.IGNORECASE)
        ]
        return any(pattern.search(text) for pattern in ai_patterns)

# Default pattern configuration
DEFAULT_PATTERN_CONFIG = PatternConfig()

# Global pattern matcher instance
pattern_matcher = PatternMatcher(DEFAULT_PATTERN_CONFIG)