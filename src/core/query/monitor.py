import google.generativeai as genai
import logging
import json
import re

from ...config.patterns import pattern_matcher
from ...config.prompts import prompt_formatter
from ...config.domains import domain_classifier
from ...config.settings import settings
from ...config.logging import get_logger

logger = get_logger(__name__)

class Monitor:
    """
    Monitor class to filter messages and determine inquiry type.
    Acts as the first step in the processing pipeline.
    """
    
    def __init__(self, api_key, model_name=None):
        """Initialize the Monitor with API access."""
        self.model_name = model_name or settings.MONITOR_MODEL_NAME
        
        # Initialize client
        genai.configure(api_key=api_key)
        self.client = genai
        
        # Configuration components
        self.pattern_matcher = pattern_matcher
        self.prompt_formatter = prompt_formatter
        self.domain_classifier = domain_classifier
    
    def determine_inquiry_type(self, user_input):
        """
        Determine the type of inquiry and whether it's an override attempt.
        
        Args:
            user_input (str): The user's message
            
        Returns:
            dict: Contains inquiry_type, override_attempt, and primary_domain
        """
        # Try a rule-based classification if enabled
        if settings.MONITOR_ENABLE_RULE_BASED:
            rule_result = self._rule_based_classification(user_input)
            if rule_result:
                return rule_result
        
        # Fall back to model-based classification if enabled
        if settings.MONITOR_ENABLE_MODEL_BASED:
            try:
                # Create conversation history using prompt formatter
                conversation = self.prompt_formatter.build_conversation_history(user_input)
                
                # Generate response
                model = genai.GenerativeModel(model_name=self.model_name)
                chat = model.start_chat(history=conversation)
                response = chat.send_message(self.prompt_formatter.get_classification_request())
            
                response_text = response.text
            
                # Extract JSON from response
                try:
                    # Find JSON in the response if it's embedded in other text
                    response_text = response_text.strip()
                    if response_text.startswith('```json'):
                        response_text = response_text.replace('```json', '', 1)
                        response_text = response_text.replace('```', '', 1)
                
                    result = json.loads(response_text.strip())
                
                    # Validate result structure using domain classifier
                    if not self.domain_classifier.validate_classification_result(result):
                        logger.warning("Invalid monitor response format")
                        return self._default_result()
                        
                    return result
                    
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse monitor response: {response_text}")
                    return self._default_result()
                    
            except Exception as e:
                logger.error(f"Error in monitor: {str(e)}")
                return self._default_result()
        
        # If both rule-based and model-based are disabled, return default
        logger.warning("Both rule-based and model-based classification are disabled")
        return self._default_result()
    
    def _rule_based_classification(self, user_input):
        """
        Perform a simple rule-based classification before using the model.
        This is faster and can handle common cases.
        
        Args:
            user_input (str): The user's message
            
        Returns:
            dict or None: Classification result or None if rules don't apply
        """
        input_lower = user_input.lower()
        
        # Check for override attempts
        if self.pattern_matcher.matches_override_patterns(input_lower):
            return {
                "inquiry_type": "OUT_OF_SCOPE",
                "override_attempt": True,
                "primary_domain": "OTHER"
            }
        
        # Check for employment-related queries
        if (self.pattern_matcher.matches_employment_patterns(input_lower) and 
            self.pattern_matcher.has_ai_context(input_lower)):
            return {
                "inquiry_type": "EMPLOYMENT_RISK",
                "override_attempt": False,
                "primary_domain": "SOCIOECONOMIC"
            }
            
        # Check for specific risk inquiries
        if (self.pattern_matcher.matches_risk_patterns(input_lower) and 
            self.pattern_matcher.has_ai_context(input_lower)):
            # Try to determine domain using keyword matching
            domain = self.domain_classifier.classify_domain_by_keywords(input_lower)
            return {
                "inquiry_type": "SPECIFIC_RISK",
                "override_attempt": False,
                "primary_domain": domain
            }
            
        # General repository questions
        if self.pattern_matcher.matches_repository_patterns(input_lower):
            return {
                "inquiry_type": "GENERAL",
                "override_attempt": False,
                "primary_domain": "OTHER"
            }
        
        # If no rules match, return None to use model classification
        return None
    
    def _default_result(self):
        """Return a default result when processing fails."""
        return self.domain_classifier.get_default_classification()