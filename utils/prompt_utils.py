"""Prompt generation utilities for the EFA project."""

import re
from pathlib import Path
from typing import Dict, List, Optional, Union

from jinja2 import Environment, FileSystemLoader, Template

from config import PROMPTS_DIR
from utils.logging_utils import get_logger

logger = get_logger(__name__)


class PromptTemplateError(Exception):
    """Custom exception for prompt template operations."""
    pass


class PromptGenerator:
    """Handles prompt template loading and rendering."""
    
    def __init__(self, template_dir: Union[str, Path] = PROMPTS_DIR):
        """
        Initialize the prompt generator.
        
        Args:
            template_dir: Directory containing Jinja2 templates
        """
        self.template_dir = Path(template_dir)
        self.template_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up Jinja2 environment
        self.env = Environment(
            loader=FileSystemLoader(str(self.template_dir)),
            trim_blocks=True,
            lstrip_blocks=True
        )
        
        logger.info(f"Initialized PromptGenerator with template dir: {template_dir}")
    
    def load_template(self, template_name: str) -> Template:
        """
        Load a Jinja2 template.
        
        Args:
            template_name: Name of the template file
            
        Returns:
            Loaded Jinja2 template
            
        Raises:
            PromptTemplateError: If template cannot be loaded
        """
        try:
            template = self.env.get_template(template_name)
            logger.debug(f"Loaded template: {template_name}")
            return template
        except Exception as e:
            raise PromptTemplateError(f"Error loading template {template_name}: {str(e)}")
    
    def render_template(
        self, 
        template_name: str, 
        context: Dict,
        validate_required_keys: Optional[List[str]] = None
    ) -> str:
        """
        Render a template with the provided context.
        
        Args:
            template_name: Name of the template file
            context: Dictionary of variables for template rendering
            validate_required_keys: Optional list of required keys in context
            
        Returns:
            Rendered template string
            
        Raises:
            PromptTemplateError: If template rendering fails
        """
        if validate_required_keys:
            missing_keys = [key for key in validate_required_keys if key not in context]
            if missing_keys:
                raise PromptTemplateError(
                    f"Missing required context keys: {missing_keys}"
                )
        
        try:
            template = self.load_template(template_name)
            rendered = template.render(context)
            logger.debug(f"Rendered template {template_name} with context keys: {list(context.keys())}")
            return rendered
        except Exception as e:
            raise PromptTemplateError(f"Error rendering template {template_name}: {str(e)}")
    
    def create_persona_prompt(
        self,
        persona_data: Dict,
        instruments_data: Dict,
        template_name: str = "system_prompt_template.jinja"
    ) -> str:
        """
        Create a prompt for a specific persona and instruments.
        
        Args:
            persona_data: Dictionary containing persona information
            instruments_data: Dictionary containing instrument questions
            template_name: Name of the template to use
            
        Returns:
            Rendered prompt string
        """
        context = {
            "persona": persona_data,
            "instruments": instruments_data,
            "total_questions": sum(
                len(scale_data.get("questions", {})) 
                for scale_data in instruments_data.values()
                if isinstance(scale_data, dict)
            )
        }
        
        required_keys = ["persona", "instruments"]
        return self.render_template(template_name, context, required_keys)


def parse_demographic_text(demographic_text: str) -> Dict[str, str]:
    """
    Parse demographic information from text using regex patterns.
    
    Args:
        demographic_text: Raw demographic text to parse
        
    Returns:
        Dictionary of parsed demographic fields
    """
    demographics = {}
    
    # Common patterns for demographic parsing
    patterns = {
        "age": [
            r"age[:\s]+(\d+)",
            r"(\d+)\s*years?\s*old",
            r"aged?\s*(\d+)"
        ],
        "gender": [
            r"gender[:\s]+(male|female|non-binary|other|prefer not to say)",
            r"(male|female|non-binary|other|prefer not to say)",
        ],
        "education": [
            r"education[:\s]+([^,\n]+)",
            r"degree[:\s]+([^,\n]+)",
            r"graduated?\s+from\s+([^,\n]+)"
        ],
        "occupation": [
            r"occupation[:\s]+([^,\n]+)",
            r"job[:\s]+([^,\n]+)",
            r"works?\s+as\s+([^,\n]+)",
            r"profession[:\s]+([^,\n]+)"
        ],
        "location": [
            r"location[:\s]+([^,\n]+)",
            r"lives?\s+in\s+([^,\n]+)",
            r"from\s+([^,\n]+)",
            r"resides?\s+in\s+([^,\n]+)"
        ]
    }
    
    # Apply patterns to extract demographic information
    text_lower = demographic_text.lower()
    
    for field, field_patterns in patterns.items():
        for pattern in field_patterns:
            match = re.search(pattern, text_lower, re.IGNORECASE)
            if match:
                demographics[field] = match.group(1).strip()
                break
    
    # Set defaults for missing fields
    defaults = {
        "age": "unknown",
        "gender": "unknown",
        "education": "unknown",
        "occupation": "unknown",
        "location": "unknown"
    }
    
    for field, default_value in defaults.items():
        if field not in demographics:
            demographics[field] = default_value
    
    logger.debug(f"Parsed demographics: {demographics}")
    return demographics


def extract_response_scales(instruments_data: Dict) -> Dict[str, str]:
    """
    Extract response scale information from instruments data.
    
    Args:
        instruments_data: Dictionary containing instrument information
        
    Returns:
        Dictionary mapping scale names to their response instructions
    """
    response_scales = {}
    
    for scale_name, scale_data in instruments_data.items():
        if isinstance(scale_data, dict):
            # Check for scale-level response instructions
            if "response_scale" in scale_data:
                response_scales[scale_name] = scale_data["response_scale"]
            else:
                # Use a default response scale if none specified
                response_scales[scale_name] = (
                    "Please respond to each item using the appropriate scale "
                    "(typically 1-5 or 1-7 scale). Provide only numeric responses."
                )
    
    logger.debug(f"Extracted response scales for {len(response_scales)} scales")
    return response_scales


def validate_prompt_variables(prompt_text: str, expected_variables: List[str]) -> bool:
    """
    Validate that a prompt contains all expected template variables.
    
    Args:
        prompt_text: The prompt text to validate
        expected_variables: List of variable names that should be present
        
    Returns:
        True if all variables are found, False otherwise
    """
    missing_variables = []
    
    for variable in expected_variables:
        # Check for Jinja2 variable syntax
        if f"{{{{{variable}" not in prompt_text and f"{variable}" not in prompt_text:
            missing_variables.append(variable)
    
    if missing_variables:
        logger.warning(f"Missing variables in prompt: {missing_variables}")
        return False
    
    logger.debug("All expected variables found in prompt")
    return True


def truncate_text(text: str, max_length: int = 1000, suffix: str = "...") -> str:
    """
    Truncate text to a maximum length while preserving word boundaries.
    
    Args:
        text: Text to truncate
        max_length: Maximum length allowed
        suffix: Suffix to add when truncating
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    # Find the last space before the max length
    truncate_at = text.rfind(' ', 0, max_length - len(suffix))
    if truncate_at == -1:
        truncate_at = max_length - len(suffix)
    
    return text[:truncate_at] + suffix