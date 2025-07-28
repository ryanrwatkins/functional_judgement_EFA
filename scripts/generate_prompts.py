"""Generate prompts for personas using Jinja2 templates."""

import sys
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

from config import INSTRUMENTS_JSON, PERSONAS_JSON, SYSTEM_PROMPT_TEMPLATE
from utils.file_io import read_json_file
from utils.logging_utils import setup_logging, get_logger
from utils.prompt_utils import PromptGenerator

logger = get_logger(__name__)


def load_personas_data(file_path: Path = PERSONAS_JSON) -> Dict:
    """
    Load personas data from JSON file.
    
    Args:
        file_path: Path to personas JSON file
        
    Returns:
        Dictionary containing personas data
    """
    try:
        data = read_json_file(file_path)
        logger.info(f"Loaded {len(data.get('personas', []))} personas")
        return data
    except Exception as e:
        logger.error(f"Error loading personas data: {str(e)}")
        raise


def load_instruments_data(file_path: Path = INSTRUMENTS_JSON) -> Dict:
    """
    Load instruments data from JSON file.
    
    Args:
        file_path: Path to instruments JSON file
        
    Returns:
        Dictionary containing instruments data
    """
    try:
        data = read_json_file(file_path)
        logger.info(f"Loaded {len(data)} psychological scales")
        return data
    except Exception as e:
        logger.error(f"Error loading instruments data: {str(e)}")
        raise


def calculate_total_questions(instruments_data: Dict) -> int:
    """
    Calculate total number of questions across all instruments.
    
    Args:
        instruments_data: Dictionary containing instruments data
        
    Returns:
        Total number of questions
    """
    total = 0
    for scale_data in instruments_data.values():
        if isinstance(scale_data, dict) and "subscales" in scale_data:
            for subscale_data in scale_data["subscales"].values():
                if isinstance(subscale_data, dict) and "questions" in subscale_data:
                    total += len(subscale_data["questions"])
    
    logger.debug(f"Total questions calculated: {total}")
    return total


def generate_persona_prompt(
    persona_data: Dict,
    instruments_data: Dict,
    template_path: Path = SYSTEM_PROMPT_TEMPLATE
) -> str:
    """
    Generate a prompt for a specific persona.
    
    Args:
        persona_data: Dictionary containing persona information
        instruments_data: Dictionary containing instruments data
        template_path: Path to the Jinja2 template
        
    Returns:
        Generated prompt string
    """
    try:
        # Initialize prompt generator
        prompt_gen = PromptGenerator()
        
        # Calculate total questions
        total_questions = calculate_total_questions(instruments_data)
        
        # Prepare context for template
        context = {
            "persona": persona_data,
            "instruments": instruments_data,
            "total_questions": total_questions
        }
        
        # Generate prompt
        template_name = template_path.name
        prompt = prompt_gen.render_template(template_name, context)
        
        logger.debug(f"Generated prompt for persona {persona_data.get('id', 'unknown')} ({len(prompt)} characters)")
        return prompt
        
    except Exception as e:
        logger.error(f"Error generating prompt for persona {persona_data.get('id', 'unknown')}: {str(e)}")
        raise


def generate_all_prompts(
    personas_file: Path = PERSONAS_JSON,
    instruments_file: Path = INSTRUMENTS_JSON,
    template_file: Path = SYSTEM_PROMPT_TEMPLATE,
    limit_personas: Optional[int] = None
) -> Dict[int, str]:
    """
    Generate prompts for all personas.
    
    Args:
        personas_file: Path to personas JSON file
        instruments_file: Path to instruments JSON file
        template_file: Path to prompt template
        limit_personas: Optional limit on number of personas to process
        
    Returns:
        Dictionary mapping persona IDs to their prompts
    """
    try:
        # Load data
        logger.info("Loading personas and instruments data...")
        personas_data = load_personas_data(personas_file)
        instruments_data = load_instruments_data(instruments_file)
        
        # Generate prompts for each persona
        prompts = {}
        personas_to_process = personas_data["personas"]
        
        if limit_personas:
            personas_to_process = personas_to_process[:limit_personas]
            logger.info(f"Limited processing to {limit_personas} personas")
        
        logger.info(f"Generating prompts for {len(personas_to_process)} personas...")
        
        for persona in personas_to_process:
            persona_id = persona["id"]
            
            try:
                prompt = generate_persona_prompt(persona, instruments_data, template_file)
                prompts[persona_id] = prompt
                
                if persona_id % 100 == 0:  # Log progress every 100 personas
                    logger.info(f"Generated prompts for {persona_id} personas")
                    
            except Exception as e:
                logger.error(f"Failed to generate prompt for persona {persona_id}: {str(e)}")
                continue
        
        logger.info(f"Successfully generated {len(prompts)} prompts")
        return prompts
        
    except Exception as e:
        logger.error(f"Error generating prompts: {str(e)}")
        raise


def validate_generated_prompts(prompts: Dict[int, str]) -> bool:
    """
    Validate generated prompts for completeness and consistency.
    
    Args:
        prompts: Dictionary of generated prompts
        
    Returns:
        True if validation passes
    """
    validation_errors = []
    
    # Check for empty prompts
    empty_prompts = [pid for pid, prompt in prompts.items() if not prompt.strip()]
    if empty_prompts:
        validation_errors.append(f"Found {len(empty_prompts)} empty prompts")
    
    # Check for reasonable prompt lengths
    prompt_lengths = [len(prompt) for prompt in prompts.values()]
    if prompt_lengths:
        avg_length = sum(prompt_lengths) / len(prompt_lengths)
        min_length = min(prompt_lengths)
        max_length = max(prompt_lengths)
        
        logger.info(f"Prompt length stats - Avg: {avg_length:.0f}, Min: {min_length}, Max: {max_length}")
        
        # Check for unusually short prompts (likely errors)
        short_prompts = [pid for pid, prompt in prompts.items() if len(prompt) < avg_length * 0.5]
        if short_prompts:
            validation_errors.append(f"Found {len(short_prompts)} unusually short prompts")
    
    # Check for required content in prompts
    required_content = [
        "Demographics:",
        "Behavioral Context:",
        "Response Scales by Instrument:",
        "Instructions"
    ]
    
    for pid, prompt in prompts.items():
        missing_content = [content for content in required_content if content not in prompt]
        if missing_content:
            validation_errors.append(f"Persona {pid} prompt missing: {missing_content}")
    
    if validation_errors:
        logger.error(f"Prompt validation failed with {len(validation_errors)} errors:")
        for error in validation_errors[:10]:  # Limit error output
            logger.error(f"  - {error}")
        return False
    
    logger.info(f"Prompt validation passed for {len(prompts)} prompts")
    return True


def main(limit_personas: Optional[int] = None) -> None:
    """
    Main function to generate and validate prompts.
    
    Args:
        limit_personas: Optional limit on number of personas to process
    """
    setup_logging()
    
    logger.info("Starting prompt generation...")
    
    try:
        # Generate prompts
        prompts = generate_all_prompts(limit_personas=limit_personas)
        
        # Validate prompts
        if validate_generated_prompts(prompts):
            logger.info("Prompt generation completed successfully!")
        else:
            logger.error("Prompt validation failed!")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Prompt generation failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    # For testing, limit to first 10 personas
    main(limit_personas=10)