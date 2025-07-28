"""Run LLM simulation across personas, models, and experimental conditions."""

import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from tqdm import tqdm

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

from config import CONDITIONS, INSTRUMENTS_JSON, MODELS, PERSONA_RESPONSES_JSON, PERSONAS_JSON
from scripts.generate_prompts import generate_persona_prompt, load_instruments_data, load_personas_data
from utils.file_io import write_json_file
from utils.llm_clients import create_llm_client, RateLimitedClient
from utils.logging_utils import setup_logging, get_logger

logger = get_logger(__name__)


class SimulationRunner:
    """Manages the execution of LLM simulations across different conditions."""
    
    def __init__(self, rate_limit: int = 30):
        """
        Initialize simulation runner.
        
        Args:
            rate_limit: Requests per minute for rate limiting
        """
        self.rate_limit = rate_limit
        self.clients = {}
        self.personas_data = None
        self.instruments_data = None
        self.simulation_results = {}
        
        logger.info(f"Initialized SimulationRunner with rate limit: {rate_limit} req/min")
    
    def load_data(self) -> None:
        """Load personas and instruments data."""
        logger.info("Loading simulation data...")
        self.personas_data = load_personas_data()
        self.instruments_data = load_instruments_data()
        
        logger.info(f"Loaded {len(self.personas_data['personas'])} personas")
        logger.info(f"Loaded {len(self.instruments_data)} psychological scales")
    
    def initialize_clients(self, model_names: List[str]) -> None:
        """
        Initialize LLM clients for specified models.
        
        Args:
            model_names: List of model names to initialize
        """
        logger.info(f"Initializing clients for models: {model_names}")
        
        for model_name in model_names:
            try:
                client = create_llm_client(model_name)
                # Add rate limiting
                rate_limited_client = RateLimitedClient(client, self.rate_limit)
                self.clients[model_name] = rate_limited_client
                
                logger.info(f"Initialized client for {model_name}")
                
            except Exception as e:
                logger.error(f"Failed to initialize client for {model_name}: {str(e)}")
                # Continue with other models
    
    def create_condition_instruments(self, condition: str) -> Dict:
        """
        Create instruments structure for a specific experimental condition.
        
        Args:
            condition: Experimental condition name
            
        Returns:
            Instruments data structured for the condition
        """
        if condition == "condition_3":
            # Randomize scale order for condition 3
            scales = list(self.instruments_data.keys())
            random.shuffle(scales)
            return {scale: self.instruments_data[scale] for scale in scales}
        else:
            # Use original order for conditions 1 and 2
            return self.instruments_data
    
    def create_questions_for_condition(self, condition: str, instruments: Dict) -> List[Tuple[str, str, str]]:
        """
        Create question list for a specific condition.
        
        Args:
            condition: Experimental condition name
            instruments: Instruments data for this condition
            
        Returns:
            List of (scale_name, subscale_name, question_id, question_text) tuples
        """
        questions = []
        
        for scale_name, scale_data in instruments.items():
            for subscale_name, subscale_data in scale_data["subscales"].items():
                for question_id, question_text in subscale_data["questions"].items():
                    questions.append((scale_name, subscale_name, question_id, question_text))
        
        logger.debug(f"Created {len(questions)} questions for {condition}")
        return questions
    
    def parse_model_responses(self, response_text: str, num_questions: int) -> List[Optional[int]]:
        """
        Parse model response text to extract numeric answers.
        
        Args:
            response_text: Raw response from the model
            num_questions: Expected number of questions
            
        Returns:
            List of parsed numeric responses (None for unparseable responses)
        """
        responses = []
        lines = response_text.strip().split('\n')
        
        # Try to extract numbers from each line
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Extract numbers from the line
            numbers = []
            for word in line.split():
                try:
                    # Try to parse as integer
                    num = int(word.strip('.,!?()[]{}'))
                    if 1 <= num <= 10:  # Reasonable range for Likert scales
                        numbers.append(num)
                except ValueError:
                    continue
            
            # Use the first valid number found
            if numbers:
                responses.append(numbers[0])
        
        # If we don't have enough responses, try a different approach
        if len(responses) < num_questions:
            # Try to extract all numbers from the entire text
            import re
            all_numbers = re.findall(r'\b([1-9]|10)\b', response_text)
            numeric_responses = [int(n) for n in all_numbers if 1 <= int(n) <= 10]
            
            if len(numeric_responses) >= num_questions:
                responses = numeric_responses[:num_questions]
        
        # Pad with None if still not enough responses
        while len(responses) < num_questions:
            responses.append(None)
        
        # Truncate if too many responses
        responses = responses[:num_questions]
        
        logger.debug(f"Parsed {len([r for r in responses if r is not None])}/{num_questions} valid responses")
        return responses
    
    def simulate_persona_condition(
        self, 
        persona: Dict, 
        model_name: str, 
        condition: str,
        max_retries: int = 3
    ) -> Dict:
        """
        Simulate a persona's responses for one condition.
        
        Args:
            persona: Persona data
            model_name: Name of the model to use
            condition: Experimental condition
            max_retries: Maximum number of retries for failed requests
            
        Returns:
            Dictionary containing the persona's responses for this condition
        """
        client = self.clients[model_name]
        persona_id = persona["id"]
        
        logger.debug(f"Simulating persona {persona_id} with {model_name} for {condition}")
        
        # Create condition-specific instruments
        condition_instruments = self.create_condition_instruments(condition)
        
        # Create questions list
        questions = self.create_questions_for_condition(condition, condition_instruments)
        
        # Generate prompt for this persona and condition
        try:
            prompt = generate_persona_prompt(persona, condition_instruments)
            
            # Add condition-specific instructions
            if condition == "condition_1":
                prompt += "\n\nIMPORTANT: Respond to each question independently. Do not consider previous questions when answering."
            elif condition == "condition_2":
                prompt += "\n\nIMPORTANT: Respond to questions within each scale as a group, but treat each scale independently."
            elif condition == "condition_3":
                prompt += "\n\nIMPORTANT: Consider all questions as part of a comprehensive assessment. Your responses should be consistent across the entire assessment."
            
        except Exception as e:
            logger.error(f"Error generating prompt for persona {persona_id}, {condition}: {str(e)}")
            return {}
        
        # Get model response with retries
        for attempt in range(max_retries + 1):
            try:
                response_text = client.generate_response(prompt)
                break
            except Exception as e:
                if attempt == max_retries:
                    logger.error(f"Failed to get response after {max_retries + 1} attempts for persona {persona_id}, {condition}: {str(e)}")
                    return {}
                else:
                    logger.warning(f"Attempt {attempt + 1} failed for persona {persona_id}, {condition}: {str(e)}. Retrying...")
                    time.sleep(2 ** attempt)  # Exponential backoff
        
        # Parse responses
        numeric_responses = self.parse_model_responses(response_text, len(questions))
        
        # Structure responses by scale and subscale
        condition_responses = {}
        response_idx = 0
        
        for scale_name, scale_data in condition_instruments.items():
            condition_responses[scale_name] = {}
            
            for subscale_name, subscale_data in scale_data["subscales"].items():
                condition_responses[scale_name][subscale_name] = {}
                
                for question_id, question_text in subscale_data["questions"].items():
                    if response_idx < len(numeric_responses):
                        condition_responses[scale_name][subscale_name][question_id] = numeric_responses[response_idx]
                    else:
                        condition_responses[scale_name][subscale_name][question_id] = None
                    response_idx += 1
        
        logger.debug(f"Completed simulation for persona {persona_id}, {condition}")
        return condition_responses
    
    def run_simulation(
        self,
        model_names: List[str],
        conditions: List[str] = CONDITIONS,
        limit_personas: Optional[int] = None,
        output_file: Path = PERSONA_RESPONSES_JSON
    ) -> None:
        """
        Run the complete simulation across models and conditions.
        
        Args:
            model_names: List of model names to use
            conditions: List of experimental conditions
            limit_personas: Optional limit on number of personas
            output_file: Path to save results
        """
        logger.info(f"Starting simulation for models: {model_names}, conditions: {conditions}")
        
        # Load data
        self.load_data()
        
        # Initialize clients
        self.initialize_clients(model_names)
        
        # Prepare personas list
        personas_to_process = self.personas_data["personas"]
        if limit_personas:
            personas_to_process = personas_to_process[:limit_personas]
        
        total_simulations = len(personas_to_process) * len(model_names) * len(conditions)
        logger.info(f"Running {total_simulations} total simulations...")
        
        # Run simulations
        with tqdm(total=total_simulations, desc="Running simulations") as pbar:
            for persona in personas_to_process:
                persona_id = persona["id"]
                
                for model_name in model_names:
                    if model_name not in self.clients:
                        logger.warning(f"Skipping {model_name} - client not available")
                        pbar.update(len(conditions))
                        continue
                    
                    # Create persona entry in results
                    persona_key = f"persona_{persona_id:03d}_{model_name}"
                    
                    if persona_key not in self.simulation_results:
                        self.simulation_results[persona_key] = {
                            "persona_id": persona_id,
                            "model": model_name,
                            "responses": {}
                        }
                    
                    for condition in conditions:
                        try:
                            condition_responses = self.simulate_persona_condition(
                                persona, model_name, condition
                            )
                            
                            self.simulation_results[persona_key]["responses"][condition] = condition_responses
                            
                        except Exception as e:
                            logger.error(f"Error in simulation for persona {persona_id}, {model_name}, {condition}: {str(e)}")
                        
                        pbar.update(1)
                
                # Save intermediate results every 10 personas
                if persona_id % 10 == 0:
                    self.save_results(output_file)
                    logger.info(f"Saved intermediate results after persona {persona_id}")
        
        # Save final results
        self.save_results(output_file)
        
        # Log statistics
        self.log_simulation_stats()
    
    def save_results(self, output_file: Path) -> None:
        """Save simulation results to JSON file."""
        try:
            write_json_file(self.simulation_results, output_file)
            logger.info(f"Saved results to {output_file}")
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
    
    def log_simulation_stats(self) -> None:
        """Log statistics about the simulation."""
        if not self.simulation_results:
            return
        
        total_personas = len(set(result["persona_id"] for result in self.simulation_results.values()))
        total_models = len(set(result["model"] for result in self.simulation_results.values()))
        total_conditions = len(CONDITIONS)
        
        logger.info(f"Simulation completed:")
        logger.info(f"  - Personas: {total_personas}")
        logger.info(f"  - Models: {total_models}")
        logger.info(f"  - Conditions: {total_conditions}")
        logger.info(f"  - Total entries: {len(self.simulation_results)}")
        
        # Log client statistics
        for model_name, client in self.clients.items():
            stats = client.get_stats()
            logger.info(f"  - {model_name}: {stats['request_count']} requests, {stats['total_tokens']} tokens")


def main(
    models: Optional[List[str]] = None,
    conditions: Optional[List[str]] = None,
    limit_personas: Optional[int] = None
) -> None:
    """
    Main function to run LLM simulation.
    
    Args:
        models: List of models to use (default: all available)
        conditions: List of conditions to run (default: all)
        limit_personas: Limit number of personas for testing
    """
    setup_logging()
    
    if models is None:
        models = list(MODELS.keys())
    
    if conditions is None:
        conditions = CONDITIONS
    
    logger.info(f"Starting LLM simulation with models: {models}")
    
    try:
        runner = SimulationRunner(rate_limit=30)  # 30 requests per minute
        runner.run_simulation(
            model_names=models,
            conditions=conditions,
            limit_personas=limit_personas
        )
        
        logger.info("LLM simulation completed successfully!")
        
    except Exception as e:
        logger.error(f"LLM simulation failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    # For testing, run with limited personas and single model
    main(
        models=["gpt-4"],  # Start with just OpenAI for testing
        limit_personas=5   # Test with 5 personas
    )