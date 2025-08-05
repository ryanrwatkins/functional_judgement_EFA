"""Configuration file for EFA project constants and settings."""

import os
from pathlib import Path
from typing import Dict, List

# Base paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PROMPTS_DIR = PROJECT_ROOT / "prompts"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
UTILS_DIR = PROJECT_ROOT / "utils"
TESTS_DIR = PROJECT_ROOT / "tests"

# Input files
PERSONAS_RAW_FILE = DATA_DIR / "personas_combined.parquet"
INSTRUMENTS_FILE = DATA_DIR / "instruments.csv"

# Output files
PERSONAS_JSON = OUTPUTS_DIR / "personas.json"
INSTRUMENTS_JSON = OUTPUTS_DIR / "instruments.json"
PERSONA_RESPONSES_JSON = OUTPUTS_DIR / "persona_responses.json"

# Template files
SYSTEM_PROMPT_TEMPLATE = PROMPTS_DIR / "system_prompt_template.jinja"

# Model configurations
MODELS: Dict[str, Dict] = {
    "gpt-4": {
        "api_base": "https://api.openai.com/v1",
        "model_name": "gpt-4",
        "max_tokens": 1000,
        "temperature": 0.7
    },
    "claude": {
        "api_base": "https://api.anthropic.com",
        "model_name": "claude-3-sonnet-20240229",
        "max_tokens": 1000,
        "temperature": 0.7
    },
    "llama": {
        "model_name": "llama3",
        "max_tokens": 1000,
        "temperature": 0.7
    }
}

# Experimental conditions
CONDITIONS: List[str] = ["condition_1", "condition_2", "condition_3"]

# Logging configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Persona configuration
QUESTIONS_PER_PERSONA = 200
TOTAL_PERSONAS = 1000

# Validation settings
REQUIRED_DEMOGRAPHIC_FIELDS = [
    "age", "gender", "education", "occupation", "location"
]