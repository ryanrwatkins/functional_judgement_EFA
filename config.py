from pathlib import Path


class Config:
    """
    Configuration constants for file paths used in scripts.
    """
    # Input paths
    PERSONA_INPUT_PATH: Path = Path('data/personas_combined.parquet')
    INSTRUMENTS_INPUT_PATH: Path = Path('data/instruments.csv')

    # Output paths
    PERSONA_OUTPUT_PATH: Path = Path('outputs/personas.json')
    INSTRUMENTS_OUTPUT_PATH: Path = Path('outputs/instruments.json')

    # Prompt template and output directory
    PROMPT_TEMPLATE_PATH: Path = Path('prompts/system_prompt_template.jinja')
    GENERATED_PROMPTS_DIR: Path = Path('outputs/prompts')

    # LLM simulation settings
    OPENAI_MODEL: str = 'gpt-4'
    CLAUDE_MODEL: str = 'claude-v1'
    LLAMA_MODEL: str = 'local-llama'
    SEED: int = 42
    RESPONSES_OUTPUT_PATH: Path = Path('outputs/persona_responses.json')
