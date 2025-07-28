import os

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")

# Model Names
GPT_MODEL = "gpt-3.5-turbo"
CLAUDE_MODEL = "claude-3-opus-20240229"
LLAMA_MODEL = "llama3"

# File Paths
PERSONAS_PATH = "outputs/personas.json"
INSTRUMENTS_PATH = "outputs/instruments.json"
PROMPT_TEMPLATE_PATH = "prompts/system_prompt_template.jinja"
PROMPTS_OUTPUT_PATH = "outputs/prompts.json"
RESPONSES_OUTPUT_PATH = "outputs/persona_responses.json"
