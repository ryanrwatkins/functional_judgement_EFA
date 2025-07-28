"""
Script to render system prompts per persona using a Jinja2 template.

Preconditions:
    - personas.json exists at Config.PERSONA_OUTPUT_PATH
    - instruments.json exists at Config.INSTRUMENTS_OUTPUT_PATH
    - Template file exists at Config.PROMPT_TEMPLATE_PATH

Postconditions:
    - A prompt text file per persona is written under Config.GENERATED_PROMPTS_DIR
"""
import json
from pathlib import Path

from config import Config
from utils.logging_utils import get_logger
from utils.prompt_utils import render_jinja_template

logger = get_logger(__name__)


def main() -> None:
    persona_file = Config.PERSONA_OUTPUT_PATH
    instr_file = Config.INSTRUMENTS_OUTPUT_PATH
    template_file = Config.PROMPT_TEMPLATE_PATH
    out_dir = Config.GENERATED_PROMPTS_DIR

    if not persona_file.exists() or not instr_file.exists():
        logger.error("Personas or instruments JSON not found; run conversion scripts first.")
        return

    logger.info(f"Loading personas from {persona_file}")
    personas = json.loads(persona_file.read_text(encoding='utf-8'))['personas']

    logger.info(f"Loading instruments from {instr_file}")
    instruments = json.loads(instr_file.read_text(encoding='utf-8'))

    out_dir.mkdir(parents=True, exist_ok=True)
    for persona in personas:
        pid = persona.get('id')
        context = {'persona': persona, 'instruments': instruments}
        prompt_text = render_jinja_template(template_file, context)
        out_path = out_dir / f'persona_{pid}_prompt.txt'
        out_path.write_text(prompt_text, encoding='utf-8')
        logger.info(f"Rendered prompt for persona {pid} to {out_path}")


if __name__ == '__main__':
    main()
