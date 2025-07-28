"""
Script to simulate LLM responses for each persona under multiple experimental conditions.

Preconditions:
    - personas.json exists at Config.PERSONA_OUTPUT_PATH
    - instruments.json exists at Config.INSTRUMENTS_OUTPUT_PATH

Postconditions:
    - responses JSON written to Config.RESPONSES_OUTPUT_PATH
"""
import json
import random
from pathlib import Path
from typing import Any, Dict

from config import Config
from utils.llm_clients import OpenAIClient, ClaudeClient, LocalLlamaClient
from utils.logging_utils import get_logger

logger = get_logger(__name__)


def simulate_responses(
    persona: Dict[str, Any], instruments: Dict[str, Any], client: Any, condition: int
) -> Dict[str, Any]:
    """
    Generate simulated responses for a single persona, LLM client, and experimental condition.

    :param persona: Persona dict with demographics and responses.
    :param instruments: Nested instrument structure.
    :param client: LLM client with .chat(prompt) -> numeric response.
    :param condition: Experimental condition (1, 2, or 3).
    :return: Nested dict of simulated responses by scale and subscale.
    """
    scales = list(instruments.keys())
    if condition == 3:
        random.shuffle(scales)

    results: Dict[str, Any] = {}
    for scale in scales:
        block = instruments[scale]
        results.setdefault(scale, {})
        targets = []
        # collect all prompt items per scale/subscale
        for sub, subblock in block.items():
            if sub in ['scale_id', 'response_scale', 'items']:
                continue
            for item_id, text in subblock.items():
                if item_id == 'response_scale':
                    continue
                targets.append((scale, sub, item_id, text))
        if 'items' in block:
            for item_id, text in block['items'].items():
                targets.append((scale, 'items', item_id, text))

        # determine grouping for context (ignored in stub)
        for scale_name, sub, item_id, text in targets:
            # each question separately or as grouped -- stub ignores context
            results[scale_name].setdefault(sub, {})[item_id] = client.chat(text)
    return results


def main() -> None:
    persona_file = Config.PERSONA_OUTPUT_PATH
    instr_file = Config.INSTRUMENTS_OUTPUT_PATH
    out_file = Config.RESPONSES_OUTPUT_PATH

    if not persona_file.exists() or not instr_file.exists():
        logger.error("Missing input JSON files; run conversion and prompt scripts first.")
        return

    logger.info(f"Loading personas from {persona_file}")
    personas = json.loads(persona_file.read_text(encoding='utf-8'))['personas']
    logger.info(f"Loading instruments from {instr_file}")
    instruments = json.loads(instr_file.read_text(encoding='utf-8'))

    # initialize clients with fixed seed for reproducibility
    random.seed(Config.SEED)
    clients = {
        'openai': OpenAIClient(Config.OPENAI_MODEL, seed=Config.SEED),
        'claude': ClaudeClient(Config.CLAUDE_MODEL, seed=Config.SEED),
        'llama': LocalLlamaClient(Config.LLAMA_MODEL, seed=Config.SEED),
    }

    all_results: Dict[str, Any] = {}
    for model_name, client in clients.items():
        for persona in personas:
            pid = persona['id']
            key = f"persona_{pid}_{model_name}"
            entry: Dict[str, Any] = {
                'persona_id': pid,
                'model': model_name,
                'responses': {}
            }
            for cond in (1, 2, 3):
                res = simulate_responses(persona, instruments, client, cond)
                entry['responses'][f'condition_{cond}'] = res
            all_results[key] = entry

    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    Path(out_file).write_text(json.dumps(all_results, indent=2), encoding='utf-8')
    logger.info(f"Saved all simulated responses to {out_file}")


if __name__ == '__main__':
    main()
