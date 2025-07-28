"""
Script to enforce final formatting of the simulated responses JSON.

Preconditions:
    - Simulation output exists at Config.RESPONSES_OUTPUT_PATH

Postconditions:
    - Pretty-printed JSON saved back to Config.RESPONSES_OUTPUT_PATH
"""
import json

from config import Config
from utils.logging_utils import get_logger

logger = get_logger(__name__)


def main() -> None:
    in_path = Config.RESPONSES_OUTPUT_PATH
    if not in_path.exists():
        logger.error(f"{in_path} not found; run run_llm_simulation first.")
        return

    logger.info(f"Loading simulation JSON from {in_path}")
    data = json.loads(in_path.read_text(encoding='utf-8'))

    logger.info(f"Writing formatted JSON back to {in_path}")
    in_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')
    logger.info("Formatting complete.")


if __name__ == '__main__':
    main()
