"""
Script to convert persona Parquet dataset into a structured JSON file.

Preconditions:
    - Input file exists at the configured PERSONA_INPUT_PATH.

Postconditions:
    - Output JSON file written to the configured PERSONA_OUTPUT_PATH.
"""
from pathlib import Path

import pandas as pd

from config import Config
from utils.file_io import read_parquet, write_json
from utils.logging_utils import get_logger

logger = get_logger(__name__)


def parse_demographics(persona_df: pd.DataFrame) -> dict:
    """
    Parse demographic information for a persona from its subset of rows.

    Currently a stub; returns an empty dict.
    """
    logger.debug(f"Parsing demographics for persona {persona_df.name}")
    # TODO: Implement regex-based parsing from demographic text block
    return {}


def group_personas(df: pd.DataFrame) -> dict:
    """
    Group rows by persona and build structured persona objects.

    :param df: DataFrame containing persona responses.
    :return: Dict with key 'personas' and list of persona objects.
    """
    personas = []
    for persona_id, group in df.groupby('persona'):
        responses = []
        for idx, row in enumerate(group.itertuples(index=False), start=1):
            responses.append({
                'question_id': idx,
                'original_response': row.original,
                'revised_response': row.data,
            })
        demographics = parse_demographics(group)
        personas.append({
            'id': int(persona_id),
            'demographics': demographics,
            'responses': responses,
        })
    return {'personas': personas}


def main() -> None:
    input_path: Path = Config.PERSONA_INPUT_PATH
    output_path: Path = Config.PERSONA_OUTPUT_PATH

    if output_path.exists():
        logger.info(f"{output_path} already exists; skipping conversion.")
        return

    logger.info(f"Loading personas from {input_path}")
    df = read_parquet(input_path)
    logger.info(f"Loaded {len(df)} rows")

    result = group_personas(df)
    write_json(result, output_path)
    logger.info(f"Wrote personas JSON to {output_path}")


if __name__ == '__main__':
    main()
