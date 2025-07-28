"""
Script to convert psychological instruments CSV into a structured JSON file.

Preconditions:
    - Input file exists at the configured INSTRUMENTS_INPUT_PATH.

Postconditions:
    - Output JSON file written to the configured INSTRUMENTS_OUTPUT_PATH.
"""
from collections import defaultdict
from pathlib import Path

import pandas as pd

from config import Config
from utils.file_io import write_json
from utils.logging_utils import get_logger

logger = get_logger(__name__)


def group_instruments(df: pd.DataFrame) -> dict:
    """
    Group instrument items by scale and subscale, capturing response instructions.

    :param df: DataFrame with columns ['number', 'item', 'subscale', 'scale', 'response scale']
    :return: Nested dict of scales.
    """
    scales = {}
    # Assign unique IDs to each scale
    scale_names = df['scale'].unique()
    for idx, scale_name in enumerate(scale_names, start=1):
        scales[scale_name] = {'scale_id': idx}

    # Collect rows by scale and subscale
    for _, row in df.iterrows():
        scale_block = scales[row['scale']]
        # Capture scale-level response_scale if subscale is empty
        resp = row.get('response scale') or row.get('response_scale')
        sub = row.get('subscale')
        # Initialize response_scale keys
        if resp and (pd.isna(sub) or sub == ''):
            scale_block['response_scale'] = resp
        # Process subscale items
        if pd.notna(sub) and sub != '':
            if sub not in scale_block:
                scale_block[sub] = {}
            # Attach subscale-level response scale if present
            if resp:
                scale_block[sub]['response_scale'] = resp
            # Add the item to the subscale dict
            scale_block[sub][str(row['number'])] = row['item']
        # If no subscale, attach items directly under scale
        else:
            # Items without subscale go under a default key
            scale_block.setdefault('items', {})[str(row['number'])] = row['item']
    return scales


def main() -> None:
    input_path: Path = Config.INSTRUMENTS_INPUT_PATH
    output_path: Path = Config.INSTRUMENTS_OUTPUT_PATH

    if output_path.exists():
        logger.info(f"{output_path} already exists; skipping conversion.")
        return

    logger.info(f"Loading instruments from {input_path}")
    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df)} rows")

    result = group_instruments(df)
    write_json(result, output_path)
    logger.info(f"Wrote instruments JSON to {output_path}")


if __name__ == '__main__':
    main()
