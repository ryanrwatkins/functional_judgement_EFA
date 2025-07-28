"""
Utility functions for reading and writing data files.
"""
import json
from pathlib import Path

import pandas as pd


def read_parquet(path: Path) -> pd.DataFrame:
    """
    Read a Parquet file into a pandas DataFrame.

    :param path: Path to the Parquet file.
    :return: DataFrame loaded from the file.
    """
    return pd.read_parquet(path)


def write_json(data: object, path: Path) -> None:
    """
    Write a Python object to a JSON file with indentation.

    :param data: Python object to serialize.
    :param path: Path to the output JSON file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(mode='w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
