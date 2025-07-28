"""File I/O utilities for the EFA project."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)


class FileIOError(Exception):
    """Custom exception for file I/O operations."""
    pass


def read_parquet_file(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Read a parquet file and return as DataFrame.
    
    Args:
        file_path: Path to the parquet file
        
    Returns:
        DataFrame containing the parquet data
        
    Raises:
        FileIOError: If file cannot be read or doesn't exist
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileIOError(f"Parquet file not found: {file_path}")
    
    try:
        df = pd.read_parquet(file_path)
        logger.info(f"Successfully read parquet file: {file_path} ({len(df)} rows)")
        return df
    except Exception as e:
        raise FileIOError(f"Error reading parquet file {file_path}: {str(e)}")


def read_csv_file(file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
    """
    Read a CSV file and return as DataFrame.
    
    Args:
        file_path: Path to the CSV file
        **kwargs: Additional arguments for pd.read_csv
        
    Returns:
        DataFrame containing the CSV data
        
    Raises:
        FileIOError: If file cannot be read or doesn't exist
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileIOError(f"CSV file not found: {file_path}")
    
    try:
        df = pd.read_csv(file_path, **kwargs)
        logger.info(f"Successfully read CSV file: {file_path} ({len(df)} rows)")
        return df
    except Exception as e:
        raise FileIOError(f"Error reading CSV file {file_path}: {str(e)}")


def write_json_file(
    data: Union[Dict, List], 
    file_path: Union[str, Path], 
    indent: int = 2,
    ensure_ascii: bool = False
) -> None:
    """
    Write data to a JSON file.
    
    Args:
        data: Data to write (dict or list)
        file_path: Path where to write the JSON file
        indent: JSON indentation level
        ensure_ascii: Whether to ensure ASCII encoding
        
    Raises:
        FileIOError: If file cannot be written
    """
    file_path = Path(file_path)
    
    # Create parent directory if it doesn't exist
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)
        logger.info(f"Successfully wrote JSON file: {file_path}")
    except Exception as e:
        raise FileIOError(f"Error writing JSON file {file_path}: {str(e)}")


def read_json_file(file_path: Union[str, Path]) -> Union[Dict, List]:
    """
    Read a JSON file and return the data.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Data from the JSON file
        
    Raises:
        FileIOError: If file cannot be read or doesn't exist
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileIOError(f"JSON file not found: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Successfully read JSON file: {file_path}")
        return data
    except json.JSONDecodeError as e:
        raise FileIOError(f"Invalid JSON in file {file_path}: {str(e)}")
    except Exception as e:
        raise FileIOError(f"Error reading JSON file {file_path}: {str(e)}")


def file_exists_and_not_empty(file_path: Union[str, Path]) -> bool:
    """
    Check if a file exists and is not empty.
    
    Args:
        file_path: Path to check
        
    Returns:
        True if file exists and is not empty
    """
    file_path = Path(file_path)
    return file_path.exists() and file_path.stat().st_size > 0