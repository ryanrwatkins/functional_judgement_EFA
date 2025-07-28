"""
Tests for the convert_personas.py script.
"""
import json

import pandas as pd
import pytest

from config import Config
from scripts.convert_personas import group_personas, main


def make_sample_df() -> pd.DataFrame:
    """
    Create a sample DataFrame with two personas, each with two responses.
    """
    data = {
        'data': ['rev1', 'rev2', 'rev3', 'rev4'],
        'persona': [1, 1, 2, 2],
        'instruction': ['Q1', 'Q2', 'Q1', 'Q2'],
        'original': ['orig1', 'orig2', 'orig3', 'orig4'],
        'critique': ['', '', '', ''],
        'type': ['train', 'train', 'train', 'train'],
    }
    return pd.DataFrame(data)


def test_group_personas_structure():
    df = make_sample_df()
    result = group_personas(df)
    assert 'personas' in result
    assert isinstance(result['personas'], list)
    assert len(result['personas']) == 2


def test_group_personas_content():
    df = make_sample_df()
    result = group_personas(df)
    p1 = next(p for p in result['personas'] if p['id'] == 1)
    assert len(p1['responses']) == 2
    first = p1['responses'][0]
    assert first['question_id'] == 1
    assert first['original_response'] == 'orig1'
    assert first['revised_response'] == 'rev1'


def test_main_creates_json(tmp_path, monkeypatch):
    df = make_sample_df()
    input_file = tmp_path / 'test.parquet'
    df.to_parquet(input_file)
    monkeypatch.setattr(Config, 'PERSONA_INPUT_PATH', input_file)
    output_file = tmp_path / 'out.json'
    monkeypatch.setattr(Config, 'PERSONA_OUTPUT_PATH', output_file)
    # Run conversion
    main()
    # Verify output file exists and has correct structure
    assert output_file.exists()
    data = json.loads(output_file.read_text(encoding='utf-8'))
    assert 'personas' in data
    assert len(data['personas']) == 2
