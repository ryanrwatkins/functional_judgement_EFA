"""
Tests for the convert_instruments.py script.
"""
import json

import pandas as pd
import pytest

from config import Config
from scripts.convert_instruments import group_instruments, main


def make_sample_df() -> pd.DataFrame:
    """
    Create a sample instruments DataFrame with two scales and subscales.
    """
    data = [
        # Scale-level response instruction row (no subscale)
        {'number': 1, 'item': 'Scale1 ItemA', 'subscale': '', 'scale': 'Scale1', 'response scale': '1-5 Likert'},
        # Subscale items for Scale1
        {'number': 2, 'item': 'Sub1 Item1', 'subscale': 'Sub1', 'scale': 'Scale1', 'response scale': ''},
        {'number': 3, 'item': 'Sub1 Item2', 'subscale': 'Sub1', 'scale': 'Scale1', 'response scale': ''},
        {'number': 4, 'item': 'Sub2 Item1', 'subscale': 'Sub2', 'scale': 'Scale1', 'response scale': ''},
        # Another scale without subscales
        {'number': 5, 'item': 'Scale2 Item1', 'subscale': '', 'scale': 'Scale2', 'response scale': 'Yes/No'},
        {'number': 6, 'item': 'Scale2 Item2', 'subscale': '', 'scale': 'Scale2', 'response scale': ''},
    ]
    return pd.DataFrame(data)


def test_group_instruments_structure_and_content():
    df = make_sample_df()
    result = group_instruments(df)
    # Two scales
    assert set(result.keys()) == {'Scale1', 'Scale2'}
    # Check scale IDs
    assert result['Scale1']['scale_id'] == 1
    assert result['Scale2']['scale_id'] == 2
    # Scale1 should have response_scale
    assert result['Scale1']['response_scale'] == '1-5 Likert'
    # Subscales under Scale1
    assert 'Sub1' in result['Scale1']
    assert 'Sub2' in result['Scale1']
    # Items under Sub1
    sub1 = result['Scale1']['Sub1']
    assert sub1['1'] == 'Scale1 ItemA' or '2' in sub1
    assert sub1['2'] == 'Sub1 Item1'
    assert sub1['3'] == 'Sub1 Item2'
    # Scale2 items under 'items'
    assert 'items' in result['Scale2']
    assert result['Scale2']['items']['5'] == 'Scale2 Item1'


def test_main_creates_json(tmp_path, monkeypatch):
    df = make_sample_df()
    input_file = tmp_path / 'instr.csv'
    df.to_csv(input_file, index=False)
    monkeypatch.setattr(Config, 'INSTRUMENTS_INPUT_PATH', input_file)
    output_file = tmp_path / 'instr_out.json'
    monkeypatch.setattr(Config, 'INSTRUMENTS_OUTPUT_PATH', output_file)
    # Run conversion
    main()
    assert output_file.exists()
    data = json.loads(output_file.read_text(encoding='utf-8'))
    # Check keys and substructures
    assert 'Scale1' in data and 'Scale2' in data
    assert data['Scale1']['scale_id'] == 1
