"""
Tests for the format_outputs.py script.
"""
import json
from pathlib import Path

import pytest

from config import Config
from scripts.format_outputs import main


def test_format_outputs_writes(tmp_path, monkeypatch):
    data = {'x': 1}
    file = tmp_path / 'resp.json'
    file.write_text(json.dumps(data), encoding='utf-8')
    monkeypatch.setattr(Config, 'RESPONSES_OUTPUT_PATH', file)
    main()
    loaded = json.loads(file.read_text(encoding='utf-8'))
    assert loaded == data


def test_format_outputs_missing(tmp_path, monkeypatch, caplog):
    missing = tmp_path / 'no.json'
    monkeypatch.setattr(Config, 'RESPONSES_OUTPUT_PATH', missing)
    main()
    assert f"{missing} not found" in caplog.text
