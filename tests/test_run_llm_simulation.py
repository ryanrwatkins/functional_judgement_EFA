"""
Test the run_llm_simulation script logic with stub clients.
"""
import json
import tempfile

import pytest

from config import Config
from scripts.run_llm_simulation import main


def test_main_writes_file(tmp_path, monkeypatch, caplog):
    # Prepare minimal personas JSON
    personas = {'personas': [{'id': 1, 'demographics': {}, 'responses': []}]}
    instr = {'ScaleX': {'scale_id': 1, 'response_scale': '1-5', 'items': {'1': 'Q1'}}}
    persona_file = tmp_path / 'p.json'
    instr_file = tmp_path / 'i.json'
    out_file = tmp_path / 'out.json'
    persona_file.write_text(json.dumps(personas))
    instr_file.write_text(json.dumps(instr))
    monkeypatch.setattr(Config, 'PERSONA_OUTPUT_PATH', persona_file)
    monkeypatch.setattr(Config, 'INSTRUMENTS_OUTPUT_PATH', instr_file)
    monkeypatch.setattr(Config, 'RESPONSES_OUTPUT_PATH', out_file)
    # Run simulation
    main()
    assert out_file.exists()
    data = json.loads(out_file.read_text())
    # Expect key persona_1_openai etc
    for model in ('openai', 'claude', 'llama'):
        key = f'persona_1_{model}'
        assert key in data
        entry = data[key]
        assert entry['persona_id'] == 1
        assert 'condition_1' in entry['responses']

def test_missing_inputs(monkeypatch, caplog):
    monkeypatch.setattr(Config, 'PERSONA_OUTPUT_PATH', tmp_path := tempfile.mkdtemp())
    monkeypatch.setattr(Config, 'INSTRUMENTS_OUTPUT_PATH', tmp_path)
    main()
    assert 'Missing input JSON files' in caplog.text
