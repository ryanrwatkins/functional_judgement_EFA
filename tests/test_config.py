"""Tests for configuration module."""

import pytest
from pathlib import Path

from config import (
    CONDITIONS,
    DATA_DIR,
    MODELS,
    OUTPUTS_DIR,
    PROJECT_ROOT,
    QUESTIONS_PER_PERSONA,
    TOTAL_PERSONAS
)


def test_project_structure():
    """Test that project paths are correctly defined."""
    assert PROJECT_ROOT.exists()
    assert PROJECT_ROOT.is_dir()


def test_experimental_conditions():
    """Test experimental conditions configuration."""
    assert len(CONDITIONS) == 3
    assert "condition_1" in CONDITIONS
    assert "condition_2" in CONDITIONS
    assert "condition_3" in CONDITIONS


def test_models_configuration():
    """Test models configuration."""
    assert "gpt-4" in MODELS
    assert "claude" in MODELS
    assert "llama" in MODELS
    
    # Check that each model has required fields
    for model_name, model_config in MODELS.items():
        assert "model_name" in model_config
        assert "max_tokens" in model_config
        assert "temperature" in model_config


def test_persona_configuration():
    """Test persona configuration constants."""
    assert QUESTIONS_PER_PERSONA == 200
    assert TOTAL_PERSONAS == 1000


def test_directory_configuration():
    """Test directory paths configuration."""
    assert DATA_DIR == PROJECT_ROOT / "data"
    assert OUTPUTS_DIR == PROJECT_ROOT / "outputs"