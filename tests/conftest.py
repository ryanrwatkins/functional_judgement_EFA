"""Pytest configuration and fixtures for EFA project tests."""

import json
import tempfile
from pathlib import Path
from typing import Dict, List

import pandas as pd
import pytest


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def sample_persona_data():
    """Create sample persona data for testing."""
    return {
        "personas": [
            {
                "id": 1,
                "name": "Test Persona 1",
                "demographics": {
                    "age": "25",
                    "gender": "female",
                    "education": "bachelor's degree",
                    "occupation": "teacher",
                    "location": "New York"
                },
                "responses": [
                    {
                        "question_id": 1,
                        "instruction": "What is your favorite color?",
                        "original_response": "Blue",
                        "revised_response": "I prefer blue colors",
                        "critique": "Good response",
                        "data_type": "train"
                    },
                    {
                        "question_id": 2,
                        "instruction": "Describe your work style",
                        "original_response": "Collaborative",
                        "revised_response": "I work best in collaborative environments",
                        "critique": "Clear description",
                        "data_type": "train"
                    }
                ]
            },
            {
                "id": 2,
                "name": "Test Persona 2", 
                "demographics": {
                    "age": "35",
                    "gender": "male",
                    "education": "master's degree",
                    "occupation": "engineer",
                    "location": "California"
                },
                "responses": [
                    {
                        "question_id": 1,
                        "instruction": "What is your favorite color?",
                        "original_response": "Green",
                        "revised_response": "I like green shades",
                        "critique": "Good choice",
                        "data_type": "test"
                    }
                ]
            }
        ]
    }


@pytest.fixture
def sample_instruments_data():
    """Create sample instruments data for testing."""
    return {
        "Big Five": {
            "scale_id": 1,
            "response_scale": "Rate from 1 (strongly disagree) to 5 (strongly agree)",
            "subscales": {
                "Openness": {
                    "response_scale": None,
                    "questions": {
                        "1": "I am original, come up with new ideas",
                        "2": "I value artistic, aesthetic experiences"
                    }
                },
                "Conscientiousness": {
                    "response_scale": None,
                    "questions": {
                        "3": "I do a thorough job",
                        "4": "I tend to be lazy"
                    }
                }
            }
        },
        "Grit Scale": {
            "scale_id": 2,
            "response_scale": "Rate from 1 (not like me at all) to 5 (very much like me)",
            "subscales": {
                "Perseverance": {
                    "response_scale": None,
                    "questions": {
                        "1": "I finish whatever I begin",
                        "2": "I am diligent"
                    }
                }
            }
        }
    }


@pytest.fixture
def sample_parquet_data():
    """Create sample parquet data for testing."""
    data = {
        "data": ["I prefer blue", "I work collaboratively", "I like green", "I am creative"],
        "persona": ["Persona A", "Persona A", "Persona B", "Persona B"], 
        "instruction": ["What is your color?", "How do you work?", "What is your color?", "Are you creative?"],
        "original": ["Blue", "Collaborative", "Green", "Creative"],
        "critique": ["Good", "Clear", "Nice", "Great"],
        "type": ["train", "train", "test", "test"]
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_csv_data():
    """Create sample CSV data for testing."""
    data = {
        "number": ["1", "2", "3", "4"],
        "item": ["I am creative", "I am organized", "I persist", "I am disciplined"],
        "subscale": ["Openness", "Conscientiousness", "Perseverance", "Perseverance"],
        "scale": ["Big Five", "Big Five", "Grit", "Grit"],
        "response scale": ["1-5 scale", "1-5 scale", "1-5 scale", "1-5 scale"]
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_simulation_data():
    """Create sample simulation results data for testing."""
    return {
        "persona_001_gpt-4": {
            "persona_id": 1,
            "model": "gpt-4",
            "responses": {
                "condition_1": {
                    "Big Five": {
                        "Openness": {
                            "1": 4,
                            "2": 5
                        },
                        "Conscientiousness": {
                            "3": 5,
                            "4": 2
                        }
                    }
                },
                "condition_2": {
                    "Big Five": {
                        "Openness": {
                            "1": 4,
                            "2": 4
                        },
                        "Conscientiousness": {
                            "3": 5,
                            "4": 2
                        }
                    }
                },
                "condition_3": {
                    "Big Five": {
                        "Openness": {
                            "1": 3,
                            "2": 4
                        },
                        "Conscientiousness": {
                            "3": 4,
                            "4": 3
                        }
                    }
                }
            }
        }
    }


@pytest.fixture
def mock_openai_response():
    """Mock OpenAI API response."""
    class MockChoice:
        def __init__(self, content):
            self.message = MockMessage(content)
    
    class MockMessage:
        def __init__(self, content):
            self.content = content
    
    class MockUsage:
        def __init__(self):
            self.total_tokens = 100
    
    class MockResponse:
        def __init__(self, content):
            self.choices = [MockChoice(content)]
            self.usage = MockUsage()
    
    return MockResponse


@pytest.fixture 
def mock_anthropic_response():
    """Mock Anthropic API response."""
    class MockContent:
        def __init__(self, text):
            self.text = text
    
    class MockUsage:
        def __init__(self):
            self.input_tokens = 50
            self.output_tokens = 50
    
    class MockResponse:
        def __init__(self, text):
            self.content = [MockContent(text)]
            self.usage = MockUsage()
    
    return MockResponse


@pytest.fixture
def create_test_files(temp_dir):
    """Factory to create test files in temp directory."""
    def _create_files(persona_data=None, instruments_data=None, parquet_data=None, csv_data=None):
        files = {}
        
        if persona_data:
            persona_file = temp_dir / "personas.json"
            with open(persona_file, 'w') as f:
                json.dump(persona_data, f)
            files['personas'] = persona_file
        
        if instruments_data:
            instruments_file = temp_dir / "instruments.json"
            with open(instruments_file, 'w') as f:
                json.dump(instruments_data, f)
            files['instruments'] = instruments_file
        
        if parquet_data is not None:
            parquet_file = temp_dir / "personas.parquet"
            parquet_data.to_parquet(parquet_file)
            files['parquet'] = parquet_file
        
        if csv_data is not None:
            csv_file = temp_dir / "instruments.csv"
            csv_data.to_csv(csv_file, index=False)
            files['csv'] = csv_file
        
        return files
    
    return _create_files