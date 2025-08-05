"""Tests for persona conversion functionality."""

import json
import pytest
from unittest.mock import patch

from scripts.convert_personas import (
    convert_personas_to_json,
    extract_persona_demographics,
    group_persona_responses,
    validate_persona_data
)
from utils.file_io import FileIOError


class TestExtractPersonaDemographics:
    """Test demographic extraction from persona text."""
    
    def test_extract_basic_demographics(self):
        """Test extraction of basic demographic information."""
        persona_text = "Age: 25, Gender: female, Education: Bachelor's degree, Occupation: teacher, Location: New York"
        
        demographics = extract_persona_demographics(persona_text)
        
        assert demographics["age"] == "25"
        assert demographics["gender"] == "female"
        assert "bachelor" in demographics["education"].lower()
        assert demographics["occupation"] == "teacher"
        assert "new york" in demographics["location"].lower()
    
    def test_extract_with_missing_fields(self):
        """Test extraction when some fields are missing."""
        persona_text = "I am 30 years old and work as an engineer"
        
        demographics = extract_persona_demographics(persona_text)
        
        assert demographics["age"] == "30"
        assert demographics["occupation"] == "engineer"
        assert demographics["gender"] == "unknown"  # Should have default
        assert demographics["education"] == "unknown"
        assert demographics["location"] == "unknown"
    
    def test_extract_from_empty_text(self):
        """Test extraction from empty or invalid text."""
        demographics = extract_persona_demographics("")
        
        # Should return all defaults
        for value in demographics.values():
            assert value == "unknown"
    
    def test_extract_with_variations(self):
        """Test extraction with different text formats."""
        persona_text = "This person is aged 45, identifies as male, graduated from university, works as a doctor, and lives in California"
        
        demographics = extract_persona_demographics(persona_text)
        
        assert demographics["age"] == "45"
        assert demographics["gender"] == "male"
        assert "university" in demographics["education"].lower()
        assert demographics["occupation"] == "doctor"
        assert "california" in demographics["location"].lower()


class TestGroupPersonaResponses:
    """Test grouping of persona responses."""
    
    def test_group_responses_basic(self, sample_parquet_data):
        """Test basic grouping functionality."""
        grouped = group_persona_responses(sample_parquet_data)
        
        assert len(grouped) == 2  # Two personas
        assert "Persona A" in grouped
        assert "Persona B" in grouped
        
        persona_a_responses = grouped["Persona A"]
        assert len(persona_a_responses) == 2
        
        # Check response structure
        response = persona_a_responses[0]
        assert "question_id" in response
        assert "instruction" in response
        assert "original_response" in response
        assert "revised_response" in response
    
    def test_group_responses_ordering(self, sample_parquet_data):
        """Test that responses maintain proper ordering."""
        grouped = group_persona_responses(sample_parquet_data)
        
        persona_a_responses = grouped["Persona A"]
        question_ids = [r["question_id"] for r in persona_a_responses]
        
        # Should be consecutive starting from 1
        assert question_ids == [1, 2]
    
    def test_group_empty_dataframe(self):
        """Test grouping with empty dataframe."""
        import pandas as pd
        empty_df = pd.DataFrame(columns=["data", "persona", "instruction", "original", "critique", "type"])
        
        grouped = group_persona_responses(empty_df)
        assert len(grouped) == 0


class TestValidatePersonaData:
    """Test persona data validation."""
    
    def test_validate_correct_data(self):
        """Test validation with correct data structure."""
        persona_groups = {
            "Persona 1": [{"question_id": i, "original_response": f"Answer {i}", "revised_response": f"Revised {i}"} 
                         for i in range(1, 201)],  # 200 responses
            "Persona 2": [{"question_id": i, "original_response": f"Answer {i}", "revised_response": f"Revised {i}"} 
                         for i in range(1, 201)]
        }
        
        with patch('config.QUESTIONS_PER_PERSONA', 200):
            result = validate_persona_data(persona_groups)
        
        assert result is True
    
    def test_validate_incorrect_response_count(self):
        """Test validation with incorrect number of responses."""
        persona_groups = {
            "Persona 1": [{"question_id": i, "original_response": f"Answer {i}", "revised_response": f"Revised {i}"} 
                         for i in range(1, 101)]  # Only 100 responses
        }
        
        with patch('config.QUESTIONS_PER_PERSONA', 200):
            result = validate_persona_data(persona_groups)
        
        assert result is False
    
    def test_validate_empty_responses(self):
        """Test validation with empty responses."""
        persona_groups = {
            "Persona 1": [
                {"question_id": 1, "original_response": "", "revised_response": "Revised"},
                {"question_id": 2, "original_response": "Answer", "revised_response": ""}
            ]
        }
        
        with patch('config.QUESTIONS_PER_PERSONA', 2):
            result = validate_persona_data(persona_groups)
        
        assert result is False


class TestConvertPersonasToJson:
    """Test complete persona conversion process."""
    
    def test_conversion_success(self, temp_dir, sample_parquet_data, create_test_files):
        """Test successful conversion process."""
        # Create test files
        files = create_test_files(parquet_data=sample_parquet_data)
        input_file = files['parquet']
        output_file = temp_dir / "output_personas.json"
        
        with patch('config.QUESTIONS_PER_PERSONA', 2):  # Match sample data
            result = convert_personas_to_json(input_file, output_file, skip_if_exists=False)
        
        assert result is True
        assert output_file.exists()
        
        # Verify output structure
        with open(output_file) as f:
            data = json.load(f)
        
        assert "personas" in data
        assert len(data["personas"]) == 2
        
        persona = data["personas"][0]
        assert "id" in persona
        assert "demographics" in persona
        assert "responses" in persona
    
    def test_conversion_skip_existing(self, temp_dir):
        """Test skipping when output file already exists."""
        output_file = temp_dir / "existing.json"
        output_file.write_text('{"test": "data"}')
        
        result = convert_personas_to_json(
            temp_dir / "nonexistent.parquet", 
            output_file, 
            skip_if_exists=True
        )
        
        assert result is True  # Should skip and return success
    
    def test_conversion_missing_input(self, temp_dir):
        """Test conversion with missing input file."""
        input_file = temp_dir / "missing.parquet"
        output_file = temp_dir / "output.json"
        
        result = convert_personas_to_json(input_file, output_file, skip_if_exists=False)
        
        assert result is False
    
    def test_conversion_invalid_columns(self, temp_dir, create_test_files):
        """Test conversion with missing required columns."""
        import pandas as pd
        
        # Create DataFrame missing required columns
        invalid_data = pd.DataFrame({
            "wrong_column": ["data1", "data2"],
            "another_wrong": ["data3", "data4"]
        })
        
        files = create_test_files(parquet_data=invalid_data)
        input_file = files['parquet']
        output_file = temp_dir / "output.json"
        
        result = convert_personas_to_json(input_file, output_file, skip_if_exists=False)
        
        assert result is False