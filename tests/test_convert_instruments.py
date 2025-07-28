"""Tests for instruments conversion functionality."""

import json
import pytest
from unittest.mock import patch

from scripts.convert_instruments import (
    clean_text_field,
    convert_instruments_to_json,
    extract_response_scale_info,
    group_instruments_by_scale,
    validate_instruments_data
)


class TestCleanTextField:
    """Test text field cleaning functionality."""
    
    def test_clean_normal_text(self):
        """Test cleaning normal text."""
        result = clean_text_field("  Normal text  ")
        assert result == "Normal text"
    
    def test_clean_multiline_text(self):
        """Test cleaning text with multiple spaces and newlines."""
        result = clean_text_field("Text  with\n\textra   spaces")
        assert result == "Text with extra spaces"
    
    def test_clean_empty_text(self):
        """Test cleaning empty or None text."""
        assert clean_text_field("") == ""
        assert clean_text_field(None) == ""
        assert clean_text_field("   ") == ""
    
    def test_clean_numeric_input(self):
        """Test cleaning numeric input."""
        result = clean_text_field(123)
        assert result == "123"


class TestExtractResponseScaleInfo:
    """Test response scale information extraction."""
    
    def test_extract_likert_scale(self):
        """Test extraction of Likert scale information."""
        scale_text = "Rate on a 5-point Likert scale from 1 (strongly disagree) to 5 (strongly agree)"
        result = extract_response_scale_info(scale_text)
        
        assert result is not None
        assert "likert" in result.lower()
        assert "strongly" in result.lower()
    
    def test_extract_rating_scale(self):
        """Test extraction of rating scale information."""
        scale_text = "Please rate each item from 1 to 7"
        result = extract_response_scale_info(scale_text)
        
        assert result is not None
        assert "rate" in result.lower()
    
    def test_extract_non_scale_text(self):
        """Test with text that doesn't contain scale information."""
        scale_text = "This is just a description without scale info"
        result = extract_response_scale_info(scale_text)
        
        assert result is None
    
    def test_extract_empty_text(self):
        """Test extraction from empty text."""
        assert extract_response_scale_info("") is None
        assert extract_response_scale_info(None) is None


class TestGroupInstrumentsByScale:
    """Test grouping instruments by scale and subscale."""
    
    def test_group_basic_structure(self, sample_csv_data):
        """Test basic grouping structure."""
        grouped = group_instruments_by_scale(sample_csv_data)
        
        assert "Big Five" in grouped
        assert "Grit" in grouped
        
        big_five = grouped["Big Five"]
        assert "scale_id" in big_five
        assert "subscales" in big_five
        assert "Openness" in big_five["subscales"]
        assert "Conscientiousness" in big_five["subscales"]
    
    def test_group_questions_structure(self, sample_csv_data):
        """Test that questions are properly grouped."""
        grouped = group_instruments_by_scale(sample_csv_data)
        
        openness = grouped["Big Five"]["subscales"]["Openness"]
        assert "questions" in openness
        assert "1" in openness["questions"]
        assert openness["questions"]["1"] == "I am creative"
    
    def test_group_scale_ids(self, sample_csv_data):
        """Test that scale IDs are assigned correctly."""
        grouped = group_instruments_by_scale(sample_csv_data)
        
        scale_ids = [data["scale_id"] for data in grouped.values()]
        assert len(set(scale_ids)) == len(scale_ids)  # All unique
        assert min(scale_ids) == 1  # Starting from 1
    
    def test_group_empty_dataframe(self):
        """Test grouping with empty DataFrame."""
        import pandas as pd
        empty_df = pd.DataFrame(columns=["number", "item", "subscale", "scale", "response scale"])
        
        grouped = group_instruments_by_scale(empty_df)
        assert len(grouped) == 0
    
    def test_group_with_missing_subscales(self):
        """Test grouping when subscale field is empty."""
        import pandas as pd
        data = pd.DataFrame({
            "number": ["1", "2"],
            "item": ["Question 1", "Question 2"],
            "subscale": ["", ""],  # Empty subscales
            "scale": ["Test Scale", "Test Scale"],
            "response scale": ["1-5", "1-5"]
        })
        
        grouped = group_instruments_by_scale(data)
        
        assert "Test Scale" in grouped
        assert "General" in grouped["Test Scale"]["subscales"]  # Default subscale name


class TestValidateInstrumentsData:
    """Test instruments data validation."""
    
    def test_validate_correct_structure(self, sample_instruments_data):
        """Test validation with correct data structure."""
        result = validate_instruments_data(sample_instruments_data)
        assert result is True
    
    def test_validate_missing_scale_id(self):
        """Test validation with missing scale_id."""
        invalid_data = {
            "Test Scale": {
                # Missing scale_id
                "subscales": {
                    "Test Subscale": {
                        "questions": {"1": "Test question"}
                    }
                }
            }
        }
        
        result = validate_instruments_data(invalid_data)
        assert result is False
    
    def test_validate_empty_subscales(self):
        """Test validation with empty subscales."""
        invalid_data = {
            "Test Scale": {
                "scale_id": 1,
                "subscales": {}  # Empty subscales
            }
        }
        
        result = validate_instruments_data(invalid_data)
        assert result is False
    
    def test_validate_empty_questions(self):
        """Test validation with empty questions."""
        invalid_data = {
            "Test Scale": {
                "scale_id": 1,
                "subscales": {
                    "Test Subscale": {
                        "questions": {}  # Empty questions
                    }
                }
            }
        }
        
        result = validate_instruments_data(invalid_data)
        assert result is False


class TestConvertInstrumentsToJson:
    """Test complete instruments conversion process."""
    
    def test_conversion_success(self, temp_dir, sample_csv_data, create_test_files):
        """Test successful conversion process."""
        files = create_test_files(csv_data=sample_csv_data)
        input_file = files['csv']
        output_file = temp_dir / "output_instruments.json"
        
        result = convert_instruments_to_json(input_file, output_file, skip_if_exists=False)
        
        assert result is True
        assert output_file.exists()
        
        # Verify output structure
        with open(output_file) as f:
            data = json.load(f)
        
        assert "Big Five" in data
        assert "Grit" in data
        
        big_five = data["Big Five"]
        assert "scale_id" in big_five
        assert "subscales" in big_five
    
    def test_conversion_skip_existing(self, temp_dir):
        """Test skipping when output file already exists."""
        output_file = temp_dir / "existing.json"
        output_file.write_text('{"test": "data"}')
        
        result = convert_instruments_to_json(
            temp_dir / "nonexistent.csv",
            output_file,
            skip_if_exists=True
        )
        
        assert result is True  # Should skip
    
    def test_conversion_missing_input(self, temp_dir):
        """Test conversion with missing input file."""
        input_file = temp_dir / "missing.csv"
        output_file = temp_dir / "output.json"
        
        result = convert_instruments_to_json(input_file, output_file, skip_if_exists=False)
        
        assert result is False
    
    def test_conversion_invalid_columns(self, temp_dir, create_test_files):
        """Test conversion with missing required columns."""
        import pandas as pd
        
        invalid_data = pd.DataFrame({
            "wrong_column": ["data1", "data2"],
            "another_wrong": ["data3", "data4"]
        })
        
        files = create_test_files(csv_data=invalid_data)
        input_file = files['csv']
        output_file = temp_dir / "output.json"
        
        result = convert_instruments_to_json(input_file, output_file, skip_if_exists=False)
        
        assert result is False
    
    def test_conversion_data_cleaning(self, temp_dir, create_test_files):
        """Test that data cleaning removes invalid rows."""
        import pandas as pd
        
        # Create data with some invalid rows
        data_with_nulls = pd.DataFrame({
            "number": ["1", None, "3"],  # Null number
            "item": ["Question 1", "Question 2", None],  # Null item
            "subscale": ["Sub1", "Sub2", "Sub3"],
            "scale": ["Scale1", None, "Scale3"],  # Null scale
            "response scale": ["1-5", "1-5", "1-5"]
        })
        
        files = create_test_files(csv_data=data_with_nulls)
        input_file = files['csv']
        output_file = temp_dir / "output.json"
        
        result = convert_instruments_to_json(input_file, output_file, skip_if_exists=False)
        
        # Should succeed but with fewer items
        assert result is True
        
        with open(output_file) as f:
            data = json.load(f)
        
        # Should only have valid scales
        assert "Scale1" in data
        assert "Scale3" in data