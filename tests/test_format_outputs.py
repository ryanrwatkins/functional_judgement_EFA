"""Tests for output formatting functionality."""

import json
import pytest
from unittest.mock import patch

from scripts.format_outputs import OutputFormatter


class TestOutputFormatter:
    """Test output formatting functionality."""
    
    def test_load_simulation_data_success(self, temp_dir, sample_simulation_data, create_test_files):
        """Test successful loading of simulation data."""
        files = create_test_files()
        simulation_file = temp_dir / "simulation_results.json"
        
        with open(simulation_file, 'w') as f:
            json.dump(sample_simulation_data, f)
        
        formatter = OutputFormatter()
        result = formatter.load_simulation_data(simulation_file)
        
        assert result is True
        assert formatter.simulation_data == sample_simulation_data
    
    def test_load_simulation_data_missing_file(self, temp_dir):
        """Test loading with missing file."""
        formatter = OutputFormatter()
        result = formatter.load_simulation_data(temp_dir / "missing.json")
        
        assert result is False
        assert formatter.simulation_data is None
    
    def test_validate_simulation_data_success(self, sample_simulation_data):
        """Test validation with correct data structure."""
        formatter = OutputFormatter()
        formatter.simulation_data = sample_simulation_data
        
        result = formatter.validate_simulation_data()
        assert result is True
        assert len(formatter.validation_errors) == 0
    
    def test_validate_simulation_data_missing_fields(self):
        """Test validation with missing required fields."""
        invalid_data = {
            "persona_001_gpt-4": {
                # Missing persona_id, model, responses
                "wrong_field": "data"
            }
        }
        
        formatter = OutputFormatter()
        formatter.simulation_data = invalid_data
        
        result = formatter.validate_simulation_data()
        assert result is False
        assert len(formatter.validation_errors) > 0
    
    def test_validate_simulation_data_missing_conditions(self):
        """Test validation with missing experimental conditions."""
        invalid_data = {
            "persona_001_gpt-4": {
                "persona_id": 1,
                "model": "gpt-4",
                "responses": {
                    "condition_1": {}
                    # Missing condition_2 and condition_3
                }
            }
        }
        
        formatter = OutputFormatter()
        formatter.simulation_data = invalid_data
        
        with patch('config.CONDITIONS', ['condition_1', 'condition_2', 'condition_3']):
            result = formatter.validate_simulation_data()
        
        assert result is False
        assert any("missing conditions" in error for error in formatter.validation_errors)
    
    def test_calculate_summary_statistics(self, sample_simulation_data):
        """Test calculation of summary statistics."""
        formatter = OutputFormatter()
        formatter.simulation_data = sample_simulation_data
        
        stats = formatter.calculate_summary_statistics()
        
        assert stats["total_entries"] == 1
        assert 1 in stats["personas"]
        assert "gpt-4" in stats["models"]
        assert "condition_1" in stats["conditions"]
        assert len(stats["response_completeness"]) > 0
    
    def test_calculate_summary_statistics_empty_data(self):
        """Test statistics calculation with empty data."""
        formatter = OutputFormatter()
        formatter.simulation_data = None
        
        stats = formatter.calculate_summary_statistics()
        
        assert stats == {}
    
    def test_export_to_csv(self, temp_dir, sample_simulation_data):
        """Test CSV export functionality."""
        formatter = OutputFormatter()
        formatter.simulation_data = sample_simulation_data
        
        with patch('config.CONDITIONS', ['condition_1', 'condition_2', 'condition_3']):
            csv_files = formatter.export_to_csv(temp_dir)
        
        assert len(csv_files) == 3  # One for each condition
        
        # Check that files were created
        for csv_file in csv_files:
            assert csv_file.exists()
            assert csv_file.suffix == '.csv'
    
    def test_export_to_csv_empty_data(self, temp_dir):
        """Test CSV export with empty data."""
        formatter = OutputFormatter()
        formatter.simulation_data = None
        
        csv_files = formatter.export_to_csv(temp_dir)
        
        assert csv_files == []
    
    def test_create_analysis_summary(self, temp_dir, sample_simulation_data):
        """Test creation of analysis summary."""
        formatter = OutputFormatter()
        formatter.simulation_data = sample_simulation_data
        formatter.calculate_summary_statistics()
        
        summary_file = formatter.create_analysis_summary(temp_dir / "summary.json")
        
        assert summary_file.exists()
        
        with open(summary_file) as f:
            summary = json.load(f)
        
        assert "simulation_overview" in summary
        assert "data_quality" in summary
        assert "response_statistics" in summary
        assert "recommendations" in summary
    
    def test_generate_recommendations_good_data(self, sample_simulation_data):
        """Test recommendation generation with good data."""
        formatter = OutputFormatter()
        formatter.simulation_data = sample_simulation_data
        formatter.validation_errors = []
        formatter.calculate_summary_statistics()
        
        recommendations = formatter._generate_recommendations()
        
        assert any("good" in rec.lower() for rec in recommendations)
    
    def test_generate_recommendations_validation_errors(self, sample_simulation_data):
        """Test recommendation generation with validation errors."""
        formatter = OutputFormatter()
        formatter.simulation_data = sample_simulation_data
        formatter.validation_errors = ["Test error"]
        formatter.calculate_summary_statistics()
        
        recommendations = formatter._generate_recommendations()
        
        assert any("validation errors" in rec.lower() for rec in recommendations)
    
    def test_format_outputs_complete_workflow(self, temp_dir, sample_simulation_data):
        """Test complete formatting workflow."""
        # Create simulation data file
        simulation_file = temp_dir / "simulation_results.json"
        with open(simulation_file, 'w') as f:
            json.dump(sample_simulation_data, f)
        
        formatter = OutputFormatter()
        
        with patch('config.CONDITIONS', ['condition_1', 'condition_2', 'condition_3']):
            result = formatter.format_outputs(
                input_file=simulation_file,
                export_csv=True,
                create_summary=True
            )
        
        assert result is True
        
        # Check that CSV files were created
        csv_files = list(temp_dir.glob("responses_*.csv"))
        assert len(csv_files) > 0
        
        # Check that analysis summary was created
        summary_files = list(temp_dir.glob("analysis_summary.json"))
        assert len(summary_files) > 0
    
    def test_format_outputs_validation_failure(self, temp_dir):
        """Test formatting workflow with validation failure."""
        # Create invalid simulation data
        invalid_data = {"invalid": "structure"}
        simulation_file = temp_dir / "invalid_simulation.json"
        with open(simulation_file, 'w') as f:
            json.dump(invalid_data, f)
        
        formatter = OutputFormatter()
        result = formatter.format_outputs(simulation_file)
        
        assert result is False
    
    def test_format_outputs_missing_file(self, temp_dir):
        """Test formatting workflow with missing input file."""
        formatter = OutputFormatter()
        result = formatter.format_outputs(temp_dir / "missing.json")
        
        assert result is False