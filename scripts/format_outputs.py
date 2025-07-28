"""Format and validate output JSON files from LLM simulations."""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
from tqdm import tqdm

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

from config import CONDITIONS, OUTPUTS_DIR, PERSONA_RESPONSES_JSON
from utils.file_io import read_json_file, write_json_file
from utils.logging_utils import setup_logging, get_logger

logger = get_logger(__name__)


class OutputFormatter:
    """Handles formatting and validation of simulation output data."""
    
    def __init__(self):
        """Initialize the output formatter."""
        self.simulation_data = None
        self.validation_errors = []
        self.summary_stats = {}
        
        logger.info("Initialized OutputFormatter")
    
    def load_simulation_data(self, file_path: Path = PERSONA_RESPONSES_JSON) -> bool:
        """
        Load simulation data from JSON file.
        
        Args:
            file_path: Path to simulation results JSON file
            
        Returns:
            True if loading successful
        """
        try:
            logger.info(f"Loading simulation data from: {file_path}")
            self.simulation_data = read_json_file(file_path)
            
            logger.info(f"Loaded data for {len(self.simulation_data)} persona-model combinations")
            return True
            
        except Exception as e:
            logger.error(f"Error loading simulation data: {str(e)}")
            return False
    
    def validate_simulation_data(self) -> bool:
        """
        Validate the structure and completeness of simulation data.
        
        Returns:
            True if validation passes
        """
        if not self.simulation_data:
            self.validation_errors.append("No simulation data loaded")
            return False
        
        logger.info("Validating simulation data structure...")
        self.validation_errors = []
        
        expected_fields = ["persona_id", "model", "responses"]
        
        for entry_key, entry_data in self.simulation_data.items():
            # Check required fields
            missing_fields = [field for field in expected_fields if field not in entry_data]
            if missing_fields:
                self.validation_errors.append(f"Entry {entry_key} missing fields: {missing_fields}")
                continue
            
            # Check conditions
            responses = entry_data.get("responses", {})
            missing_conditions = [cond for cond in CONDITIONS if cond not in responses]
            if missing_conditions:
                self.validation_errors.append(f"Entry {entry_key} missing conditions: {missing_conditions}")
            
            # Check response structure for each condition
            for condition, condition_responses in responses.items():
                if not isinstance(condition_responses, dict):
                    self.validation_errors.append(f"Entry {entry_key}, {condition}: responses not a dictionary")
                    continue
                
                # Check that responses contain scales and subscales
                for scale_name, scale_responses in condition_responses.items():
                    if not isinstance(scale_responses, dict):
                        self.validation_errors.append(f"Entry {entry_key}, {condition}, {scale_name}: not a dictionary")
                        continue
                    
                    for subscale_name, subscale_responses in scale_responses.items():
                        if not isinstance(subscale_responses, dict):
                            self.validation_errors.append(f"Entry {entry_key}, {condition}, {scale_name}, {subscale_name}: not a dictionary")
        
        if self.validation_errors:
            logger.error(f"Validation failed with {len(self.validation_errors)} errors:")
            for error in self.validation_errors[:10]:  # Limit error output
                logger.error(f"  - {error}")
            return False
        
        logger.info("Validation passed")
        return True
    
    def calculate_summary_statistics(self) -> Dict:
        """
        Calculate summary statistics about the simulation data.
        
        Returns:
            Dictionary of summary statistics
        """
        if not self.simulation_data:
            return {}
        
        logger.info("Calculating summary statistics...")
        
        stats = {
            "total_entries": len(self.simulation_data),
            "personas": set(),
            "models": set(),
            "conditions": set(),
            "response_completeness": {},
            "missing_responses": {},
            "response_distributions": {}
        }
        
        for entry_data in self.simulation_data.values():
            stats["personas"].add(entry_data.get("persona_id"))
            stats["models"].add(entry_data.get("model"))
            
            responses = entry_data.get("responses", {})
            for condition, condition_responses in responses.items():
                stats["conditions"].add(condition)
                
                # Count complete vs missing responses
                total_responses = 0
                missing_responses = 0
                valid_responses = []
                
                for scale_responses in condition_responses.values():
                    for subscale_responses in scale_responses.values():
                        for response_value in subscale_responses.values():
                            total_responses += 1
                            if response_value is None:
                                missing_responses += 1
                            else:
                                valid_responses.append(response_value)
                
                # Store completeness stats
                condition_key = f"{entry_data.get('model')}_{condition}"
                if condition_key not in stats["response_completeness"]:
                    stats["response_completeness"][condition_key] = []
                    stats["missing_responses"][condition_key] = []
                
                completeness = (total_responses - missing_responses) / total_responses if total_responses > 0 else 0
                stats["response_completeness"][condition_key].append(completeness)
                stats["missing_responses"][condition_key].append(missing_responses)
                
                # Store response values for distribution analysis
                if condition_key not in stats["response_distributions"]:
                    stats["response_distributions"][condition_key] = []
                stats["response_distributions"][condition_key].extend(valid_responses)
        
        # Convert sets to lists and calculate averages
        stats["personas"] = sorted(list(stats["personas"]))
        stats["models"] = sorted(list(stats["models"]))
        stats["conditions"] = sorted(list(stats["conditions"]))
        
        # Calculate average completeness
        for condition_key in stats["response_completeness"]:
            completeness_values = stats["response_completeness"][condition_key]
            stats["response_completeness"][condition_key] = {
                "mean": sum(completeness_values) / len(completeness_values),
                "min": min(completeness_values),
                "max": max(completeness_values),
                "count": len(completeness_values)
            }
        
        # Calculate response distribution stats
        for condition_key in stats["response_distributions"]:
            responses = stats["response_distributions"][condition_key]
            if responses:
                stats["response_distributions"][condition_key] = {
                    "mean": sum(responses) / len(responses),
                    "min": min(responses),
                    "max": max(responses),
                    "count": len(responses)
                }
        
        self.summary_stats = stats
        logger.info(f"Calculated statistics for {stats['total_entries']} entries")
        return stats
    
    def export_to_csv(self, output_dir: Path = OUTPUTS_DIR) -> List[Path]:
        """
        Export simulation data to CSV files for analysis.
        
        Args:
            output_dir: Directory to save CSV files
            
        Returns:
            List of created CSV file paths
        """
        if not self.simulation_data:
            logger.error("No simulation data to export")
            return []
        
        logger.info("Exporting simulation data to CSV files...")
        created_files = []
        
        # Create CSV for each condition
        for condition in CONDITIONS:
            csv_data = []
            
            for entry_key, entry_data in tqdm(self.simulation_data.items(), desc=f"Processing {condition}"):
                persona_id = entry_data.get("persona_id")
                model = entry_data.get("model")
                condition_responses = entry_data.get("responses", {}).get(condition, {})
                
                # Flatten responses for CSV
                row = {
                    "persona_id": persona_id,
                    "model": model,
                    "condition": condition
                }
                
                for scale_name, scale_responses in condition_responses.items():
                    for subscale_name, subscale_responses in scale_responses.items():
                        for question_id, response_value in subscale_responses.items():
                            column_name = f"{scale_name}_{subscale_name}_{question_id}".replace(" ", "_")
                            row[column_name] = response_value
                
                csv_data.append(row)
            
            # Save to CSV
            if csv_data:
                df = pd.DataFrame(csv_data)
                csv_file = output_dir / f"responses_{condition}.csv"
                df.to_csv(csv_file, index=False)
                created_files.append(csv_file)
                
                logger.info(f"Exported {len(csv_data)} rows to {csv_file}")
        
        return created_files
    
    def create_analysis_summary(self, output_file: Path = None) -> Path:
        """
        Create a summary report of the analysis.
        
        Args:
            output_file: Path to save summary (default: outputs/analysis_summary.json)
            
        Returns:
            Path to created summary file
        """
        if output_file is None:
            output_file = OUTPUTS_DIR / "analysis_summary.json"
        
        logger.info("Creating analysis summary...")
        
        summary = {
            "simulation_overview": {
                "total_entries": self.summary_stats.get("total_entries", 0),
                "unique_personas": len(self.summary_stats.get("personas", [])),
                "models_used": self.summary_stats.get("models", []),
                "conditions_tested": self.summary_stats.get("conditions", [])
            },
            "data_quality": {
                "validation_errors": len(self.validation_errors),
                "response_completeness": self.summary_stats.get("response_completeness", {}),
                "missing_responses_summary": self.summary_stats.get("missing_responses", {})
            },
            "response_statistics": self.summary_stats.get("response_distributions", {}),
            "recommendations": self._generate_recommendations()
        }
        
        write_json_file(summary, output_file)
        logger.info(f"Created analysis summary: {output_file}")
        
        return output_file
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on data quality analysis."""
        recommendations = []
        
        if len(self.validation_errors) > 0:
            recommendations.append("Address validation errors before proceeding with analysis")
        
        # Check response completeness
        completeness_stats = self.summary_stats.get("response_completeness", {})
        for condition_key, stats in completeness_stats.items():
            if stats["mean"] < 0.9:
                recommendations.append(f"Low response completeness for {condition_key}: {stats['mean']:.2%}")
        
        # Check response distributions
        distribution_stats = self.summary_stats.get("response_distributions", {})
        for condition_key, stats in distribution_stats.items():
            if stats["min"] == stats["max"]:
                recommendations.append(f"No response variation in {condition_key} - check model outputs")
        
        if not recommendations:
            recommendations.append("Data quality appears good - proceed with statistical analysis")
        
        return recommendations
    
    def format_outputs(
        self,
        input_file: Path = PERSONA_RESPONSES_JSON,
        export_csv: bool = True,
        create_summary: bool = True
    ) -> bool:
        """
        Main function to format and validate outputs.
        
        Args:
            input_file: Path to simulation results JSON
            export_csv: Whether to export CSV files
            create_summary: Whether to create analysis summary
            
        Returns:
            True if formatting successful
        """
        try:
            # Load data
            if not self.load_simulation_data(input_file):
                return False
            
            # Validate data
            if not self.validate_simulation_data():
                logger.error("Data validation failed - check validation errors")
                return False
            
            # Calculate statistics
            self.calculate_summary_statistics()
            
            # Export to CSV if requested
            if export_csv:
                csv_files = self.export_to_csv()
                logger.info(f"Created {len(csv_files)} CSV files")
            
            # Create summary if requested
            if create_summary:
                summary_file = self.create_analysis_summary()
                logger.info(f"Created analysis summary: {summary_file}")
            
            logger.info("Output formatting completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error formatting outputs: {str(e)}", exc_info=True)
            return False


def main() -> None:
    """Main function to run output formatting."""
    setup_logging()
    
    logger.info("Starting output formatting...")
    
    formatter = OutputFormatter()
    success = formatter.format_outputs()
    
    if success:
        logger.info("Output formatting completed successfully!")
        sys.exit(0)
    else:
        logger.error("Output formatting failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()