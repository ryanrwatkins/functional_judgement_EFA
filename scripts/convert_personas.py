"""Convert personas dataset from parquet to structured JSON format."""

import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from tqdm import tqdm

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

from config import PERSONAS_JSON, PERSONAS_RAW_FILE, QUESTIONS_PER_PERSONA
from utils.file_io import file_exists_and_not_empty, read_parquet_file, write_json_file


def extract_persona_demographics(persona_text: str) -> Dict[str, str]:
    """
    Extract demographic information from persona text.
    
    Args:
        persona_text: Raw persona description text
        
    Returns:
        Dictionary containing parsed demographic information
    """
    import re
    
    demographics = {}
    
    # Common patterns for demographic parsing
    patterns = {
        "age": [r"age[:\s]+(\d+)", r"(\d+)\s*years?\s*old", r"aged?\s*(\d+)"],
        "gender": [r"gender[:\s]+(male|female|non-binary|other)", r"(male|female|non-binary|other)"],
        "education": [r"education[:\s]+([^,\n]+)", r"degree[:\s]+([^,\n]+)"],
        "occupation": [r"occupation[:\s]+([^,\n]+)", r"job[:\s]+([^,\n]+)", r"works?\s+as\s+([^,\n]+)"],
        "location": [r"location[:\s]+([^,\n]+)", r"lives?\s+in\s+([^,\n]+)", r"from\s+([^,\n]+)"]
    }
    
    # Apply patterns to extract demographic information
    text_lower = persona_text.lower()
    
    for field, field_patterns in patterns.items():
        for pattern in field_patterns:
            match = re.search(pattern, text_lower, re.IGNORECASE)
            if match:
                demographics[field] = match.group(1).strip()
                break
    
    # Set defaults for missing fields
    defaults = {
        "age": "unknown", "gender": "unknown", "education": "unknown",
        "occupation": "unknown", "location": "unknown"
    }
    
    for field, default_value in defaults.items():
        if field not in demographics:
            demographics[field] = default_value
    
    return demographics


def group_persona_responses(df: pd.DataFrame) -> Dict[str, List[Dict]]:
    """
    Group responses by persona ID.
    
    Args:
        df: DataFrame containing persona data
        
    Returns:
        Dictionary mapping persona names to their response lists
    """
    persona_groups = {}
    
    # Group by persona column (column B)
    for persona_name, group in df.groupby('persona'):
        responses = []
        
        # Sort by index to maintain consistent ordering
        group_sorted = group.sort_index()
        
        for idx, row in group_sorted.iterrows():
            response_data = {
                "question_id": idx + 1,  # 1-based question ID
                "instruction": str(row.get('instruction', '')).strip(),
                "original_response": str(row.get('original', '')).strip(),
                "revised_response": str(row.get('data', '')).strip(),
                "critique": str(row.get('critique', '')).strip(),
                "data_type": str(row.get('type', 'unknown')).strip()
            }
            responses.append(response_data)
        
        persona_groups[persona_name] = responses
    
    return persona_groups


def convert_personas_to_json(
    input_file: Path = PERSONAS_RAW_FILE,
    output_file: Path = PERSONAS_JSON,
    skip_if_exists: bool = True
) -> bool:
    """
    Convert personas dataset from parquet to JSON format.
    
    Args:
        input_file: Path to input parquet file
        output_file: Path to output JSON file
        skip_if_exists: Whether to skip if output file already exists
        
    Returns:
        True if conversion successful
    """
    # Check if output already exists
    if skip_if_exists and file_exists_and_not_empty(output_file):
        print(f"Output file already exists, skipping: {output_file}")
        return True
    
    try:
        # Read input data
        print(f"Reading persona data from: {input_file}")
        df = read_parquet_file(input_file)
        
        # Log basic info about the dataset
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Expected columns: data, persona, instruction, original, critique, type
        expected_columns = ['data', 'persona', 'instruction', 'original', 'critique', 'type']
        missing_columns = [col for col in expected_columns if col not in df.columns]
        if missing_columns:
            print(f"ERROR: Missing expected columns: {missing_columns}")
            return False
        
        # Group responses by persona
        print("Grouping responses by persona...")
        persona_groups = group_persona_responses(df)
        
        # Build the final JSON structure
        print("Building final JSON structure...")
        personas_json = {"personas": []}
        
        for persona_id, (persona_name, responses) in enumerate(
            tqdm(persona_groups.items(), desc="Processing personas")
        ):
            # Extract demographics from the persona name/description
            demographics = extract_persona_demographics(persona_name)
            
            persona_data = {
                "id": persona_id + 1,
                "name": persona_name,
                "demographics": demographics,
                "responses": responses
            }
            
            personas_json["personas"].append(persona_data)
        
        # Write to JSON file
        print(f"Writing personas to JSON file: {output_file}")
        write_json_file(personas_json, output_file)
        
        # Final validation
        total_personas = len(personas_json["personas"])
        total_responses = sum(len(p["responses"]) for p in personas_json["personas"])
        
        print(f"Successfully converted {total_personas} personas with {total_responses} total responses")
        print(f"Output file: {output_file}")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Error converting personas: {str(e)}")
        return False


if __name__ == "__main__":
    print("Starting persona dataset conversion...")
    success = convert_personas_to_json()
    
    if success:
        print("Persona conversion completed successfully!")
        sys.exit(0)
    else:
        print("Persona conversion failed!")
        sys.exit(1)