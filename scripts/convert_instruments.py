"""Convert psychological instruments dataset from CSV to structured JSON format."""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

import pandas as pd
from tqdm import tqdm

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

from config import INSTRUMENTS_FILE, INSTRUMENTS_JSON
from utils.file_io import file_exists_and_not_empty, read_csv_file, write_json_file
from utils.logging_utils import setup_logging, get_logger

logger = get_logger(__name__)


def clean_text_field(text: str) -> str:
    """
    Clean and normalize text fields.
    
    Args:
        text: Raw text to clean
        
    Returns:
        Cleaned text
    """
    if pd.isna(text) or text is None:
        return ""
    
    # Convert to string and strip whitespace
    cleaned = str(text).strip()
    
    # Remove extra whitespace and normalize
    cleaned = ' '.join(cleaned.split())
    
    return cleaned


def extract_response_scale_info(scale_text: str) -> Optional[str]:
    """
    Extract response scale information from text.
    
    Args:
        scale_text: Text potentially containing response scale info
        
    Returns:
        Cleaned response scale text or None
    """
    if not scale_text or pd.isna(scale_text):
        return None
    
    scale_text = clean_text_field(scale_text)
    
    # Check if this looks like response scale instructions
    scale_indicators = [
        "likert", "scale", "rate", "respond", "answer", 
        "1 =", "strongly", "agree", "disagree", "never", "always"
    ]
    
    if any(indicator in scale_text.lower() for indicator in scale_indicators):
        return scale_text
    
    return None


def group_instruments_by_scale(df: pd.DataFrame) -> Dict[str, Dict]:
    """
    Group instruments by scale and subscale.
    
    Args:
        df: DataFrame containing instrument data
        
    Returns:
        Dictionary organized by scale and subscale
    """
    instruments_structure = {}
    scale_id_counter = 1
    
    # Group by scale
    for scale_name, scale_group in df.groupby('scale'):
        scale_name = clean_text_field(scale_name)
        
        if not scale_name:
            logger.warning("Found empty scale name, skipping...")
            continue
        
        logger.debug(f"Processing scale: {scale_name}")
        
        # Initialize scale structure
        scale_data = {
            "scale_id": scale_id_counter,
            "response_scale": None,
            "subscales": {}
        }
        
        # Look for scale-level response instructions
        response_scales = scale_group['response scale'].dropna().unique()
        if len(response_scales) > 0:
            # Use the first non-empty response scale
            for rs in response_scales:
                scale_response = extract_response_scale_info(rs)
                if scale_response:
                    scale_data["response_scale"] = scale_response
                    break
        
        # Group by subscale within this scale
        for subscale_name, subscale_group in scale_group.groupby('subscale'):
            subscale_name = clean_text_field(subscale_name)
            
            if not subscale_name:
                subscale_name = "General"  # Default subscale name
            
            logger.debug(f"  Processing subscale: {subscale_name}")
            
            # Initialize subscale structure
            subscale_data = {
                "response_scale": None,
                "questions": {}
            }
            
            # Check for subscale-specific response instructions
            subscale_response_scales = subscale_group['response scale'].dropna().unique()
            if len(subscale_response_scales) > 0:
                for rs in subscale_response_scales:
                    subscale_response = extract_response_scale_info(rs)
                    if subscale_response and subscale_response != scale_data.get("response_scale"):
                        subscale_data["response_scale"] = subscale_response
                        break
            
            # Add questions to subscale
            for _, row in subscale_group.iterrows():
                question_num = clean_text_field(str(row['number']))
                question_text = clean_text_field(row['item'])
                
                if question_num and question_text:
                    subscale_data["questions"][question_num] = question_text
            
            # Only add subscale if it has questions
            if subscale_data["questions"]:
                scale_data["subscales"][subscale_name] = subscale_data
        
        # Only add scale if it has subscales
        if scale_data["subscales"]:
            instruments_structure[scale_name] = scale_data
            scale_id_counter += 1
    
    return instruments_structure


def validate_instruments_data(instruments_data: Dict) -> bool:
    """
    Validate the extracted instruments data.
    
    Args:
        instruments_data: Dictionary of instrument data
        
    Returns:
        True if validation passes
    """
    validation_errors = []
    total_questions = 0
    
    for scale_name, scale_data in instruments_data.items():
        # Check scale structure
        if "scale_id" not in scale_data:
            validation_errors.append(f"Scale {scale_name} missing scale_id")
        
        if "subscales" not in scale_data or not scale_data["subscales"]:
            validation_errors.append(f"Scale {scale_name} has no subscales")
            continue
        
        # Check subscales
        for subscale_name, subscale_data in scale_data["subscales"].items():
            if "questions" not in subscale_data or not subscale_data["questions"]:
                validation_errors.append(
                    f"Subscale {subscale_name} in scale {scale_name} has no questions"
                )
            else:
                total_questions += len(subscale_data["questions"])
    
    if validation_errors:
        logger.error(f"Validation failed with {len(validation_errors)} errors:")
        for error in validation_errors:
            logger.error(f"  - {error}")
        return False
    
    logger.info(f"Validation passed for {len(instruments_data)} scales with {total_questions} total questions")
    return True


def convert_instruments_to_json(
    input_file: Path = INSTRUMENTS_FILE,
    output_file: Path = INSTRUMENTS_JSON,
    skip_if_exists: bool = True
) -> bool:
    """
    Convert instruments dataset from CSV to JSON format.
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output JSON file
        skip_if_exists: Whether to skip if output file already exists
        
    Returns:
        True if conversion successful
    """
    # Check if output already exists
    if skip_if_exists and file_exists_and_not_empty(output_file):
        logger.info(f"Output file already exists, skipping: {output_file}")
        return True
    
    try:
        # Read input data
        logger.info(f"Reading instruments data from: {input_file}")
        df = read_csv_file(input_file)
        
        # Log basic info about the dataset
        logger.info(f"Dataset shape: {df.shape}")
        logger.info(f"Columns: {list(df.columns)}")
        
        # Expected columns: number, item, subscale, scale, response scale
        expected_columns = ['number', 'item', 'subscale', 'scale', 'response scale']
        missing_columns = [col for col in expected_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Missing expected columns: {missing_columns}")
            return False
        
        # Clean the data
        logger.info("Cleaning instrument data...")
        df['number'] = df['number'].astype(str)
        df['item'] = df['item'].astype(str)
        df['subscale'] = df['subscale'].astype(str)
        df['scale'] = df['scale'].astype(str)
        
        # Remove rows with missing essential data
        initial_rows = len(df)
        df = df.dropna(subset=['number', 'item', 'scale'])
        final_rows = len(df)
        
        if initial_rows != final_rows:
            logger.info(f"Removed {initial_rows - final_rows} rows with missing essential data")
        
        # Group instruments by scale and subscale
        logger.info("Grouping instruments by scale and subscale...")
        instruments_data = group_instruments_by_scale(df)
        
        # Validate the grouped data
        logger.info("Validating instruments data...")
        if not validate_instruments_data(instruments_data):
            return False
        
        # Write to JSON file
        logger.info(f"Writing instruments to JSON file: {output_file}")
        write_json_file(instruments_data, output_file)
        
        # Final summary
        total_scales = len(instruments_data)
        total_subscales = sum(len(scale["subscales"]) for scale in instruments_data.values())
        total_questions = sum(
            len(subscale["questions"]) 
            for scale in instruments_data.values() 
            for subscale in scale["subscales"].values()
        )
        
        logger.info(f"Successfully converted {total_scales} scales, {total_subscales} subscales, {total_questions} questions")
        logger.info(f"Output file: {output_file} ({output_file.stat().st_size / 1024:.2f} KB)")
        
        return True
        
    except Exception as e:
        logger.error(f"Error converting instruments: {str(e)}", exc_info=True)
        return False


def main() -> None:
    """Main function to run instruments conversion."""
    setup_logging()
    
    logger.info("Starting instruments dataset conversion...")
    
    success = convert_instruments_to_json()
    
    if success:
        logger.info("Instruments conversion completed successfully!")
        sys.exit(0)
    else:
        logger.error("Instruments conversion failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()