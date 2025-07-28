import json
from config import RESPONSES_OUTPUT_PATH

def format_outputs(input_path, output_path):
    """Formats the LLM simulation outputs into the desired JSON structure."""
    with open(input_path, 'r') as f:
        raw_responses = json.load(f)

    formatted_output = {}

    for persona_id, data in raw_responses.items():
        formatted_output[persona_id] = {
            "persona_id": data["persona_id"],
            "model": data["model"], # This will be overwritten per condition
            "responses": {
                "condition_1": {},
                "condition_2": {},
                "condition_3": {}
            }
        }

        for condition, models_data in data["responses"].items():
            for model_name, responses_data in models_data.items():
                # For condition 1, responses_data is already structured by scale/subscale/question
                if condition == "condition_1":
                    formatted_output[persona_id]["responses"][condition][model_name] = responses_data
                # For conditions 2 and 3, responses_data is a single string that needs parsing
                else:
                    # Placeholder for parsing logic. This will depend on the actual LLM output format.
                    # For now, we'll just store the raw string.
                    formatted_output[persona_id]["responses"][condition][model_name] = {"raw_response": responses_data}

    with open(output_path, 'w') as f:
        json.dump(formatted_output, f, indent=2)

if __name__ == '__main__':
    format_outputs(RESPONSES_OUTPUT_PATH, RESPONSES_OUTPUT_PATH) # Overwrite the same file for now
