import json
import os
import random

def get_llm_response(prompt):
    """Gets a response from the LLM."""
    print(f"Getting response for prompt: {prompt[:50]}...")
    return "4" # Placeholder response

def run_simulation(prompts_path, output_path):
    """Runs the LLM simulation."""
    with open(prompts_path, 'r') as f:
        prompts = json.load(f)

    results = {}

    # Condition 1
    for i, prompt in enumerate(prompts['condition_1']):
        persona_id = f"persona_{i+1:03d}"
        if persona_id not in results:
            results[persona_id] = {
                "persona_id": i + 1,
                "model": "gpt-3.5-turbo",
                "responses": {
                    "condition_1": {},
                    "condition_2": {},
                    "condition_3": {}
                }
            }
        response = get_llm_response(prompt)
        results[persona_id]["responses"]["condition_1"][f"Q{i+1}"] = response

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == '__main__':
    run_simulation('outputs/prompts.json', 'outputs/persona_responses.json')