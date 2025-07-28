import pandas as pd
import json

def convert_personas_to_json(input_path, output_path):
    """Converts the persona dataset from Parquet to JSON."""
    df = pd.read_parquet(input_path)
    personas = []
    # Assign a unique integer ID to each persona based on the 'cut' column
    unique_cuts = df['cut'].unique()
    cut_to_id = {cut: i + 1 for i, cut in enumerate(unique_cuts)}

    for cut_value in unique_cuts:
        group = df[df['cut'] == cut_value]
        persona_id = cut_to_id[cut_value]
        persona_data = {
            'id': persona_id,
            'demographics': {},
            'responses': []
        }
        # Assuming demographics are the same for all rows of a persona
        # and are in a parsable format in the 'critique' column.
        # This is a placeholder for the actual parsing logic.
        # You will need to implement the logic to extract demographics.
        persona_data['demographics'] = {'age': 30, 'gender': 'female'} # Example

        for _, row in group.iterrows():
            persona_data['responses'].append({
                'question_id': row['color'],
                'original_response': row['clarity'],
                'revised_response': row['depth']
            })
        personas.append(persona_data)

    with open(output_path, 'w') as f:
        json.dump({'personas': personas}, f, indent=2)

if __name__ == '__main__':
    convert_personas_to_json('data/personas_combined.parquet', 'outputs/personas.json')
