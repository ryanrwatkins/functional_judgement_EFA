import pandas as pd
import json

def convert_instruments_to_json(input_path, output_path):
    """Converts the instruments dataset from CSV to JSON."""
    df = pd.read_csv(input_path)
    instruments = {}
    for _, row in df.iterrows():
        scale = row['scale']
        subscale = row['subscale']
        if scale not in instruments:
            instruments[scale] = {
                'scale_id': len(instruments) + 1,
                'response_scale': row['response scale'],
            }
        if subscale not in instruments[scale]:
            instruments[scale][subscale] = {}
        instruments[scale][subscale][row['number']] = row['item']

    with open(output_path, 'w') as f:
        json.dump(instruments, f, indent=2)

if __name__ == '__main__':
    convert_instruments_to_json('data/instruments.csv', 'outputs/instruments.json')
