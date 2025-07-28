# functional_judgement_EFA
# Scripts

## Scripts

### convert_personas.py

Convert the persona dataset from Parquet to JSON format:
```bash
python -m scripts.convert_personas
```
### convert_instruments.py

Convert the instruments CSV dataset to JSON format:
```bash
python -m scripts.convert_instruments

### generate_prompts.py

Render system prompts for each persona (requires personas.json and instruments.json):
```bash
python -m scripts.generate_prompts

### run_llm_simulation.py

Simulate LLM responses across three experimental conditions:
```bash
python -m scripts.run_llm_simulation

### format_outputs.py

Pretty-print the simulation JSON:
```bash
python -m scripts.format_outputs
```
```
```
```
