# Functional Judgement EFA Project

This project generates synthetic personas and simulates their responses to psychological instruments using large language models (LLMs). The goal is to create a comprehensive dataset for Exploratory Factor Analysis (EFA) research.

## Project Overview

The project implements a complete pipeline to:

1. **Convert raw datasets** to structured JSON format
2. **Generate prompts** for synthetic personas using psychological instruments
3. **Simulate LLM responses** across multiple models and experimental conditions
4. **Format and validate outputs** for statistical analysis

## Features

- **Multi-model support**: OpenAI GPT-4, Anthropic Claude, and local Llama models
- **Experimental conditions**: Three different context management strategies
- **Comprehensive validation**: Data quality checks and error handling
- **Flexible output formats**: JSON for processing, CSV for analysis
- **Type-safe implementation**: Full type hints and validation
- **Extensive testing**: Comprehensive test suite with pytest

## Project Structure

```
functional_judgement_EFA/
├── data/                     # Input datasets (gitignored)
│   ├── personas_combined.parquet
│   └── instruments.csv
├── outputs/                  # Generated outputs (gitignored)
│   ├── personas.json
│   ├── instruments.json
│   ├── persona_responses.json
│   ├── responses_condition1.csv
│   ├── responses_condition2.csv
│   └── responses_condition3.csv
├── prompts/                  # Jinja2 templates
│   └── system_prompt_template.jinja
├── scripts/                  # Main processing scripts
│   ├── convert_personas.py
│   ├── convert_instruments.py
│   ├── generate_prompts.py
│   ├── run_llm_simulation.py
│   └── format_outputs.py
├── utils/                    # Utility modules
│   ├── file_io.py
│   ├── prompt_utils.py
│   ├── logging_utils.py
│   └── llm_clients.py
├── tests/                    # Test suite
├── config.py                 # Configuration settings
├── Makefile                  # Automation tasks
└── README.md
```

## Quick Start

### Prerequisites

- Python 3.9+
- API keys for LLM services (optional for local models)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/ryanrwatkins/functional_judgement_EFA.git
cd functional_judgement_EFA
```

2. Install dependencies:
```bash
make install
# or
pip install -r requirements.txt
```

3. Set up environment variables (for API access):
```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
```

### Basic Usage

#### 1. Convert Raw Datasets

Convert the raw persona dataset to structured JSON:
```bash
make convert-personas
# or
python scripts/convert_personas.py
```

Convert the psychological instruments dataset:
```bash
make convert-instruments  
# or
python scripts/convert_instruments.py
```

#### 2. Generate Prompts

Create prompts for personas using psychological instruments:
```bash
make generate-prompts
# or
python scripts/generate_prompts.py
```

#### 3. Run LLM Simulation

Simulate persona responses across different models and conditions:
```bash
make run-simulation
# or  
python scripts/run_llm_simulation.py
```

#### 4. Format Outputs

Format simulation results for analysis:
```bash
make format-outputs
# or
python scripts/format_outputs.py
```

### Complete Pipeline

Run the entire data processing pipeline:
```bash
make data-pipeline
```

## Scripts

### convert_personas.py

Converts the raw persona dataset from Parquet to structured JSON format.

- **Input**: `data/personas_combined.parquet`
- **Output**: `outputs/personas.json`
- **Features**: Demographic parsing, response grouping, data validation

### convert_instruments.py

Converts psychological instruments from CSV to structured JSON format.

- **Input**: `data/instruments.csv`  
- **Output**: `outputs/instruments.json`
- **Features**: Scale/subscale grouping, response scale extraction

### generate_prompts.py

Generates prompts for personas using Jinja2 templates.

- **Dependencies**: `personas.json`, `instruments.json`
- **Features**: Template rendering, context preparation, validation

### run_llm_simulation.py

Runs LLM simulations across multiple models and experimental conditions.

- **Models**: OpenAI GPT-4, Anthropic Claude, local Llama
- **Conditions**: 
  - Condition 1: Clear context after each question
  - Condition 2: Clear context after each scale
  - Condition 3: Maintain full context with randomized scale order
- **Output**: `outputs/persona_responses.json`

### format_outputs.py

Formats and validates simulation outputs for analysis.

- **Input**: `outputs/persona_responses.json`
- **Outputs**: CSV files per condition, analysis summary
- **Features**: Data validation, statistical summaries, quality checks

## Configuration

The `config.py` file contains all project settings:

- File paths and directory structure
- Model configurations and API settings
- Experimental conditions
- Validation parameters
- Logging configuration

## Testing

Run the comprehensive test suite:

```bash
make test        # Full test suite with coverage
make test-fast   # Quick tests without coverage
```

Test specific modules:
```bash
pytest tests/test_convert_personas.py -v
pytest tests/test_utils.py -v
```

## Code Quality

The project includes comprehensive code quality tools:

```bash
make format      # Format code with Black
make lint        # Lint with Ruff  
make type-check  # Type check with MyPy
make all         # Run all quality checks
```

## Development

### Setting up Development Environment

```bash
make setup       # Install dependencies and pre-commit hooks
```

### Pre-commit Hooks

The project uses pre-commit hooks for code quality:
- Black (formatting)
- Ruff (linting)
- MyPy (type checking)
- Various file checks

### Adding New Features

1. Create feature branch
2. Implement functionality with type hints
3. Add comprehensive tests
4. Update documentation
5. Run quality checks: `make all`
6. Submit pull request

## Experimental Conditions

The simulation implements three experimental conditions to study context effects:

1. **Condition 1 - Question-level**: Context cleared after each question
2. **Condition 2 - Scale-level**: Context cleared after each psychological scale  
3. **Condition 3 - Full context**: Complete context maintained with randomized scale ordering

## Output Format

### JSON Structure

```json
{
  "persona_001_gpt-4": {
    "persona_id": 1,
    "model": "gpt-4", 
    "responses": {
      "condition_1": {
        "Big Five": {
          "Openness": {
            "Q1": 4,
            "Q2": 3
          }
        }
      }
    }
  }
}
```

### CSV Export

Each experimental condition exports to a separate CSV file with columns:
- `persona_id`: Unique persona identifier
- `model`: LLM model used
- `condition`: Experimental condition
- `{Scale}_{Subscale}_{QuestionID}`: Response values

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Run quality checks: `make all`
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this project in your research, please cite:

```bibtex
@software{functional_judgement_efa,
  title = {Functional Judgement EFA: Synthetic Persona Response Generation},
  author = {Ryan Watkins},
  year = {2024},
  url = {https://github.com/ryanrwatkins/functional_judgement_EFA}
}
```

## Support

For questions, issues, or contributions:
- Open an issue on GitHub
- Contact: ryanrwatkins@gmail.com

## Acknowledgments

- Built for EFA research in psychological assessment
- Utilizes OpenAI, Anthropic, and local LLM technologies
- Implements best practices for reproducible research