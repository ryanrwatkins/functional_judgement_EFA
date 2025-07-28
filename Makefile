# Makefile for EFA project automation

.PHONY: help install test lint format clean all

# Default target
help:
	@echo "Available targets:"
	@echo "  help        - Show this help message"
	@echo "  install     - Install project dependencies"
	@echo "  test        - Run all tests"
	@echo "  lint        - Run linting with ruff"
	@echo "  format      - Format code with black"
	@echo "  clean       - Clean generated files"
	@echo "  all         - Run format, lint, and test"

# Development setup
install:
	pip install -r requirements.txt

# Code quality
format:
	@echo "Formatting code with black..."
	black .
	@echo "Done."

lint:
	@echo "Linting code with ruff..."
	ruff check .
	@echo "Done."

# Testing
test:
	@echo "Running tests with pytest..."
	pytest -v
	@echo "Done."

# Cleanup
clean:
	@echo "Cleaning generated files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	rm -rf htmlcov/
	rm -rf .coverage
	@echo "Done."

# Run all quality checks
all: format lint test
	@echo "All quality checks completed."

# Data processing targets
convert-personas:
	python scripts/convert_personas.py

convert-instruments:
	python scripts/convert_instruments.py