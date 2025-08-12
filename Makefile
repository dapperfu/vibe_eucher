.PHONY: help venv install test run clean

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

venv: ## Create virtual environment
	python3 -m venv venv

install: venv ## Install dependencies
	venv/bin/pip install --upgrade pip
	venv/bin/pip install -r requirements.txt

test: install ## Run tests
	venv/bin/pytest tests/ -v

run: install ## Run the euchre game
	venv/bin/python -m euchre.cli

clean: ## Clean up generated files
	rm -rf venv/
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete 