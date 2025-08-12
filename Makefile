.PHONY: help venv install test run clean ai-game ncurses logged profiles mass-games analyze cleanup

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
	venv/bin/python -m euchre.cli play

ai-game: install ## Run AI vs AI euchre game with ncurses
	venv/bin/python -m euchre.cli ai-vs-ai

ncurses: install ## Run AI vs AI euchre game with ncurses interface
	venv/bin/python -m euchre.cli ncurses

logged: install ## Run AI vs AI euchre game with logging (no ncurses)
	venv/bin/python -m euchre.cli logged-game

profiles: install ## Run AI vs AI game with custom profiles and risk ratios
	venv/bin/python -m euchre.cli ai-profiles

mass-games: install ## Run thousands of games in parallel
	venv/bin/python -m euchre.cli run-mass-games

analyze: install ## Analyze game results and generate statistics
	venv/bin/python -m euchre.cli analyze-games

cleanup: install ## Clean up old game result files
	venv/bin/python -m euchre.cli cleanup-games

clean: ## Clean up generated files
	rm -rf venv/
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete 