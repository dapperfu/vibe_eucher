.PHONY: help venv install test run clean train-ai evaluate-ai ai-game ncurses logged profiles mass-games analyze cleanup jupyter jupyter-lab train-self-play list-players play-trained tournament benchmark

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

jupyter: install ## Start Jupyter Notebook
	cd notebooks && ../venv/bin/jupyter notebook

jupyter-lab: install ## Start Jupyter Lab
	cd notebooks && ../venv/bin/jupyter lab

clean: ## Clean up generated files
	rm -rf venv/
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete

train-ai: install ## Train the euchre AI model
	venv/bin/python train_euchre_ai.py --data-dir data/games --generate-data --evaluate

evaluate-ai: install ## Evaluate a trained AI model
	venv/bin/python -m euchre.ai_model.model_evaluator --model models/euchre_model.pth --games 100

train-self-play: install ## Train AI models using self-play
	venv/bin/python -m euchre.cli train-self-play --num-games 100000 --players Alice Bob Charlie David

list-players: install ## List available trained AI players
	venv/bin/python -m euchre.cli list-players

play-trained: install ## Play a game with trained AI players
	venv/bin/python -m euchre.cli play-trained-players --player1 Alice --player2 Bob --player3 Charlie --player4 David

tournament: install ## Run a tournament between trained players
	venv/bin/python -m euchre.cli tournament --num-games 100

benchmark: install ## Run performance benchmark comparing float vs integer models
	venv/bin/python -m euchre.cli benchmark --device cpu --save-results 