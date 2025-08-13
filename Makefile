.PHONY: help venv install install-pip test run clean train-ai evaluate-ai ai-game ai-game-ncurses ncurses logged profiles mass-games analyze cleanup jupyter jupyter-lab train-self-play train-integer-vs-float generate-profiles generate-profiles-cpu generate-profiles-gpu list-players play-trained tournament benchmark neural-tournament neural-analysis list-neural-models install-gpu train-m-series train-m-series-fast train-m-series-intensive evaluate-m-series train-m-series-vs-traditional m-series-tournament m-series-analysis list-m-series-models

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

venv: ## Create virtual environment
	python3 -m venv venv

install: venv ## Install dependencies (traditional method)
	venv/bin/pip install --upgrade pip
	venv/bin/pip install -r requirements.txt

install-pip: venv ## Install package with pip (editable mode)
	venv/bin/pip install --upgrade pip
	venv/bin/pip install -e .

test: install ## Run tests
	venv/bin/pytest tests/ -v

run: install ## Run the euchre game
	venv/bin/python -m euchre.cli_main play

ai-game: install ## Run AI vs AI euchre game with full verbosity
	venv/bin/python -m euchre.cli_main --very-verbose ai-vs-ai

ai-game-ncurses: install ## Run AI vs AI euchre game with ncurses interface
	venv/bin/python -m euchre.cli_main ai-vs-ai

ncurses: install ## Run AI vs AI euchre game with ncurses interface
	venv/bin/python -m euchre.cli_main ncurses

logged: install ## Run AI vs AI euchre game with logging (no ncurses)
	venv/bin/python -m euchre.cli_main logged-game

profiles: install ## Run AI vs AI game with custom profiles and risk ratios
	venv/bin/python -m euchre.cli_main ai-profiles

mass-games: install ## Run thousands of games in parallel
	venv/bin/python -m euchre.cli_main run-mass-games

analyze: install ## Analyze game results and generate statistics
	venv/bin/python -m euchre.cli_main analyze-games

cleanup: install ## Clean up old game result files
	venv/bin/python -m euchre.cli_main cleanup-games

jupyter: install ## Start Jupyter Notebook
	venv/bin/python -m euchre.cli_main jupyter

jupyter-lab: install ## Start Jupyter Lab
	venv/bin/python -m euchre.cli_main jupyter-lab

clean: ## Clean up generated files
	rm -rf venv/
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete 

train-ai: install ## Train the euchre AI model
	venv/bin/python train_euchre_ai.py --data-dir data/games --generate-data --evaluate

evaluate-ai: install ## Evaluate a trained AI model
	venv/bin/python -m euchre.ai_model.model_evaluator --model models/euchre_model.pth --games 100

train-self-play: install ## Train AI models using self-play
	venv/bin/python -m euchre.cli_main train-self-play --num-games 100000 --players Alice Bob Charlie David

train-integer-vs-float: install ## Train integer models against float models
	venv/bin/python -m euchre.cli_main train-integer-vs-float --num-games 200000

generate-profiles: install ## Generate 5 distinct AI player profiles with different playing styles
	venv/bin/python -m euchre.cli_main generate-player-profiles --num-games 15000 --output-dir trained_models --device auto --enable-amp

generate-profiles-cpu: install ## Generate AI player profiles using CPU only
	venv/bin/python -m euchre.cli_main generate-player-profiles --num-games 15000 --output-dir trained_models --device cpu

generate-profiles-gpu: install ## Generate AI player profiles with GPU acceleration
	venv/bin/python -m euchre.cli_main generate-player-profiles --num-games 15000 --output-dir trained_models --device cuda --enable-amp --gpu-memory-fraction 0.9

list-players: install ## List available trained AI players
	venv/bin/python -m euchre.cli_main list-players

play-trained: install ## Play a game with trained AI players
	venv/bin/python -m euchre.cli_main play-trained-players --player1 Alice --player2 Bob --player3 Charlie --player4 David

tournament: install ## Run a tournament between trained players
	venv/bin/python -m euchre.cli_main tournament --num-games 100

benchmark: install ## Run performance benchmark comparing float vs integer models
	venv/bin/python -m euchre.cli_main benchmark --device cpu --save-results

neural-tournament: install ## Run neural network tournament (Alice vs Bob, 1000 games)
	venv/bin/python -m euchre.cli_main run-neural-games -m1 Alice -m2 Bob -n 1000

neural-analysis: install ## Analyze neural network tournament results
	venv/bin/python analyze_neural_results.py

list-neural-models: install ## List available neural network models
	venv/bin/python -m euchre.cli_main list-neural-models

# =============================================================================
# M-SERIES AI TRAINING TARGETS
# =============================================================================

install-gpu: install ## Install GPU dependencies for M-Series training
	@echo "Installing GPU dependencies..."
	@echo "Installing PyTorch with automatic backend detection..."
	venv/bin/pip install torch torchvision torchaudio
	@echo "GPU dependencies installed. Checking availability..."
	venv/bin/python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA/ROCm available: {torch.cuda.is_available()}'); print(f'Device count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}')"

train-m-series: install-gpu ## Train M-Series AI models using unified trainer (auto-detects best backend)
	@echo "Starting M-Series AI training with unified trainer..."
	@echo "This will automatically detect and use the best available backend (GPU/CPU)"
	@echo "Training 20000 games over 1000 epochs..."
	venv/bin/python unified_trainer.py \
		--epochs 1000 \
		--total-games 20000 \
		--batch-size 64 \
		--learning-rate 0.001 \
		--device auto \
		--max-gpus 2

train-m-series-fast: install-gpu ## Quick M-Series training for smoke testing (<1000 games, few epochs)
	@echo "Starting fast M-Series AI training for smoke testing..."
	@echo "Training 500 games over 10 epochs to verify the process works..."
	venv/bin/python unified_trainer.py \
		--epochs 10 \
		--total-games 500 \
		--batch-size 32 \
		--learning-rate 0.001 \
		--device auto \
		--max-gpus 2

train-m-series-intensive: install-gpu ## Intensive M-Series training for production models
	@echo "Starting intensive M-Series AI training for production..."
	@echo "Training 50000 games over 2000 epochs with optimized parameters..."
	venv/bin/python unified_trainer.py \
		--epochs 2000 \
		--total-games 50000 \
		--batch-size 128 \
		--learning-rate 0.0005 \
		--device auto \
		--max-gpus 2

evaluate-m-series: install-gpu ## Evaluate trained M-Series models
	@echo "Evaluating trained M-Series models..."
	venv/bin/python -c " \
import json; \
import os; \
model_dir = 'trained_models/unified_trained'; \
if os.path.exists(model_dir): \
    models = [f for f in os.listdir(model_dir) if f.endswith('.json')]; \
    print(f'Found {len(models)} M-Series models:'); \
    for model in models: \
        print(f'  - {model}'); \
else: \
    print('No trained M-Series models found. Run train-m-series first.') \
"

# =============================================================================
# M-SERIES VS TRADITIONAL AI COMPETITION
# =============================================================================

train-m-series-vs-traditional: install-gpu ## Train M-Series models specifically to compete against traditional AI
	@echo "Training M-Series models to compete against traditional AI..."
	@echo "This will focus on strategies that outperform traditional rule-based AI..."
	venv/bin/python unified_trainer.py \
		--epochs 1500 \
		--total-games 30000 \
		--batch-size 64 \
		--learning-rate 0.001 \
		--device auto \
		--max-gpus 2

m-series-tournament: install-gpu ## Run tournament: M-Series AI vs Traditional AI
	@echo "🏆 M-Series AI vs Traditional AI Tournament"
	@echo "Running tournament with trained M-Series models against traditional AI..."
	venv/bin/python ai_tournament.py \
		--m-series-models trained_models/unified_trained \
		--traditional-ai-types balanced aggressive conservative opportunistic \
		--games-per-match 100 \
		--output-dir tournament_results/m_series_vs_traditional

m-series-analysis: install-gpu ## Analyze M-Series vs Traditional AI tournament results
	@echo "📊 Analyzing M-Series vs Traditional AI tournament results..."
	@if [ -d "tournament_results/m_series_vs_traditional" ]; then \
		venv/bin/python -c " \
import json; \
import os; \
import glob; \
results_dir = 'tournament_results/m_series_vs_traditional'; \
results_files = glob.glob(os.path.join(results_dir, '*.json')); \
if results_files: \
    print(f'Found {len(results_files)} tournament result files:'); \
    for f in results_files: \
        print(f'  - {os.path.basename(f)}'); \
    print('\\nRun: venv/bin/python analyze_neural_results.py for detailed analysis'); \
else: \
    print('No tournament results found. Run m-series-tournament first.'); \
"; \
	else \
		echo "No tournament results directory found. Run m-series-tournament first."; \
	fi

list-m-series-models: install-gpu ## List available M-Series models
	@echo "Available M-Series AI models:"
	@if [ -d "trained_models/unified_trained" ]; then \
		ls -la trained_models/unified_trained/*.json 2>/dev/null | sed 's/.*\//  - /' || echo "  No trained models found"; \
	else \
		echo "  No trained models directory found"; \
	fi

# =============================================================================
# HUMAN VS AI GAMEPLAY
# =============================================================================

human-vs-ai: install ## Play as human vs AI with configurable AI types
	@echo "🎮 Human vs AI Euchre Game"
	@echo "Usage examples:"
	@echo "  make human-vs-ai-default          # Play with default settings"
	@echo "  make human-vs-ai-aggressive      # Play with aggressive partner"
	@echo "  make human-vs-ai-conservative    # Play with conservative partner"
	@echo "  make human-vs-ai-balanced        # Play with balanced partner"
	@echo "  make human-vs-ai-opportunistic   # Play with opportunistic partner"
	@echo "  make human-vs-ai-custom          # Play with custom AI configuration"

human-vs-ai-default: install ## Play as human with balanced AI partner vs balanced opponents
	venv/bin/python -m euchre.cli_main human-vs-ai --player-name "Player" --your-position 0 --partner-ai-type balanced --opponent1-ai-type balanced --opponent2-ai-type balanced

human-vs-ai-aggressive: install ## Play as human with aggressive AI partner
	venv/bin/python -m euchre.cli_main human-vs-ai --player-name "Player" --your-position 0 --partner-ai-type aggressive --opponent1-ai-type balanced --opponent2-ai-type balanced

human-vs-ai-conservative: install ## Play as human with conservative AI partner
	venv/bin/python -m euchre.cli_main human-vs-ai --player-name "Player" --your-position 0 --partner-ai-type conservative --opponent1-ai-type balanced --opponent2-ai-type balanced

human-vs-ai-balanced: install ## Play as human with balanced AI partner
	venv/bin/python -m euchre.cli_main human-vs-ai --player-name "Player" --your-position 0 --partner-ai-type balanced --opponent1-ai-type balanced --opponent2-ai-type balanced

human-vs-ai-opportunistic: install ## Play as human with opportunistic AI partner
	venv/bin/python -m euchre.cli_main human-vs-ai --player-name "Player" --your-position 0 --partner-ai-type opportunistic --opponent1-ai-type balanced --opponent2-ai-type balanced

human-vs-ai-custom: install ## Play as human with custom AI configuration
	@echo "🎮 Custom Human vs AI Configuration"
	@echo "Available AI types: aggressive, conservative, balanced, opportunistic"
	@echo "Risk ratios: 0.0 (very conservative) to 1.0 (very aggressive)"
	@echo ""
	@read -p "Enter your name: " name; \
	read -p "Enter your position (0=Alice, 1=Bob, 2=Charlie, 3=David): " pos; \
	read -p "Enter partner AI type: " partner; \
	read -p "Enter opponent1 AI type: " opp1; \
	read -p "Enter opponent2 AI type: " opp2; \
	read -p "Enter partner risk ratio (0.0-1.0): " prisk; \
	read -p "Enter opponent1 risk ratio (0.0-1.0): " orisk1; \
	read -p "Enter opponent2 risk ratio (0.0-1.0): " orisk2; \
	venv/bin/python -m euchre.cli_main human-vs-ai --player-name "$$name" --your-position $$pos --partner-ai-type $$partner --opponent1-ai-type $$opp1 --opponent2-ai-type $$opp2 --partner-risk $$prisk --opponent1-risk $$orisk1 --opponent2-risk $$orisk2 