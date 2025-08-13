.PHONY: help venv install install-pip test run clean train-ai evaluate-ai ai-game ai-game-ncurses ncurses logged profiles mass-games analyze cleanup jupyter jupyter-lab train-self-play train-integer-vs-float generate-profiles generate-profiles-cpu generate-profiles-gpu list-players play-trained tournament benchmark neural-tournament neural-analysis list-neural-models install-gpu train-m-series train-m-series-fast train-m-series-intensive evaluate-m-series train-m-series-vs-traditional m-series-tournament m-series-analysis list-m-series-models benchmark-m-series benchmark-training-progression benchmark-comprehensive quick-benchmark benchmark-analysis human-vs-ai human-vs-m-series list-ai-types logged-game

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================

venv: ## Create virtual environment
	python3 -m venv venv

install: venv ## Install dependencies (traditional method)
	venv/bin/pip install --upgrade pip
	venv/bin/pip install -r requirements.txt

install-pip: venv ## Install package with pip (editable mode)
	venv/bin/pip install --upgrade pip
	venv/bin/pip install -e .

# =============================================================================
# TESTING
# =============================================================================

test: install ## Run tests
	venv/bin/pytest tests/ -v

# =============================================================================
# BASIC GAME MODES
# =============================================================================

run: install ## Run the euchre game
	venv/bin/python -m euchre.cli_main play

ai-game: install ## Play AI vs AI game with full verbosity
	@echo "🤖 Starting AI vs AI Game with full verbosity..."
	venv/bin/python -m euchre.cli_main --very-verbose ai-vs-ai

ai-game-ncurses: install ## Play AI vs AI game with ncurses interface
	@echo "🖥️  Starting ncurses interface..."
	venv/bin/python -m euchre.cli_main ncurses

ncurses: install ## Run AI vs AI euchre game with ncurses interface
	venv/bin/python -m euchre.cli_main ncurses

logged: install ## Run AI vs AI euchre game with logging (no ncurses)
	venv/bin/python -m euchre.cli_main logged-game

logged-game: install ## Play AI vs AI game with logging
	@echo "📝 Starting logged game..."
	venv/bin/python -m euchre.cli_main logged-game

# =============================================================================
# ADVANCED GAME MODES
# =============================================================================

profiles: install ## Run AI vs AI game with custom profiles and risk ratios
	venv/bin/python -m euchre.cli_main ai-profiles

mass-games: install ## Run thousands of games in parallel
	venv/bin/python -m euchre.cli_main run-mass-games

human-vs-ai: install ## Play human vs AI game with configurable AI types
	@echo "🎮 Starting Human vs AI Game..."
	@echo "Usage: make human-vs-ai PLAYER_NAME=YourName POSITION=0 PARTNER_AI=balanced OPP1_AI=aggressive OPP2_AI=conservative"
	@echo "Available AI types: aggressive, conservative, balanced, opportunistic, magnus, maverick, mentor, mystic"
	@echo "Example: make human-vs-ai PLAYER_NAME=Alice POSITION=0 PARTNER_AI=magnus OPP1_AI=aggressive OPP2_AI=balanced"
	venv/bin/python -m euchre.cli_main human-vs-ai $(PLAYER_NAME) --your-position $(POSITION) --partner-ai-type $(PARTNER_AI) --opponent1-ai-type $(OPP1_AI) --opponent2-ai-type $(OPP2_AI)

human-vs-m-series: install ## Play human vs M-Series AI specifically
	@echo "🧠 Starting Human vs M-Series AI Game..."
	@echo "Usage: make human-vs-m-series PLAYER_NAME=YourName POSITION=0 MODEL=magnus MODEL_PATH=path/to/model.pth"
	@echo "Available M-Series models: magnus, maverick, mentor, mystic"
	@echo "Example: make human-vs-m-series PLAYER_NAME=Alice POSITION=0 MODEL=magnus MODEL_PATH=models/magnus_trained.pth"
	venv/bin/python -m euchre.cli_main human-vs-m-series $(PLAYER_NAME) --your-position $(POSITION) --m-series-model $(MODEL) --model-path $(MODEL_PATH)

# =============================================================================
# ANALYSIS AND UTILITIES
# =============================================================================

analyze: install ## Analyze game results and generate statistics
	venv/bin/python -m euchre.cli_main analyze-games

cleanup: install ## Clean up old game result files
	venv/bin/python -m euchre.cli_main cleanup-games

jupyter: install ## Start Jupyter Notebook
	venv/bin/python -m euchre.cli_main jupyter

jupyter-lab: install ## Start Jupyter Lab
	venv/bin/python -m euchre.cli_main jupyter-lab

list-ai-types: install ## List all available AI types including M-Series models
	@echo "🤖 Listing Available AI Types..."
	venv/bin/python -m euchre.cli_main list-ai-types

list-players: install ## List available trained AI players
	venv/bin/python -m euchre.cli_main list-players

# =============================================================================
# CLEANUP
# =============================================================================

clean: ## Clean up generated files
	rm -rf venv/
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete

# =============================================================================
# TRADITIONAL AI TRAINING
# =============================================================================

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

play-trained: install ## Play a game with trained AI players
	venv/bin/python -m euchre.cli_main play-trained-players --player1 Alice --player2 Bob --player3 Charlie --player4 David

tournament: install ## Run a tournament between trained players
	venv/bin/python -m euchre.cli_main tournament --num-games 100

# =============================================================================
# NEURAL NETWORK MODELS
# =============================================================================

neural-tournament: install ## Run neural network tournament (Alice vs Bob, 1000 games)
	venv/bin/python -m euchre.cli_main run-neural-games -m1 Alice -m2 Bob -n 1000

neural-analysis: install ## Analyze neural network tournament results
	venv/bin/python analyze_neural_results.py

list-neural-models: install ## List available neural network models
	venv/bin/python -m euchre.cli_main list-neural-models

benchmark: install ## Run performance benchmark comparing float vs integer models
	venv/bin/python -m euchre.cli_main benchmark --device cpu --save-results

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
	@echo "🚀 Quick M-Series Training (Smoke Test)"
	@echo "Training 500 games over 50 epochs for quick validation..."
	venv/bin/python unified_trainer.py \
		--epochs 50 \
		--total-games 500 \
		--batch-size 32 \
		--learning-rate 0.001 \
		--device auto \
		--max-gpus 1

train-m-series-intensive: install-gpu ## Intensive M-Series training for production models
	@echo "🔥 Intensive M-Series Training (Production)"
	@echo "Training 50000 games over 2000 epochs for production models..."
	venv/bin/python unified_trainer.py \
		--epochs 2000 \
		--total-games 50000 \
		--batch-size 128 \
		--learning-rate 0.0005 \
		--device auto \
		--max-gpus 4

evaluate-m-series: install-gpu ## Evaluate trained M-Series models
	@echo "Evaluating M-Series AI models..."
	@if [ -d "trained_models/unified_trained" ]; then \
		ls -la trained_models/unified_trained/*.json 2>/dev/null | sed 's/.*\//  - /' || echo "  No trained models found"; \
	else \
		echo "  No trained models directory found"; \
	fi
	@echo ""
	@echo "To evaluate a specific model, run:"
	@echo "  venv/bin/python -m euchre.ai_model.model_evaluator --model path/to/model.pth --games 100"

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

# =============================================================================
# M-SERIES BENCHMARKING AND TOURNAMENTS
# =============================================================================

benchmark-m-series: install-gpu ## Run comprehensive M-Series vs Traditional AI benchmark
	@echo "🏆 M-Series vs Traditional AI Benchmark"
	@echo "Running comprehensive benchmark comparing M-Series AI against traditional AI..."
	venv/bin/python m_series_benchmark.py --mode comprehensive --games 200

benchmark-training-progression: install-gpu ## Run training progression benchmark to show improvement
	@echo "📈 Training Progression Benchmark"
	@echo "Running benchmark to show how M-Series AI improves during training..."
	venv/bin/python m_series_benchmark.py --mode progression --games 100

benchmark-comprehensive: install-gpu ## Run both tournament and progression benchmarks
	@echo "🔍 Comprehensive Benchmark Suite"
	@echo "Running both tournament and progression benchmarks..."
	$(MAKE) benchmark-m-series
	$(MAKE) benchmark-training-progression

quick-benchmark: install-gpu ## Run quick benchmark (50 games per matchup)
	@echo "⚡ Running Quick Benchmark (50 games per matchup)..."
	venv/bin/python m_series_benchmark.py --mode tournament --games 50

benchmark-analysis: install ## Analyze benchmark results and generate reports
	@echo "📊 Analyzing benchmark results..."
	@if [ -f "m_series_benchmark_*.json" ]; then \
		echo "Found benchmark results:"; \
		ls -la m_series_benchmark_*.json; \
		echo ""; \
		echo "Run: venv/bin/python -c \"import json; print(json.dumps(json.load(open('m_series_benchmark_*.json')), indent=2))\" for detailed results"; \
	else \
		echo "No benchmark results found. Run benchmark-m-series first."; \
	fi

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