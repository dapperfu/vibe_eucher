.PHONY: help venv test run clean train-ai evaluate-ai ai-game ai-game-ncurses ncurses logged profiles mass-games analyze cleanup jupyter jupyter-lab train-self-play train-integer-vs-float generate-profiles generate-profiles-cpu generate-profiles-gpu list-players play-trained tournament benchmark neural-tournament neural-analysis list-neural-models train-level2 train-level2-fast train-level2-intensive evaluate-level2 train-level2-vs-traditional level2-tournament level2-analysis list-level2-models benchmark-level2 benchmark-training-progression benchmark-comprehensive quick-benchmark benchmark-analysis human-vs-ai human-vs-level2 list-ai-types logged-game train-level3 train-level3-fast train-level3-medium train-level3-deep evaluate-level3 list-level3-models

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z0-9_-]+:.*##' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================

venv: ## Create virtual environment
	python3 -m venv venv

install: venv ## Install dependencies (traditional method)
	venv/bin/pip install --upgrade pip
	venv/bin/pip install uv
	venv/bin/uv pip install --python venv/bin/python -r requirements.txt

install-pip: venv/lib/python3.12/site-packages/euchre.egg-link ## Install package with pip (editable mode)
	@echo "Package already installed in editable mode"

venv/lib/python3.12/site-packages/euchre.egg-link: venv
	venv/bin/uv pip install --python venv/bin/python -e .
	touch $@

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
	venv/bin/python -m euchre.cli_main ai-vs-ai --very-verbose

ai-game-ncurses: install ## Play AI vs AI game with ncurses interface
	@echo "🖥️  Starting ncurses interface..."
	venv/bin/python -m euchre.cli_main ncurses

ncurses: install ## Run AI vs AI euchre game with ncurses interface
	venv/bin/python -m euchre.cli_main ncurses

ncurses-human: install ## Run human vs AI euchre game with ncurses interface
	@echo "🎮 Starting Human vs AI Ncurses Game..."
	@echo "Usage: make ncurses-human POSITION=2"
	@echo "Positions: 0=North, 1=East, 2=South (default), 3=West"
	venv/bin/python -m euchre.cli_main ncurses-human-vs-ai --position $(or $(POSITION),2)

demo-ncurses: install ## Run interactive demo of enhanced ncurses game
	@echo "🎴 Starting Enhanced Ncurses Game Demo..."
	venv/bin/python demo_ncurses_game.py

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
	@echo "Usage: make human-vs-ai PLAYER_NAME=YourName POSITION=0 PARTNER_AI=level1_balanced OPP1_AI=level1_aggressive OPP2_AI=level2_strategic"
	@echo "Available AI types: level1_aggressive, level1_conservative, level1_balanced, level1_opportunistic, level2_strategic, level2_aggressive, level2_balanced, level2_intuitive"
	@echo "Example: make human-vs-ai PLAYER_NAME=Alice POSITION=0 PARTNER_AI=level1_balanced OPP1_AI=level1_aggressive OPP2_AI=level2_strategic"
	venv/bin/python -m euchre.cli_main human-vs-ai $(PLAYER_NAME) --your-position $(POSITION) --partner-ai-type $(PARTNER_AI) --opponent1-ai-type $(OPP1_AI) --opponent2-ai-type $(OPP2_AI)

human-vs-level1: install ## Play human vs Level1 AI specifically
	@echo "🧠 Starting Human vs Level1 AI Game..."
	@echo "Usage: make human-vs-level1 PLAYER_NAME=YourName POSITION=0 MODEL=level1_aggressive"
	@echo "Available Level 1 models: level1_aggressive, level1_conservative, level1_balanced, level1_opportunistic"
	@echo "Example: make human-vs-level1 PLAYER_NAME=Alice POSITION=0 MODEL=level1_aggressive"
	venv/bin/python -m euchre.cli_main human-vs-level1 $(PLAYER_NAME) --your-position $(POSITION) --level1-model $(MODEL)

human-vs-level2: install ## Play human vs Level2 AI specifically
	@echo "🧠 Starting Human vs Level2 AI Game..."
	@echo "Usage: make human-vs-level2 PLAYER_NAME=YourName POSITION=0 MODEL=level2_strategic MODEL_PATH=path/to/model.pth"
	@echo "Available Level 2 models: level2_strategic, level2_aggressive, level2_balanced, level2_intuitive"
	@echo "Example: make human-vs-level2 PLAYER_NAME=Alice POSITION=0 MODEL=level2_strategic MODEL_PATH=models/level2_strategic_trained.pth"
	venv/bin/python -m euchre.cli_main human-vs-level2 $(PLAYER_NAME) --your-position $(POSITION) --level2-model $(MODEL) --model-path $(MODEL_PATH)

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
	find . -type f -name "euchre_game_*.txt" -delete

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
# LEVEL2 AI TRAINING TARGETS
# =============================================================================

install-gpu: venv/lib/python3.12/site-packages/torch ## Install GPU dependencies for Level2 training
	@echo "GPU dependencies already installed"
	@echo "Checking availability..."
	venv/bin/python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA/ROCm available: {torch.cuda.is_available()}'); print(f'Device count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}')"

venv/lib/python3.12/site-packages/torch: install
	venv/bin/uv pip install --python venv/bin/python torch torchvision torchaudio

train-level2: install-gpu ## Train Level2 AI models using unified trainer
	@echo "Starting Level2 AI training with unified trainer..."
	@echo "This will automatically detect and use the best available backend (GPU/CPU)"
	@echo "Training 20000 games over 1000 epochs..."
	venv/bin/python unified_trainer.py \
		--epochs 1000 \
		--total-games 20000 \
		--batch-size 64 \
		--learning-rate 0.001 \
		--device auto \
		--max-gpus 2

train-level2-fast: install-gpu ## Quick Level2 training for smoke testing (<1000 games, few epochs)
	@echo "🚀 Quick Level2 Training (Smoke Test)"
	@echo "Training 500 games over 50 epochs for quick validation..."
	venv/bin/python unified_trainer.py \
		--epochs 50 \
		--total-games 500 \
		--batch-size 32 \
		--learning-rate 0.001 \
		--device auto \
		--max-gpus 1

train-level2-intensive: install-gpu ## Intensive Level2 training for production models
	@echo "🔥 Intensive Level2 Training (Production)"
	@echo "Training 50000 games over 2000 epochs for production models..."
	venv/bin/python unified_trainer.py \
		--epochs 2000 \
		--total-games 50000 \
		--batch-size 128 \
		--learning-rate 0.0005 \
		--device auto \
		--max-gpus 4

evaluate-level2: install-gpu ## Evaluate trained Level2 models
	@echo "Evaluating Level2 AI models..."
	@if [ -d "trained_models/unified_trained" ]; then \
		ls -la trained_models/unified_trained/*.json 2>/dev/null | sed 's/.*\//  - /' || echo "  No trained models found"; \
	else \
		echo "  No trained models directory found"; \
	fi
	@echo ""
	@echo "To evaluate a specific model, run:"
	@echo "  venv/bin/python -m euchre.ai_model.model_evaluator --model path/to/model.pth --games 100"

train-level2-vs-traditional: install-gpu ## Train Level2 models specifically to compete against traditional AI
	@echo "Training Level2 models to compete against traditional AI..."
	@echo "This will focus on strategies that outperform traditional rule-based AI..."
	venv/bin/python unified_trainer.py \
		--epochs 1500 \
		--total-games 30000 \
		--batch-size 64 \
		--learning-rate 0.001 \
		--device auto \
		--max-gpus 2

# =============================================================================
# LEVEL2 BENCHMARKING AND TOURNAMENTS
# =============================================================================

benchmark-level2: install-gpu ## Run comprehensive Level2 vs Traditional AI benchmark
	@echo "🏆 Level2 vs Traditional AI Benchmark"
	@echo "Running comprehensive benchmark comparing Level2 AI against traditional AI..."
	venv/bin/python m_series_benchmark.py --mode comprehensive --games 200

benchmark-training-progression: install-gpu ## Run training progression benchmark to show improvement
	@echo "📈 Training Progression Benchmark"
	@echo "Running benchmark to show how Level2 AI improves during training..."
	venv/bin/python m_series_benchmark.py --mode progression --games 100

benchmark-comprehensive: install-gpu ## Run both tournament and progression benchmarks
	@echo "🔍 Comprehensive Benchmark Suite"
	@echo "Running both tournament and progression benchmarks..."
	$(MAKE) benchmark-level2
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
		echo "No benchmark results found. Run benchmark-level2 first."; \
	fi

level2-tournament: install-gpu ## Run tournament: Level2 AI vs Traditional AI
	@echo "🏆 Level2 AI vs Traditional AI Tournament"
	@echo "Running tournament with trained Level2 models against traditional AI..."
	venv/bin/python ai_tournament.py \
		--m-series-models trained_models/unified_trained \
		--traditional-ai-types balanced aggressive conservative opportunistic \
		--games-per-match 100 \
		--output-dir tournament_results/level2_vs_traditional

level2-analysis: install-gpu ## Analyze Level2 vs Traditional AI tournament results
	@echo "📊 Analyzing Level2 vs Traditional AI tournament results..."
	@if [ -d "tournament_results/level2_vs_traditional" ]; then \
		venv/bin/python -c " \
import json; \
import os; \
import glob; \
results_dir = 'tournament_results/level2_vs_traditional'; \
results_files = glob.glob(os.path.join(results_dir, '*.json')); \
if results_files: \
    print(f'Found {len(results_files)} tournament result files:'); \
    for f in results_files: \
        print(f'  - {os.path.basename(f)}'); \
    print('\\nRun: venv/bin/python analyze_neural_results.py for detailed analysis'); \
else: \
    print('No tournament results found. Run level2-tournament first.'); \
"; \
	else \
		echo "No tournament results directory found. Run level2-tournament first."; \
	fi

list-level2-models: install-gpu ## List available Level2 models
	@echo "Available Level2 AI models:"
	@if [ -d "trained_models/unified_trained" ]; then \
		ls -la trained_models/unified_trained/*.json 2>/dev/null | sed 's/.*\//  - /' || echo "  No trained models found"; \
	else \
		echo "  No trained models directory found"; \
	fi

# =============================================================================
# LEVEL3 AI TRAINING TARGETS
# =============================================================================

train-level3: install-gpu ## Train Level3 AI models (default configuration)
	@echo "🧠 Level3 AI Training (Default)"
	@echo "Training Level3 AI with default configuration..."
	@echo "This will use 2048 input features, 1024 hidden units, 8 layers"
	@echo "Expected training time: 2-4 weeks on 24GB GPU"
	venv/bin/python train_level3_models.py \
		--input-size 2048 \
		--hidden-size 1024 \
		--num-layers 8 \
		--epochs 100 \
		--batch-size 32 \
		--learning-rate 0.0001 \
		--data-dir training_data/level3 \
		--output-dir trained_models/level3

train-level3-fast: install-gpu ## Fast Level3 training (smoke test - 1k games, 50 epochs)
	@echo "⚡ Level3 AI Fast Training (Smoke Test)"
	@echo "Training Level3 AI with minimal configuration for testing..."
	@echo "This will use 2048 input features, 512 hidden units, 4 layers"
	@echo "Expected training time: 2-4 hours on 24GB GPU"
	venv/bin/python train_level3_models.py \
		--input-size 2048 \
		--hidden-size 512 \
		--num-layers 4 \
		--epochs 50 \
		--batch-size 64 \
		--learning-rate 0.001 \
		--data-dir training_data/level3 \
		--output-dir trained_models/level3_fast \
		--max-samples 1000

train-level3-medium: install-gpu ## Medium Level3 training (10k games, 200 epochs)
	@echo "🔄 Level3 AI Medium Training (Development)"
	@echo "Training Level3 AI with medium configuration for development..."
	@echo "This will use 2048 input features, 768 hidden units, 6 layers"
	@echo "Expected training time: 1-2 days on 24GB GPU"
	venv/bin/python train_level3_models.py \
		--input-size 2048 \
		--hidden-size 768 \
		--num-layers 6 \
		--epochs 200 \
		--batch-size 48 \
		--learning-rate 0.0005 \
		--data-dir training_data/level3 \
		--output-dir trained_models/level3_medium \
		--max-samples 10000

train-level3-deep: install-gpu ## Deep Level3 training (100k+ games, 500+ epochs)
	@echo "🔥 Level3 AI Deep Training (Production)"
	@echo "Training Level3 AI with full configuration for production..."
	@echo "This will use 2048 input features, 1024 hidden units, 8 layers"
	@echo "Expected training time: 1-2 weeks on 24GB GPU"
	venv/bin/python train_level3_models.py \
		--input-size 2048 \
		--hidden-size 1024 \
		--num-layers 8 \
		--epochs 500 \
		--batch-size 32 \
		--learning-rate 0.0001 \
		--data-dir training_data/level3 \
		--output-dir trained_models/level3_deep \
		--max-samples 100000

evaluate-level3: install-gpu ## Evaluate trained Level3 models
	@echo "🧠 Evaluating Level3 AI models..."
	@echo "Available Level3 model directories:"
	@if [ -d "trained_models/level3" ]; then \
		echo "  - level3 (default training)"; \
		ls -la trained_models/level3/*.pth 2>/dev/null | sed 's/.*\//    /' || echo "    No trained models found"; \
	fi
	@if [ -d "trained_models/level3_fast" ]; then \
		echo "  - level3_fast (fast training)"; \
		ls -la trained_models/level3_fast/*.pth 2>/dev/null | sed 's/.*\//    /' || echo "    No trained models found"; \
	fi
	@if [ -d "trained_models/level3_medium" ]; then \
		echo "  - level3_medium (medium training)"; \
		ls -la trained_models/level3_medium/*.pth 2>/dev/null | sed 's/.*\//    /' || echo "    No trained models found"; \
	fi
	@if [ -d "trained_models/level3_deep" ]; then \
		echo "  - level3_deep (deep training)"; \
		ls -la trained_models/level3_deep/*.pth 2>/dev/null | sed 's/.*\//    /' || echo "    No trained models found"; \
	fi
	@echo ""
	@echo "To evaluate a specific model, run:"
	@echo "  venv/bin/python -m euchre.ai_model.model_evaluator --model path/to/model.pth --games 100"

list-level3-models: install-gpu ## List available Level3 models
	@echo "🧠 Available Level3 AI models:"
	@echo "Checking all Level3 model directories..."
	@for dir in level3 level3_fast level3_medium level3_deep; do \
		if [ -d "trained_models/$$dir" ]; then \
			echo "  $$dir:"; \
			ls -la trained_models/$$dir/*.pth 2>/dev/null | sed 's/.*\//    - /' || echo "    No trained models found"; \
		fi; \
	done
	@echo ""
	@echo "To see detailed model information, run:"
	@echo "  venv/bin/python -c \"import torch; model = torch.load('path/to/model.pth'); print(f'Model config: {model.get(\"model_config\", \"N/A\")}')\"" 