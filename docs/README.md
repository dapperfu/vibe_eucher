# Euchre Documentation

This directory contains documentation for the Euchre game implementation and AI systems.

## Documentation Structure

### EucherZero
Comprehensive documentation for the EucherZero implementation (AlphaZero/MuZero-inspired RL system).

**Location**: [`eucher_zero/`](./eucher_zero/)

**Start here**: [`eucher_zero/README.md`](./eucher_zero/README.md)

Contains:
- [Implementation Plan](./eucher_zero/implementation_plan.md) - Complete roadmap
- [Technical Specification](./eucher_zero/technical_spec.md) - Code structure and examples
- [Summary](./eucher_zero/summary.md) - Quick reference guide

### Computer Bots
Comprehensive documentation for all computer bot types available in the game.

**Location**: [`bots/`](./bots/)

**Start here**: [`bots/README.md`](./bots/README.md)

Contains documentation for:
- [Random Bot](./bots/random.md) - Random decision-making
- [Heuristic Bot](./bots/heuristic.md) - Rule-based strategy
- [AI Bot](./bots/ai.md) - Weighted heuristics
- [ML Supervised Bot](./bots/ml_supervised.md) - Supervised learning
- [ML RL Bot](./bots/ml_rl.md) - Reinforcement learning
- [ML GAN Bot](./bots/ml_gan.md) - Generative adversarial networks
- [PyTorch Strategic Bot](./bots/pytorch_strategic.md) - Deep neural networks
- [EuchreZero Bot](./bots/euchre_zero.md) - AlphaZero-style MCTS
- [PerceiverMuZero Bot](./bots/perceiver_muzero.md) - Perceiver architecture
- [EucherGo Bot](./bots/euchergo.md) - Hybrid MCTS and neural networks

### General Documentation

- [CUDA Acceleration](./CUDA_ACCELERATION.md) - GPU acceleration guide
- [Generating Training Data](./generating_training_data.md) - Data collection guide
- [Model Inputs](./model_inputs.md) - Model input specifications
- [Training Guide](./training_guide.md) - General training documentation

## Quick Links

### For EucherZero Implementation
1. Read the [EucherZero README](./eucher_zero/README.md)
2. Review the [Implementation Plan](./eucher_zero/implementation_plan.md)
3. Check the [Technical Spec](./eucher_zero/technical_spec.md) for code examples
4. Use the [Summary](./eucher_zero/summary.md) as a quick reference

### For Computer Bots
- See [Bot Documentation](./bots/README.md) for overview of all bot types
- Individual bot pages contain architecture, usage, and training details
- All bots use the plugin system for automatic discovery

### For Existing ML Systems
- See [Training Guide](./training_guide.md) for training existing models
- See [Model Inputs](./model_inputs.md) for input format specifications
- See [Generating Training Data](./generating_training_data.md) for data collection

## Requirements

EucherZero requirements are documented in:
- `../bot_requirements/eucher_zero_requirements.md`

