# CUDA Acceleration in Training

## Why GPU Speed May Appear Similar to CPU

The training code uses CUDA/GPU, but you may not see significant speedup because:

### 1. **CPU-Bound Game Simulation**
The main bottleneck is the **game simulation itself**, which is pure Python code running on CPU:
- Card dealing, trick playing, game state management
- This takes most of the time during training
- GPU cannot accelerate this part

### 2. **Single-Sample Inference During Gameplay**
During gameplay, ML models make decisions one at a time:
- Each decision uses `batch_size=1` inference
- GPU overhead (memory transfer, kernel launch) dominates for single samples
- CPU is often **faster** for single-sample inference due to no transfer overhead
- GPU only helps with **batch inference** (multiple samples at once)

### 3. **Training Batch Size**
The training code has been optimized:
- **Before**: batch_size=32 (too small for GPU efficiency)
- **After**: batch_size=256 (better GPU utilization)
- Training happens more frequently now

## What GPU Actually Accelerates

GPU acceleration helps with:
- ✅ **Training** (batch_size=256): Forward/backward passes on batches
- ✅ **Batch inference**: Processing multiple game states at once
- ❌ **Single-sample inference**: CPU is faster (no transfer overhead)
- ❌ **Game simulation**: Pure Python, cannot be accelerated

## Performance Expectations

- **Game simulation**: ~same speed on CPU/GPU (CPU-bound)
- **Training**: 5-20x faster on GPU (depending on GPU model)
- **Single inference**: Faster on CPU (no GPU transfer overhead)
- **Batch inference**: Much faster on GPU

## How to Verify GPU Usage

1. Check training output - it should show:
   ```
   Training RL agent on device: cuda
   GPU: <Your GPU Name>
   ```

2. Monitor GPU usage during training:
   ```bash
   watch -n 1 nvidia-smi
   ```
   You should see GPU utilization spike during training steps (not during game simulation).

3. Check if CUDA is available:
   ```python
   import torch
   print(torch.cuda.is_available())
   print(torch.cuda.get_device_name(0))
   ```

## Optimizations Made

1. **Increased training batch size**: 32 → 256
2. **More frequent training**: Trains every game once buffer has 256+ samples
3. **Better diagnostics**: Shows GPU info and explains bottlenecks
4. **CPU fallback for inference**: Single-sample inference uses CPU (faster)

## Future Optimizations

To get more GPU benefit:
1. **Batch game state collection**: Collect multiple game states, then batch inference
2. **Parallel game simulation**: Run multiple games in parallel (complex)
3. **Larger training batches**: Increase to 512-1024 if memory allows
4. **Mixed precision training**: Use FP16 to speed up training

## Summary

**GPU is being used for training**, but the speedup may not be obvious because:
- Most time is spent in CPU-bound game simulation
- Single-sample inference is faster on CPU
- Only the training step (batch processing) benefits from GPU

To see GPU benefit, look at the **training time per batch**, not the overall game speed.

