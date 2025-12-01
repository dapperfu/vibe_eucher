# Using Python 3.13 NoGIL for Training

Python 3.13 NoGIL (no Global Interpreter Lock) has been installed and is available at:
- Executable: `/usr/bin/python3.13-nogil`
- Version: Python 3.13.8 experimental free-threading build

## Benefits

- **True parallel execution**: Multiple threads can run Python code simultaneously
- **Better CPU utilization**: Can utilize all 16 CPU cores for CPU-bound operations
- **Improved performance**: For multi-threaded workloads like MCTS and game generation

## Setup Dependencies

You'll need to install PyTorch and other dependencies for the nogil Python:

```bash
# Install pip if not already available
python3.13-nogil -m ensurepip --upgrade

# Install PyTorch for nogil Python (you may need to build from source or use compatible wheels)
python3.13-nogil -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other project dependencies
python3.13-nogil -m pip install -r requirements.txt
```

## Running Training with NoGIL

Update your training script's shebang or run directly:

```bash
# Option 1: Update shebang in train_eucher_zero.py
# Change first line to: #!/usr/bin/env python3.13-nogil

# Option 2: Run directly
python3.13-nogil scripts/eucher_zero/train_eucher_zero.py --device gpu --duration 12h --batch-size 128
```

## Important Notes

1. **Experimental**: This is still experimental - test thoroughly
2. **Compatibility**: Some C extensions may not be compatible
3. **Performance**: May have ~10-40% single-threaded overhead, but better multi-threaded performance
4. **PyTorch**: PyTorch may need to be built from source for nogil, or use compatible pre-built wheels if available

## Verification

Check that GIL is disabled:
```bash
python3.13-nogil -c "import sys; print('GIL disabled:', hasattr(sys, '_is_gil_enabled') and not sys._is_gil_enabled())"
```

Expected output: `GIL disabled: True`

