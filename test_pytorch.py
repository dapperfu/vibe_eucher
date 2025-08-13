#!/usr/bin/env python3
"""Simple PyTorch test script."""

try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Device: {torch.device('cpu')}")
    
    # Test basic operations
    x = torch.randn(3, 3)
    y = torch.randn(3, 3)
    z = torch.mm(x, y)
    print(f"Matrix multiplication test: {z.shape}")
    print("PyTorch is working!")
    
except Exception as e:
    print(f"PyTorch error: {e}")
    import traceback
    traceback.print_exc() 