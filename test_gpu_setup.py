#!/usr/bin/env python3
"""
Test GPU Setup for M-Series Training

This script tests the GPU setup to ensure everything is working correctly
before starting training.

Author: Claude Sonnet 4 (claude-3-5-sonnet-20241022)
Generated via Cursor IDE (cursor.sh) with AI assistance
Model: Anthropic Claude 3.5 Sonnet
Generation timestamp: 2025-01-13 00:00:00
Context: Testing GPU setup for M-Series training system
"""

import sys
import os

def test_pytorch_installation():
    """Test PyTorch installation and CUDA availability."""
    print("🔍 Testing PyTorch Installation...")
    
    try:
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
    except ImportError:
        print("❌ PyTorch not installed")
        return False
    
    try:
        import torchvision
        print(f"✅ TorchVision version: {torchvision.__version__}")
    except ImportError:
        print("❌ TorchVision not installed")
        return False
    
    try:
        import torchaudio
        print(f"✅ TorchAudio version: {torchaudio.__version__}")
    except ImportError:
        print("❌ TorchAudio not installed")
        return False
    
    return True

def test_cuda_availability():
    """Test CUDA availability and GPU detection."""
    print("\n🔍 Testing CUDA Availability...")
    
    try:
        import torch
        
        if not torch.cuda.is_available():
            print("❌ CUDA not available")
            print("   This could mean:")
            print("   - PyTorch was installed without CUDA support")
            print("   - NVIDIA drivers are not installed")
            print("   - GPU is not CUDA-compatible")
            return False
        
        print("✅ CUDA is available")
        
        # Check CUDA version
        cuda_version = torch.version.cuda
        print(f"✅ CUDA version: {cuda_version}")
        
        # Check GPU count
        gpu_count = torch.cuda.device_count()
        print(f"✅ GPU count: {gpu_count}")
        
        if gpu_count == 0:
            print("❌ No GPUs detected")
            return False
        
        # Check each GPU
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing CUDA: {e}")
        return False

def test_gpu_training_components():
    """Test GPU training components."""
    print("\n🔍 Testing GPU Training Components...")
    
    try:
        # Test DataParallel
        import torch.nn as nn
        from torch.nn.parallel import DataParallel
        
        # Create a simple model
        model = nn.Linear(10, 1)
        model = model.cuda()
        
        # Test DataParallel
        if torch.cuda.device_count() > 1:
            model = DataParallel(model)
            print("✅ DataParallel working")
        else:
            print("⚠️  DataParallel not tested (only 1 GPU)")
        
        # Test mixed precision
        try:
            from torch.cuda.amp import GradScaler, autocast
            scaler = GradScaler()
            print("✅ Mixed precision components available")
        except ImportError:
            print("❌ Mixed precision components not available")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing training components: {e}")
        return False

def test_memory_usage():
    """Test GPU memory usage."""
    print("\n🔍 Testing GPU Memory Usage...")
    
    try:
        import torch
        
        if not torch.cuda.is_available():
            print("⚠️  Skipping memory test (CUDA not available)")
            return True
        
        # Get memory info
        for i in range(torch.cuda.device_count()):
            torch.cuda.set_device(i)
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            
            print(f"   GPU {i}: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved, {total:.1f} GB total")
        
        # Test memory allocation
        try:
            # Allocate a small tensor
            test_tensor = torch.randn(1000, 1000).cuda()
            print("✅ GPU memory allocation working")
            
            # Clean up
            del test_tensor
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"❌ GPU memory allocation failed: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing memory: {e}")
        return False

def test_training_script():
    """Test if the training script can be imported."""
    print("\n🔍 Testing Training Script...")
    
    try:
        # Check if training script exists
        if not os.path.exists("gpu_train_m_series.py"):
            print("❌ Training script not found: gpu_train_m_series.py")
            return False
        
        print("✅ Training script found")
        
        # Try to import the main components
        sys.path.insert(0, os.path.dirname(__file__))
        
        # Test basic imports
        try:
            from gpu_train_m_series import GPUTrainingConfig, GPUTrainer
            print("✅ Training components can be imported")
        except ImportError as e:
            print(f"❌ Training components import failed: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing training script: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 GPU Training Setup Test")
    print("=" * 50)
    
    tests = [
        ("PyTorch Installation", test_pytorch_installation),
        ("CUDA Availability", test_cuda_availability),
        ("Training Components", test_gpu_training_components),
        ("Memory Usage", test_memory_usage),
        ("Training Script", test_training_script)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! GPU training setup is ready.")
        print("\nNext steps:")
        print("1. Run: make install-gpu")
        print("2. Run: make train-gpu-m-series-fast")
        print("3. Check: make evaluate-gpu-models")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please fix issues before training.")
        print("\nCommon solutions:")
        print("1. Install PyTorch with CUDA: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        print("2. Check NVIDIA drivers: nvidia-smi")
        print("3. Verify CUDA installation: nvcc --version")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 