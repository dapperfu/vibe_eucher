#!/usr/bin/env python3
"""Test script for GPU acceleration in the euchre AI training system."""

import sys
from pathlib import Path

# Add the euchre package to the path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import time


def test_gpu_detection():
    """Test GPU detection and capabilities."""
    print("🔍 GPU Detection Test")
    print("=" * 50)
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"✅ CUDA is available!")
        print(f"🚀 CUDA version: {torch.version.cuda}")
        print(f"🐍 PyTorch version: {torch.__version__}")
        
        # Get GPU information
        gpu_count = torch.cuda.device_count()
        print(f"🖥️  Number of GPUs: {gpu_count}")
        
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
            gpu_compute = torch.cuda.get_device_properties(i).major
            print(f"   GPU {i}: {gpu_name}")
            print(f"      Memory: {gpu_memory:.1f} GB")
            print(f"      Compute Capability: {gpu_compute}.x")
        
        # Test GPU operations
        print("\n🧪 Testing GPU Operations:")
        device = torch.device("cuda:0")
        
        # Create tensors on GPU
        start_time = time.time()
        x = torch.randn(1000, 1000, device=device)
        y = torch.randn(1000, 1000, device=device)
        
        # Matrix multiplication on GPU
        z = torch.mm(x, y)
        gpu_time = time.time() - start_time
        
        print(f"   GPU matrix multiplication: {gpu_time:.4f} seconds")
        
        # Test on CPU for comparison
        start_time = time.time()
        x_cpu = x.cpu()
        y_cpu = y.cpu()
        z_cpu = torch.mm(x_cpu, y_cpu)
        cpu_time = time.time() - start_time
        
        print(f"   CPU matrix multiplication: {cpu_time:.4f} seconds")
        
        if gpu_time < cpu_time:
            speedup = cpu_time / gpu_time
            print(f"   🚀 GPU is {speedup:.1f}x faster than CPU!")
        else:
            print(f"   ⚠️  GPU is slower than CPU (this is unusual)")
        
        # Memory usage
        allocated = torch.cuda.memory_allocated(device) / 1e6
        reserved = torch.cuda.memory_reserved(device) / 1e6
        print(f"   GPU Memory: {allocated:.1f} MB allocated, {reserved:.1f} MB reserved")
        
        # Clear GPU memory
        del x, y, z
        torch.cuda.empty_cache()
        
    else:
        print("❌ CUDA is not available")
        print("💡 This system will use CPU for training")
    
    # Check MPS (Apple Silicon)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print(f"\n🍎 MPS (Apple Silicon) is available!")
        print(f"   This system can use Apple's Metal Performance Shaders")
    
    print()


def test_training_device():
    """Test the training device setup."""
    print("🧠 Training Device Test")
    print("=" * 50)
    
    try:
        from euchre.ai_model.self_play_trainer import SelfPlayTrainer
        
        # Test different device configurations
        devices = ["auto", "cpu", "cuda"]
        
        for device in devices:
            print(f"\n🔧 Testing device: {device}")
            
            try:
                model_config = {
                    'type': 'standard',
                    'input_size': 128,
                    'hidden_size': 256,
                    'output_size': 64,
                    'risk_embedding_size': 32,
                    'use_risk_attention': True
                }
                
                trainer = SelfPlayTrainer(model_config, "test_output", device)
                print(f"   ✅ Successfully initialized on {trainer.device}")
                
                # Test model creation
                model, player = trainer.create_player_model("TestPlayer", "balanced")
                print(f"   ✅ Model created successfully")
                
                # Clean up
                del trainer, model, player
                if device == "cuda" and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"   ❌ Failed: {e}")
        
    except ImportError as e:
        print(f"❌ Could not import SelfPlayTrainer: {e}")
    
    print()


def test_performance_comparison():
    """Test performance comparison between CPU and GPU."""
    print("⚡ Performance Comparison Test")
    print("=" * 50)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available, skipping performance test")
        return
    
    # Test neural network forward pass
    print("🧠 Testing Neural Network Forward Pass:")
    
    # Create a simple network
    class SimpleNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = torch.nn.Linear(128, 256)
            self.fc2 = torch.nn.Linear(256, 64)
            self.relu = torch.nn.ReLU()
        
        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.fc2(x)
            return x
    
    # Test on CPU
    model_cpu = SimpleNet()
    x_cpu = torch.randn(100, 128)
    
    start_time = time.time()
    for _ in range(100):
        _ = model_cpu(x_cpu)
    cpu_time = time.time() - start_time
    
    print(f"   CPU forward pass (100 iterations): {cpu_time:.4f} seconds")
    
    # Test on GPU
    model_gpu = SimpleNet().cuda()
    x_gpu = torch.randn(100, 128).cuda()
    
    start_time = time.time()
    for _ in range(100):
        _ = model_gpu(x_gpu)
    torch.cuda.synchronize()  # Ensure GPU operations complete
    gpu_time = time.time() - start_time
    
    print(f"   GPU forward pass (100 iterations): {gpu_time:.4f} seconds")
    
    if gpu_time < cpu_time:
        speedup = cpu_time / gpu_time
        print(f"   🚀 GPU is {speedup:.1f}x faster than CPU!")
    else:
        print(f"   ⚠️  GPU is slower than CPU")
    
    # Clean up
    del model_cpu, model_gpu, x_cpu, x_gpu
    torch.cuda.empty_cache()
    
    print()


def main():
    """Run all GPU acceleration tests."""
    print("🚀 Euchre AI GPU Acceleration Test Suite")
    print("=" * 60)
    
    test_gpu_detection()
    test_training_device()
    test_performance_comparison()
    
    print("🎯 Test Summary:")
    print("=" * 60)
    
    if torch.cuda.is_available():
        print("✅ Your system supports GPU acceleration!")
        print("💡 Use 'make generate-profiles-gpu' for GPU-accelerated training")
        print("💡 Use 'make generate-profiles' for automatic device selection")
    else:
        print("❌ Your system does not support GPU acceleration")
        print("💡 Use 'make generate-profiles-cpu' for CPU training")
    
    print("\n📚 Available Make Targets:")
    print("   make generate-profiles      # Auto device selection")
    print("   make generate-profiles-cpu  # CPU only")
    print("   make generate-profiles-gpu  # GPU accelerated")
    
    print("\n🔧 Manual CLI Options:")
    print("   --device auto              # Automatic device selection")
    print("   --device cpu               # Force CPU usage")
    print("   --device cuda              # Force GPU usage")
    print("   --enable-amp               # Enable automatic mixed precision")
    print("   --gpu-memory-fraction 0.9  # Control GPU memory usage")


if __name__ == "__main__":
    main() 