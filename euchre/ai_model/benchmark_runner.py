"""
Benchmark runner for comparing float vs integer-based Euchre neural networks.

This module provides tools to benchmark the performance differences between
the standard float-based EuchreNN and the integer-based EuchreNNInt.
"""

import time
import torch
import numpy as np
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import json
import os
from pathlib import Path

from .euchre_nn import EuchreNN, RiskParameters, create_euchre_model, create_risk_profile
from .euchre_nn_int import EuchreNNInt, RiskParametersInt, create_euchre_model_int, create_risk_profile_int
from .game_state_encoder import GameStateEncoder


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    model_type: str  # "float" or "int"
    operation: str   # "forward_pass", "inference", "training_step"
    batch_size: int
    input_size: int
    duration_ms: float
    memory_usage_mb: float
    throughput: float  # operations per second
    error_rate: float  # if applicable


class ModelBenchmarker:
    """Benchmark runner for comparing float vs integer models."""
    
    def __init__(self, device: str = "cpu", num_warmup_runs: int = 10):
        self.device = device
        self.num_warmup_runs = num_warmup_runs
        self.results: List[BenchmarkResult] = []
        
        # Create models for benchmarking
        model_config = {
            'type': 'standard',
            'input_size': 128,  # Float model always uses 128
            'hidden_size': 256,
            'output_size': 64,
            'risk_embedding_size': 32,
            'use_risk_attention': True
        }
        self.float_model = create_euchre_model(model_config)
        # Int model will be recreated for each input size test
        self.int_model = None
        
        # Create risk profiles
        self.float_risk = create_risk_profile("balanced")
        self.int_risk = create_risk_profile_int("balanced")
        
        # Game state encoder
        self.encoder = GameStateEncoder()
        
        # Warm up models
        self._warmup_models()
    
    def _warmup_models(self) -> None:
        """Warm up models to ensure consistent performance measurements."""
        print("Warming up models...")
        
        # Create dummy input
        dummy_input = torch.randn(1, 128, device=self.device)
        
        # Warm up float model
        for _ in range(self.num_warmup_runs):
            with torch.no_grad():
                _ = self.float_model(dummy_input, self.float_risk)
        
        # Warm up int model (create a default one for warmup)
        int_model_warmup = create_euchre_model_int(input_size=128, hidden_size=256, risk_embedding_size=32, device=self.device)
        for _ in range(self.num_warmup_runs):
            with torch.no_grad():
                _ = int_model_warmup(dummy_input, self.int_risk)
        
        print("Models warmed up successfully.")
    
    def _measure_memory(self) -> float:
        """Measure current memory usage in MB."""
        if torch.cuda.is_available() and self.device != "cpu":
            return torch.cuda.memory_allocated(self.device) / 1024 / 1024
        else:
            # For CPU, we can't easily measure memory usage
            return 0.0
    
    def benchmark_forward_pass(self, batch_sizes: List[int] = [1, 4, 8, 16], 
                              input_sizes: List[int] = [64, 128, 256]) -> List[BenchmarkResult]:
        """Benchmark forward pass performance."""
        print("Benchmarking forward pass performance...")
        
        for batch_size in batch_sizes:
            for input_size in input_sizes:
                # Create input tensors with correct sizes for each model
                # Float model expects 128 (it adds risk embeddings internally)
                # Int model expects the specified input_size
                float_input = torch.randn(batch_size, 128, device=self.device)
                int_input = torch.randn(batch_size, input_size, device=self.device)
                
                # Create int model for this input size
                int_model = create_euchre_model_int(input_size=input_size, hidden_size=256, risk_embedding_size=32, device=self.device)
                
                # Benchmark float model
                start_time = time.time()
                memory_before = self._measure_memory()
                
                with torch.no_grad():
                    for _ in range(100):  # Multiple runs for accuracy
                        _ = self.float_model(float_input, self.float_risk)
                
                duration = (time.time() - start_time) * 1000  # Convert to ms
                memory_after = self._measure_memory()
                memory_used = memory_after - memory_before
                
                float_result = BenchmarkResult(
                    model_type="float",
                    operation="forward_pass",
                    batch_size=batch_size,
                    input_size=input_size,
                    duration_ms=duration,
                    memory_usage_mb=memory_used,
                    throughput=100.0 / (duration / 1000.0),  # ops per second
                    error_rate=0.0
                )
                self.results.append(float_result)
                
                # Benchmark int model
                start_time = time.time()
                memory_before = self._measure_memory()
                
                with torch.no_grad():
                    for _ in range(100):  # Multiple runs for accuracy
                        _ = int_model(int_input, self.int_risk)
                
                duration = (time.time() - start_time) * 1000  # Convert to ms
                memory_after = self._measure_memory()
                memory_used = memory_after - memory_before
                
                int_result = BenchmarkResult(
                    model_type="int",
                    operation="forward_pass",
                    batch_size=batch_size,
                    input_size=input_size,
                    duration_ms=duration,
                    memory_usage_mb=memory_used,
                    throughput=100.0 / (duration / 1000.0),  # ops per second
                    error_rate=0.0
                )
                self.results.append(int_result)
                
                print(f"Batch: {batch_size}, Input: {input_size}")
                print(f"  Float: {float_result.duration_ms:.2f}ms, {float_result.throughput:.1f} ops/s")
                print(f"  Int:   {int_result.duration_ms:.2f}ms, {int_result.throughput:.1f} ops/s")
                print(f"  Speedup: {float_result.duration_ms / int_result.duration_ms:.2f}x")
        
        return self.results
    
    def benchmark_inference(self, num_games: int = 1000) -> List[BenchmarkResult]:
        """Benchmark inference performance on simulated game states."""
        print(f"Benchmarking inference performance on {num_games} game states...")
        
        # Generate random game states
        game_states = []
        for _ in range(num_games):
            # Create a random game state (simplified)
            hand = torch.randint(0, 52, (6,), device=self.device)  # 6 cards
            trump_suit = torch.randint(0, 4, (1,), device=self.device)
            game_context = torch.randn(6, device=self.device)  # 6 context features
            
            game_states.append((hand, trump_suit, game_context))
        
        # Benchmark float model inference
        start_time = time.time()
        memory_before = self._measure_memory()
        
        with torch.no_grad():
            for hand, trump_suit, game_context in game_states:
                # Combine features
                combined_input = torch.cat([hand.float(), trump_suit.float(), game_context])
                combined_input = combined_input.unsqueeze(0)  # Add batch dimension
                
                # Pad to expected input size
                if combined_input.size(1) < 128:
                    padding = torch.zeros(1, 128 - combined_input.size(1), device=self.device)
                    combined_input = torch.cat([combined_input, padding], dim=1)
                elif combined_input.size(1) > 128:
                    combined_input = combined_input[:, :128]
                
                _ = self.float_model(combined_input, self.float_risk)
        
        float_duration = (time.time() - start_time) * 1000
        memory_after = self._measure_memory()
        float_memory = memory_after - memory_before
        
        # Benchmark int model inference
        start_time = time.time()
        memory_before = self._measure_memory()
        
        # Create int model for inference benchmark
        int_model = create_euchre_model_int(input_size=128, hidden_size=256, risk_embedding_size=32, device=self.device)
        
        with torch.no_grad():
            for hand, trump_suit, game_context in game_states:
                # Combine features
                combined_input = torch.cat([hand.float(), trump_suit.float(), game_context])
                combined_input = combined_input.unsqueeze(0)  # Add batch dimension
                
                # Pad to expected input size
                if combined_input.size(1) < 128:
                    padding = torch.zeros(1, 128 - combined_input.size(1), device=self.device)
                    combined_input = torch.cat([combined_input, padding], dim=1)
                elif combined_input.size(1) > 128:
                    combined_input = combined_input[:, :128]
                
                _ = int_model(combined_input, self.int_risk)
        
        int_duration = (time.time() - start_time) * 1000
        memory_after = self._measure_memory()
        int_memory = memory_after - memory_before
        
        # Create results
        float_result = BenchmarkResult(
            model_type="float",
            operation="inference",
            batch_size=num_games,
            input_size=128,
            duration_ms=float_duration,
            memory_usage_mb=float_memory,
            throughput=num_games / (float_duration / 1000.0),
            error_rate=0.0
        )
        
        int_result = BenchmarkResult(
            model_type="int",
            operation="inference",
            batch_size=num_games,
            input_size=128,
            duration_ms=int_duration,
            memory_usage_mb=int_memory,
            throughput=num_games / (int_duration / 1000.0),
            error_rate=0.0
        )
        
        self.results.extend([float_result, int_result])
        
        print(f"Inference Benchmark Results:")
        print(f"  Float: {float_duration:.2f}ms, {float_result.throughput:.1f} games/s")
        print(f"  Int:   {int_duration:.2f}ms, {int_result.throughput:.1f} games/s")
        print(f"  Speedup: {float_duration / int_duration:.2f}x")
        
        return [float_result, int_result]
    
    def benchmark_training_step(self, batch_sizes: List[int] = [1, 4, 8]) -> List[BenchmarkResult]:
        """Benchmark training step performance."""
        print("Benchmarking training step performance...")
        
        for batch_size in batch_sizes:
            # Create input and target tensors
            input_tensor = torch.randn(batch_size, 128, device=self.device)
            # Float model outputs [batch_size, 2] for trump decisions, int model outputs [batch_size, 1]
            trump_targets_float = torch.randint(0, 2, (batch_size, 2), device=self.device).float()
            trump_targets_int = torch.randint(0, 2, (batch_size, 1), device=self.device).float()
            # Float model outputs [batch_size, 5] for card selection, int model outputs [batch_size, 52]
            card_targets_float = torch.randint(0, 5, (batch_size,), device=self.device)
            card_targets_int = torch.randint(0, 52, (batch_size,), device=self.device)
            
            # Create int model for this batch size
            int_model = create_euchre_model_int(input_size=128, hidden_size=256, risk_embedding_size=32, device=self.device)
            
            # Loss function and optimizer
            criterion = torch.nn.BCEWithLogitsLoss()
            optimizer_float = torch.optim.Adam(self.float_model.parameters(), lr=0.001)
            optimizer_int = torch.optim.Adam(int_model.parameters(), lr=0.001)
            
            # Benchmark float model training
            start_time = time.time()
            memory_before = self._measure_memory()
            
            for _ in range(10):  # Multiple training steps
                optimizer_float.zero_grad()
                float_outputs = self.float_model(input_tensor, self.float_risk)
                trump_logits = float_outputs['trump_decision']
                card_logits = float_outputs['card_selection']
                
                trump_loss = criterion(trump_logits, trump_targets_float)
                card_loss = torch.nn.functional.cross_entropy(card_logits, card_targets_float)
                total_loss = trump_loss + card_loss
                
                total_loss.backward()
                optimizer_float.step()
            
            float_duration = (time.time() - start_time) * 1000
            memory_after = self._measure_memory()
            float_memory = memory_after - memory_before
            
            # Benchmark int model training
            start_time = time.time()
            memory_before = self._measure_memory()
            
            for _ in range(10):  # Multiple training steps
                optimizer_int.zero_grad()
                trump_logits, card_logits = int_model(input_tensor, self.int_risk)
                
                trump_loss = criterion(trump_logits, trump_targets_int)
                card_loss = torch.nn.functional.cross_entropy(card_logits, card_targets_int)
                total_loss = trump_loss + card_loss
                
                total_loss.backward()
                optimizer_int.step()
            
            int_duration = (time.time() - start_time) * 1000
            memory_after = self._measure_memory()
            int_memory = memory_after - memory_before
            
            # Create results
            float_result = BenchmarkResult(
                model_type="float",
                operation="training_step",
                batch_size=batch_size,
                input_size=128,
                duration_ms=float_duration,
                memory_usage_mb=float_memory,
                throughput=10.0 / (float_duration / 1000.0),
                error_rate=0.0
            )
            
            int_result = BenchmarkResult(
                model_type="int",
                operation="training_step",
                batch_size=batch_size,
                input_size=128,
                duration_ms=int_duration,
                memory_usage_mb=int_memory,
                throughput=10.0 / (int_duration / 1000.0),
                error_rate=0.0
            )
            
            self.results.extend([float_result, int_result])
            
            print(f"Training Benchmark (Batch: {batch_size}):")
            print(f"  Float: {float_duration:.2f}ms, {float_result.throughput:.1f} steps/s")
            print(f"  Int:   {int_duration:.2f}ms, {int_result.throughput:.1f} steps/s")
            print(f"  Speedup: {float_duration / int_duration:.2f}x")
        
        return self.results
    
    def run_full_benchmark(self) -> Dict[str, Any]:
        """Run all benchmarks and return comprehensive results."""
        print("Starting comprehensive benchmark...")
        
        # Run all benchmark types
        self.benchmark_forward_pass()
        self.benchmark_inference()
        self.benchmark_training_step()
        
        # Analyze results
        analysis = self._analyze_results()
        
        # Save results
        self._save_results(analysis)
        
        return analysis
    
    def _analyze_results(self) -> Dict[str, Any]:
        """Analyze benchmark results and generate summary statistics."""
        analysis = {
            "summary": {},
            "detailed_results": [],
            "recommendations": []
        }
        
        # Group results by operation
        operations = set(result.operation for result in self.results)
        
        for operation in operations:
            op_results = [r for r in self.results if r.operation == operation]
            float_results = [r for r in op_results if r.model_type == "float"]
            int_results = [r for r in op_results if r.model_type == "int"]
            
            if float_results and int_results:
                # Calculate averages
                float_avg_duration = np.mean([r.duration_ms for r in float_results])
                int_avg_duration = np.mean([r.duration_ms for r in int_results])
                
                speedup = float_avg_duration / int_avg_duration
                
                analysis["summary"][operation] = {
                    "float_avg_duration_ms": float_avg_duration,
                    "int_avg_duration_ms": int_avg_duration,
                    "speedup": speedup,
                    "int_faster": 1 if speedup > 1.0 else 0  # Use 1/0 instead of True/False
                }
                
                # Add recommendations
                if speedup > 1.1:
                    analysis["recommendations"].append(
                        f"Integer model is {speedup:.2f}x faster for {operation}"
                    )
                elif speedup < 0.9:
                    analysis["recommendations"].append(
                        f"Float model is {1/speedup:.2f}x faster for {operation}"
                    )
                else:
                    analysis["recommendations"].append(
                        f"Performance is similar for {operation} (speedup: {speedup:.2f}x)"
                    )
        
        # Add detailed results
        for result in self.results:
            analysis["detailed_results"].append({
                "model_type": result.model_type,
                "operation": result.operation,
                "batch_size": result.batch_size,
                "input_size": result.input_size,
                "duration_ms": result.duration_ms,
                "memory_usage_mb": result.memory_usage_mb,
                "throughput": result.throughput,
                "error_rate": result.error_rate
            })
        
        return analysis
    
    def _save_results(self, analysis: Dict[str, Any]) -> None:
        """Save benchmark results to file."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"benchmark_results_{timestamp}.json"
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, bool):
                return bool(obj)  # Ensure boolean values are properly handled
            return obj
        
        # Recursively convert numpy types
        def convert_recursive(obj):
            if isinstance(obj, dict):
                return {k: convert_recursive(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_recursive(v) for v in obj]
            else:
                return convert_numpy(obj)
        
        analysis_converted = convert_recursive(analysis)
        
        with open(filename, 'w') as f:
            json.dump(analysis_converted, f, indent=2)
        
        print(f"Benchmark results saved to {filename}")
    
    def print_summary(self) -> None:
        """Print a summary of benchmark results."""
        if not self.results:
            print("No benchmark results available.")
            return
        
        print("\n" + "="*60)
        print("BENCHMARK SUMMARY")
        print("="*60)
        
        # Group by operation
        operations = set(result.operation for result in self.results)
        
        for operation in operations:
            op_results = [r for r in self.results if r.operation == operation]
            float_results = [r for r in op_results if r.model_type == "float"]
            int_results = [r for r in op_results if r.model_type == "int"]
            
            if float_results and int_results:
                float_avg = np.mean([r.duration_ms for r in float_results])
                int_avg = np.mean([r.duration_ms for r in int_results])
                speedup = float_avg / int_avg
                
                print(f"\n{operation.upper().replace('_', ' ')}:")
                print(f"  Float model: {float_avg:.2f}ms average")
                print(f"  Int model:  {int_avg:.2f}ms average")
                print(f"  Speedup:    {speedup:.2f}x")
                print(f"  Winner:     {'Integer' if speedup > 1.0 else 'Float'}")
        
        print("\n" + "="*60)


def run_benchmark(device: str = "cpu", save_results: bool = True) -> Dict[str, Any]:
    """Convenience function to run a complete benchmark."""
    benchmarker = ModelBenchmarker(device=device)
    results = benchmarker.run_full_benchmark()
    
    if save_results:
        benchmarker.print_summary()
    
    return results


if __name__ == "__main__":
    # Run benchmark on CPU
    print("Running benchmark on CPU...")
    cpu_results = run_benchmark(device="cpu")
    
    # Run benchmark on GPU if available
    if torch.cuda.is_available():
        print("\nRunning benchmark on GPU...")
        gpu_results = run_benchmark(device="cuda")
    else:
        print("\nGPU not available, skipping GPU benchmark.") 