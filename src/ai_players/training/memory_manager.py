"""Memory management for GPU/CPU optimization."""

import gc
from typing import Dict, Optional, Tuple

import torch


class MemoryManager:
    """Manages memory usage for training across different devices.

    Parameters
    ----------
    device : torch.device
        Device to manage memory for.
    """

    def __init__(self, device: torch.device) -> None:
        """Initialize memory manager.

        Parameters
        ----------
        device : torch.device
            Device to manage.
        """
        self.device = device
        self.is_gpu = device.type == "cuda"
        self.memory_config: Dict = {}

    def detect_device_capabilities(self) -> Dict:
        """Detect device capabilities and memory.

        Returns
        -------
        Dict
            Device capabilities dictionary.
        """
        if self.is_gpu:
            return self._detect_gpu_capabilities()
        else:
            return self._detect_cpu_capabilities()

    def _detect_gpu_capabilities(self) -> Dict:
        """Detect GPU capabilities.

        Returns
        -------
        Dict
            GPU capabilities.
        """
        if not torch.cuda.is_available():
            return {"available": False}

        props = torch.cuda.get_device_properties(self.device)
        total_memory = props.total_memory / (1024**3)  # GB
        allocated = torch.cuda.memory_allocated(self.device) / (1024**3)
        reserved = torch.cuda.memory_reserved(self.device) / (1024**3)
        free = total_memory - reserved

        # Detect GPU type based on memory
        gpu_type = "unknown"
        if total_memory <= 12.5:  # ~12GB
            gpu_type = "RTX_3060"
        elif total_memory <= 25:  # ~24GB
            gpu_type = "P40"

        capabilities = {
            "available": True,
            "device_name": props.name,
            "total_memory_gb": total_memory,
            "free_memory_gb": free,
            "allocated_memory_gb": allocated,
            "reserved_memory_gb": reserved,
            "gpu_type": gpu_type,
            "compute_capability": f"{props.major}.{props.minor}",
            "supports_fp16": props.major >= 7,  # Volta and newer
            "supports_bf16": props.major >= 8,  # Ampere and newer
        }

        return capabilities

    def _detect_cpu_capabilities(self) -> Dict:
        """Detect CPU capabilities.

        Returns
        -------
        Dict
            CPU capabilities.
        """
        import psutil

        memory = psutil.virtual_memory()
        capabilities = {
            "available": True,
            "device_name": "CPU",
            "total_memory_gb": memory.total / (1024**3),
            "free_memory_gb": memory.available / (1024**3),
            "cpu_count": psutil.cpu_count(logical=False),
            "cpu_count_logical": psutil.cpu_count(logical=True),
        }

        return capabilities

    def get_optimal_batch_size(
        self, model_size_mb: float, feature_size_mb: float, target_memory_usage: float = 0.8
    ) -> int:
        """Calculate optimal batch size based on available memory.

        Parameters
        ----------
        model_size_mb : float
            Model size in MB.
        feature_size_mb : float
            Feature size per sample in MB.
        target_memory_usage : float
            Target memory usage fraction (0.0-1.0).

        Returns
        -------
        int
            Optimal batch size.
        """
        caps = self.detect_device_capabilities()

        if self.is_gpu:
            available_memory_gb = caps.get("free_memory_gb", 0)
            available_memory_mb = available_memory_gb * 1024
        else:
            available_memory_gb = caps.get("free_memory_gb", 0)
            available_memory_mb = available_memory_gb * 1024

        # Reserve memory for model and overhead
        reserved_mb = model_size_mb * 1.5  # Model + gradients + optimizer states
        usable_memory_mb = (available_memory_mb * target_memory_usage) - reserved_mb

        if usable_memory_mb <= 0:
            return 1

        # Calculate batch size (account for forward + backward pass)
        # Backward pass typically uses 2-3x forward pass memory
        memory_per_sample_mb = feature_size_mb * 3.5  # Conservative estimate
        batch_size = int(usable_memory_mb / memory_per_sample_mb)

        return max(1, batch_size)

    def get_training_config(self, model_parameter_count: int) -> Dict:
        """Get optimal training configuration for device.

        Parameters
        ----------
        model_parameter_count : int
            Number of model parameters.

        Returns
        -------
        Dict
            Training configuration.
        """
        caps = self.detect_device_capabilities()

        if self.is_gpu:
            return self._get_gpu_config(caps, model_parameter_count)
        else:
            return self._get_cpu_config(caps, model_parameter_count)

    def _get_gpu_config(self, caps: Dict, param_count: int) -> Dict:
        """Get GPU training configuration.

        Parameters
        ----------
        caps : Dict
            GPU capabilities.
        param_count : int
            Model parameter count.

        Returns
        -------
        Dict
            Training configuration.
        """
        gpu_type = caps.get("gpu_type", "unknown")
        total_memory = caps.get("total_memory_gb", 0)

        config = {
            "use_mixed_precision": True,
            "precision": "fp16" if caps.get("supports_fp16", False) else "fp32",
            "gradient_checkpointing": True,
            "gradient_accumulation_steps": 1,
        }

        if gpu_type == "RTX_3060" or total_memory <= 12.5:
            # RTX 3060 configuration
            config.update(
                {
                    "batch_size": 32,
                    "max_batch_size": 64,
                    "gradient_accumulation_steps": 2,
                    "precision": "fp16",
                    "gradient_checkpointing": True,
                    "data_prefetch": 2,
                }
            )
        elif gpu_type == "P40" or total_memory <= 25:
            # P40 configuration
            config.update(
                {
                    "batch_size": 128,
                    "max_batch_size": 256,
                    "gradient_accumulation_steps": 1,
                    "precision": "fp16" if caps.get("supports_fp16", False) else "fp32",
                    "gradient_checkpointing": False,
                    "data_prefetch": 4,
                }
            )
        else:
            # Default GPU configuration
            config.update(
                {
                    "batch_size": 64,
                    "max_batch_size": 128,
                    "gradient_accumulation_steps": 1,
                    "precision": "fp16" if caps.get("supports_fp16", False) else "fp32",
                    "gradient_checkpointing": True,
                    "data_prefetch": 2,
                }
            )

        return config

    def _get_cpu_config(self, caps: Dict, param_count: int) -> Dict:
        """Get CPU training configuration.

        Parameters
        ----------
        caps : Dict
            CPU capabilities.
        param_count : int
            Model parameter count.

        Returns
        -------
        Dict
            Training configuration.
        """
        total_memory = caps.get("total_memory_gb", 0)
        cpu_count = caps.get("cpu_count", 4)

        config = {
            "use_mixed_precision": False,
            "precision": "fp32",
            "gradient_checkpointing": False,
            "gradient_accumulation_steps": 1,
            "batch_size": 16,
            "max_batch_size": 32,
            "num_workers": min(cpu_count, 8),
            "data_prefetch": 2,
            "use_ram_cache": total_memory >= 32,  # Use RAM cache if 32GB+ available
        }

        if total_memory >= 32:
            config["batch_size"] = 32
            config["max_batch_size"] = 64
            config["num_workers"] = min(cpu_count, 8)

        return config

    def clear_cache(self) -> None:
        """Clear memory cache."""
        if self.is_gpu:
            torch.cuda.empty_cache()
        gc.collect()

    def get_memory_stats(self) -> Dict:
        """Get current memory statistics.

        Returns
        -------
        Dict
            Memory statistics.
        """
        if self.is_gpu:
            return {
                "allocated_gb": torch.cuda.memory_allocated(self.device) / (1024**3),
                "reserved_gb": torch.cuda.memory_reserved(self.device) / (1024**3),
                "max_allocated_gb": torch.cuda.max_memory_allocated(self.device) / (1024**3),
            }
        else:
            import psutil

            memory = psutil.virtual_memory()
            return {
                "used_gb": memory.used / (1024**3),
                "available_gb": memory.available / (1024**3),
                "percent": memory.percent,
            }

    def optimize_for_training(self, model: torch.nn.Module) -> None:
        """Optimize model and memory for training.

        Parameters
        ----------
        model : torch.nn.Module
            Model to optimize.
        """
        if self.is_gpu:
            # Enable cuDNN benchmarking for consistent input sizes
            torch.backends.cudnn.benchmark = True
            # Enable deterministic mode if needed (slower but reproducible)
            # torch.backends.cudnn.deterministic = True

        # Clear cache before training
        self.clear_cache()

