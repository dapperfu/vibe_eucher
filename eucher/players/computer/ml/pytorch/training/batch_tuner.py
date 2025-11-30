"""Batch size tuning utility for GPU memory optimization."""

import gc
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


def test_batch_size(
    model: nn.Module, dataset: Dataset, batch_size: int, device: torch.device
) -> bool:
    """Test if a batch size works without OOM error.

    Parameters
    ----------
    model : nn.Module
        Model to test.
    dataset : Dataset
        Dataset to use.
    batch_size : int
        Batch size to test.
    device : torch.device
        Device to test on.

    Returns
    -------
    bool
        True if batch size works, False if OOM error occurs.
    """
    if len(dataset) == 0:
        return False

    try:
        # Create data loader with test batch size
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        # Get a sample batch
        batch = next(iter(data_loader))

        # Move batch to device
        batch_device = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch_device[key] = value.to(device)
            else:
                batch_device[key] = value

        # Set model to training mode BEFORE forward pass to ensure gradients
        model.train()
        
        # Forward pass WITH gradients (no torch.no_grad())
        outputs = model(
            hand=batch_device.get("hand"),
            trick_history=batch_device.get("trick_history"),
            won_tricks_summary=batch_device.get("won_tricks_summary"),
            game_context=batch_device.get("game_context"),
            current_trick=batch_device.get("current_trick"),
        )

        # Compute loss and backward pass
        criterion = nn.CrossEntropyLoss()
        
        # Get target - should be 1D (batch_size,) after DataLoader batching scalars
        target = batch_device.get("target")
        if target is None:
            target = torch.zeros(batch_size, dtype=torch.long, device=device)
        else:
            # DataLoader batches scalar (0D) tensors to 1D (batch_size,)
            # But handle edge cases where shape might be wrong
            if target.dim() == 0:
                # Single scalar - shouldn't happen after batching, but handle it
                target = target.unsqueeze(0).expand(batch_size)
            elif target.dim() == 1:
                # Correct shape (batch_size,) - ensure length matches
                if target.shape[0] != batch_size:
                    if target.shape[0] > batch_size:
                        target = target[:batch_size]
                    else:
                        # Pad or repeat if too short (shouldn't happen)
                        padding = torch.zeros(batch_size - target.shape[0], dtype=torch.long, device=device)
                        target = torch.cat([target, padding])
            elif target.dim() == 2:
                # 2D (batch_size, 1) - squeeze to 1D (legacy format)
                target = target.squeeze(-1)
            # Ensure correct dtype
            if target.dtype != torch.long:
                target = target.long()
        
        # Use card_play output (most common action type)
        output = outputs["card_play"]
        # Ensure output and target batch sizes match
        actual_batch_size = min(output.shape[0], target.shape[0])
        if actual_batch_size < batch_size:
            # Adjust if batch was smaller than expected
            output = output[:actual_batch_size]
            target = target[:actual_batch_size]
        
        # CRITICAL: Clamp target values to valid range [0, num_classes-1]
        # This prevents CUDA device-side assert errors from out-of-bounds indices
        num_classes = output.shape[-1]
        target = torch.clamp(target, 0, num_classes - 1)
        
        # Validate targets are within range (additional safety check)
        if torch.any(target < 0) or torch.any(target >= num_classes):
            # This should not happen after clamping, but log if it does
            target = torch.clamp(target, 0, num_classes - 1)
        
        # Compute loss and backward pass
        loss = criterion(output, target)
        loss.backward()

        # Clean up
        del batch, batch_device, outputs, loss
        torch.cuda.empty_cache() if device.type == "cuda" else None
        gc.collect()

        return True

    except RuntimeError as e:
        error_str = str(e).lower()
        # Check if it's an OOM error
        if "out of memory" in error_str or ("cuda" in error_str and "out of memory" in error_str):
            # Clean up
            try:
                torch.cuda.empty_cache() if device.type == "cuda" else None
            except Exception:
                pass  # Ignore cleanup errors
            gc.collect()
            return False
        # Check for CUDA assert errors (device-side assert)
        if "cuda" in error_str and ("assert" in error_str or "device-side" in error_str):
            # This usually means invalid target values or other data issues
            # Clean up and return False (batch size test failed)
            try:
                torch.cuda.empty_cache() if device.type == "cuda" else None
            except Exception:
                pass  # Ignore cleanup errors
            gc.collect()
            return False
        # Other runtime errors should be raised
        raise
    except Exception as e:
        # Any other error means batch size doesn't work
        try:
            torch.cuda.empty_cache() if device.type == "cuda" else None
        except Exception:
            pass  # Ignore cleanup errors
        gc.collect()
        return False


def tune_batch_size(
    model: nn.Module,
    dataset: Dataset,
    device: torch.device,
    start_size: int = 8,
    max_size: int = 1024,
    use_binary_search: bool = True,
) -> int:
    """Find optimal batch size that fits in GPU memory.

    Parameters
    ----------
    model : nn.Module
        Model to tune for.
    dataset : Dataset
        Dataset to use.
    device : torch.device
        Device to tune on.
    start_size : int
        Starting batch size (default: 8).
    max_size : int
        Maximum batch size to try (default: 1024).
    use_binary_search : bool
        If True, use binary search after finding upper bound (faster).

    Returns
    -------
    int
        Optimal batch size.
    """
    if len(dataset) == 0:
        return start_size

    # Exponential search to find upper bound
    current_size = start_size
    last_working_size = start_size

    print(f"Tuning batch size (starting at {start_size})...")

    # First, try exponential growth
    while current_size <= max_size:
        print(f"  Testing batch size: {current_size}", end=" ... ")
        if test_batch_size(model, dataset, current_size, device):
            print("✓ OK")
            last_working_size = current_size
            current_size *= 2
        else:
            print("✗ OOM")
            break

    # If we found an upper bound and binary search is enabled, refine it
    if use_binary_search and current_size > last_working_size * 2:
        # Binary search between last_working_size and current_size
        low = last_working_size
        high = min(current_size, max_size)

        print(f"  Binary search between {low} and {high}...")
        while low < high - 1:
            mid = (low + high) // 2
            print(f"  Testing batch size: {mid}", end=" ... ")
            if test_batch_size(model, dataset, mid, device):
                print("✓ OK")
                last_working_size = mid
                low = mid
            else:
                print("✗ OOM")
                high = mid

    optimal_size = last_working_size
    print(f"Optimal batch size: {optimal_size}")
    return optimal_size

