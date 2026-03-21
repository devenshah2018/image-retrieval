"""Pick PyTorch compute device: CUDA (NVIDIA) → MPS (Apple Silicon) → CPU."""
from __future__ import annotations

import torch


def torch_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def torch_device_str() -> str:
    return str(torch_device())
