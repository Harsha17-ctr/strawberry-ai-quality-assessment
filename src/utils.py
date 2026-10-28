"""
Utility functions for Strawberry AI project:
- Configuration management
- Reproducible random seed setting
- Device selection
- Logging & plotting helpers
"""

import os
import random
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt


def load_config(config_path: str = "config.yaml") -> dict:
    """Load and parse YAML configuration file."""
    if not os.path.exists(config_path):
        # Check parent directories if called from scripts or tests
        alt_path = os.path.join("..", config_path)
        if os.path.exists(alt_path):
            config_path = alt_path
        else:
            raise FileNotFoundError(f"Configuration file not found at {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed: int = 42) -> None:
    """Set random seed across Python, NumPy, and PyTorch for deterministic execution."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(prefer_cuda: bool = True) -> torch.device:
    """Detect available hardware and return torch.device."""
    if prefer_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device


def ensure_dir(dir_path: str) -> str:
    """Ensure directory exists and return absolute or normalized path."""
    os.makedirs(dir_path, exist_ok=True)
    return dir_path
