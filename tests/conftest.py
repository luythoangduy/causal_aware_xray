from __future__ import annotations
import pytest
import torch
import warnings

# Configure pytest
def pytest_configure(config):
    """Configure pytest settings."""
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

@pytest.fixture(autouse=True)
def set_seed():
    """Set random seed for reproducible tests."""
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)