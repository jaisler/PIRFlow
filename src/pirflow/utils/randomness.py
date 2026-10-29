# SPDX-License-Identifier: MIT
"""Shared random seed for PIRFlow."""

import torch

RANDOM_SEED = 1234

def seed_random_generators():
    """Reset PyTorch random generators at the start of a run."""
    torch.manual_seed(RANDOM_SEED)
