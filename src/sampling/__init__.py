# SPDX-License-Identifier: MIT
"""Sample CFD and collocation points and prepare their datasets."""

# src/sampling/__init__.py
from .data import get_data_points, get_collocation_points
from .sampling import SamplingData
from .splitting import prepare_cfd_datasets, prepare_collocation_dataset

__all__ = [
        "SamplingData",
        "prepare_cfd_datasets",
        "prepare_collocation_dataset",
        "get_data_points",
        "get_collocation_points",
]
