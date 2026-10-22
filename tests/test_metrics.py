# SPDX-License-Identifier: MIT
import math
import pytest
import torch

from pirflow.utils.metrics import compute_metrics

def test_compute_metrics_matches_known_values():
    # Arrange: a small example with hand-calculated results.
    y_true = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    y_pred = torch.tensor([1.0, 2.0, 4.0], dtype=torch.float64)

    # Act.
    metrics = compute_metrics(y_pred=y_pred, y_true=y_true)

    # Assert: expected values come from the calculations above.
    expected = {
        "mse": 1.0 / 3.0,
        "rmse": math.sqrt(1.0 / 3.0),
        "rel_l2": 1.0 / math.sqrt(14.0),
        "r2": 0.5,
    }

    assert metrics == pytest.approx(
        expected,
        rel=1e-12,
        abs=1e-12,
    )
