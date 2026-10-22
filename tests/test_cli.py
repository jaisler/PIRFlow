# SPDX-License-Identifier: MIT
import sys
import pytest
from unittest.mock import patch

from pirflow.cli import main

def test_existing_config_is_passed_to_run(tmp_path, monkeypatch):
    # Create a temporary configuration file.
    config_path = tmp_path / "config.yaml"
    config_path.write_text("{}\n", encoding="utf-8")

    # Simulate running: pirflow --config config.yaml
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys, "argv", ["pirflow", "--config", "config.yaml"]
    )

    # Replace run() so the actual workflow does not execute.
    with patch("pirflow.cli.run") as mock_run:
        main()

    mock_run.assert_called_once_with(
        config_path=config_path.resolve()
    )