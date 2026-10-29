# SPDX-License-Identifier: MIT
"""Command-line interface for running PIRFlow from a YAML configuration."""

import argparse

from pathlib import Path
from pirflow.runner import run

def main():
    """Parse command-line arguments and launch the configured workflow.

    The required ``--config`` option specifies the YAML configuration file.
    Its path is expanded and resolved before being passed to ``run``.

    Returns
    -------
    None
        Executes the workflow using the selected configuration.
    """
    parser = argparse.ArgumentParser(description="Run PIRFlow.")
    
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the YAML configuration file.",
    )
    args = parser.parse_args()

    config_path = args.config.expanduser().resolve(strict=True)
    run(config_path=config_path)
