# SPDX-License-Identifier: MIT
import argparse
from pathlib import Path

from pirflow.runner import run

def main():
    
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
