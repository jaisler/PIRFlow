<!-- SPDX-License-Identifier: MIT -->

# PIRFlow

PIRFlow (Physics-Informed Reconstruction of Flow fields) is a PyTorch 
framework for physics-informed reconstruction of compressible flow fields 
from sparse data.

## Overview

PIRFlow combines neural networks with the steady compressible Euler or
Reynolds-averaged Navier–Stokes (RANS) equations to reconstruct density,
velocity, pressure, and, for RANS configurations, turbulent viscosity.

The framework supports two workflows:

- **Forward reconstruction:** learn flow fields from sampled CFD data using
  multilayer perceptrons (MLPs) or graph neural networks (GNNs), with optional
  physics-informed constraints.
- **Inverse reconstruction:** infer flow fields from heterogeneous
  observations—including synthetic Schlieren data, velocity profiles, and
  pressure taps using governing equation residuals and optional
  boundary condition data. This workflow currently supports **MLPs only**.

PIRFlow targets compressible flows featuring shocks, boundary layers,
shear layers, and recirculation regions. Its modular structure supports
experimentation with network architectures, observation types, sampling
strategies, and loss formulations.

## Features

- **Network architectures:** MLPs and GNNs for reconstruction on computational
  flow domains.
- **Governing equations:** steady compressible Euler and RANS formulations
  with effective viscosity.
- **Training:** supervised and physics-informed learning, with
  governing equation residuals evaluated at collocation points.
- **Inverse observations:** density gradient, velocity, and pressure losses
  with configurable weights and optional boundary constraints.
- **Synthetic Schlieren:** configurable image resolution, supersampling,
  blur, noise, and uniform or signal-based sampling.
- **Data preparation:** sampling utilities for CFD data, observations,
  collocation points, and selected boundaries.
- **Scaling and constraints:** nondimensionalization, input normalization,
  and positivity constraints for density, pressure, and turbulent viscosity.
- **Optimization:** Adam and L-BFGS, with CFD and observation validation
  and test metrics.
- **Post-processing:** full-mesh prediction, nondimensional mesh-based
  metrics, VTK export, and PyVista visualization of reconstructed fields
  and errors.
- **Modular implementation:** separate components for networks, physical
  residuals, losses, sampling, evaluation, and post-processing.

## Installation

Run the following commands from the repository root, where
`pyproject.toml` is located.

### Create a virtual environment

On macOS or Linux:

```bash
python3 -m venv .venv
```

and activate it:
```
source .venv/bin/activate
```

### Install PIRFlow

With your virtual environment activated, install PIRFlow 
and its declared Python dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install .
```

For development, install the project in editable mode with its 
test dependencies:

```bash
python -m pip install -e ".[test]"
```

An editable installation makes changes to the Python source code
available on subsequent runs without reinstalling. Reinstall after
changing dependencies or other package metadata.

Installation makes the pirflow command available in that environment.

### Running PIRFlow

Run PIRFlow with a YAML configuration file:

```bash
pirflow --config config/configuration.yaml
```

The --config argument is required and specifies the configuration 
file for the reconstruction worflow. Replace the example path with 
the configuration you want to use.

To display the available command-line options:

```bash
python main.py --help
```

Before launching a case, update the YAML configuration with the
appropriate input files, output directories, physical parameters,
network settings, and training options.

## Testing

After installing the development dependencies, run the automated tests
from the repository root with your virtual environment activated:

```bash
python -m pytest -v
```

To run an individual test file:

```bash
python -m pytest tests/test_metrics.py -v
```

## Documentation

See [CONTRIBUTING.md](CONTRIBUTING.md) for code contribution standards.

## License

PIRFlow is released under the [MIT License](LICENSE).
