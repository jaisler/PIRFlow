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
source .venv/bin/activate
```

### Install PIRFlow

Install PIRFlow and its declared Python dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install .
```

For development, use an editable installation instead:

```bash
python -m pip install -e .
```

An editable installation makes changes to the Python source code
available on subsequent runs without reinstalling. Reinstall after
changing dependencies or other package metadata.

### Running PIRFlow

With your virtual environment active, run the launcher from the
repository root and provide a configuration file:

```bash
python main.py --config config/configuration.yaml
```

The selected file must exist and contain the settings for the intended
case.

Display the available command-line options with:

```bash
python main.py --help
```

### Configuration

Before launching a case, update the YAML configuration with the
appropriate input files, output directories, physical parameters,
network settings, and training options.

## Documentation

See [CONTRIBUTING.md](CONTRIBUTING.md) for code contribution standards.

## License

PIRFlow is released under the [MIT License](LICENSE).
