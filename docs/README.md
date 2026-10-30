# PIRFlow User Guide

Version 0.1.0

PIRFlow (Physics-Informed Reconstruction of Flow fields) reconstructs steady,
two-dimensional compressible flow fields using PyTorch. Forward runs learn
from sampled CFD fields and inverse runs learn from observations and optional
boundary data. Physics-informed runs also minimize Euler or Reynolds-averaged
Navier–Stokes (RANS) equation residuals at collocation points.

## Installation and quick start

PIRFlow requires Python 3.12. On macOS or Linux, run these commands from the
repository root, where [pyproject.toml](../pyproject.toml) is located. To create
the environment:

```bash
python3.12 -m venv .venv
```

To install the package and its dependencies, activate the virtual environment
and run the following commands:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
```

Copy the complete example [configuration](../configs/configuration.yaml):

```bash
cp configs/configuration.yaml configs/local.yaml
```

Edit `configs/local.yaml` for your data, physical parameters, network,
and training settings using the following sections. The supplied configuration
describes an inverse RANS case and needs your own input files; datasets are not
bundled with the project. Launch the workflow from the repository root:

```bash
pirflow --config configs/local.yaml
```

The `--config` argument is required. Use `pirflow --help` to inspect
command-line options. The workflow prepares samples, splits training,
validation, and test data, builds the network, trains, evaluates held-out
data, and optionally post-processes the full CFD mesh.

## Input files and paths

### Directories and filenames

The `paths` section contains directories. The `files` section selects the CFD
file, Gmsh geometry, and checkpoint basename. `paths.geometry` selects the
Gmsh geometry directory and `files.geometry` its filename. Replace these
sections in your local copy with one entry per key. For example, with commands
run from the repository root:

```yaml
paths:
  # In
  flow: /path/to/case/cfd
  geometry: /path/to/case/geometry
  observations: /path/to/case/observations
  # Out
  results: results/my_case
  samples: samples/my_case
  model: model/my_case

files:
  flowfield: flow.vtu
  geometry: domain.geo
  model_name: reconstruction
```

This reads `/path/to/case/cfd/flow.vtu` and, when boundary sampling is
requested, `/path/to/case/geometry/domain.geo`. PIRFlow accepts VTK files for
the flow field and Gmsh files for the geometry. A saved checkpoint becomes
`model/my_case/reconstruction.pth`.

Relative paths are resolved from the directory where you launch PIRFlow,
even when the YAML file lives in `configs/`. Absolute paths are also accepted.
PIRFlow creates the `results`, `samples`, and `model` directories if they do
not exist. Input files must already exist. Separate output directories by case
to keep saved samples and checkpoints associated with the correct geometry
and settings.

### CFD fields, coordinates, and units

Use a VTK dataset such as `flow.vtu` containing the computational mesh and
flow solution. The current workflow reconstructs the two velocity components
in the x-y plane. Configure the solver's array names and components once under
the top-level `cfd` section. Sampling, synthetic Schlieren observations, and
post-processing share this mapping. The table below shows example CFD array
names, corresponding model variables, and physical units. Keep units
consistent with your setup and physical reference scales.

```yaml
cfd:
  # Order: rho, p, u, v, mut (omit mut for Euler)
  fields: [Density, Pressure, Velocity, Velocity, Eddy_Viscosity]
  components: [0, 0, 0, 1, 0]
```

Use component `0` for scalar arrays. Velocity components can also come from
separate scalar arrays by configuring their names and using component `0`.

| CFD array | Model variable | Content | Unit |
| --- | --- | --- | --- |
| `Density` | `rho` | Scalar density | kg/m³ |
| `Velocity` | `u`, `v` | Vector; components 0 and 1 | m/s |
| `Pressure` | `p` | Scalar static pressure | Pa |
| `Eddy_Viscosity` | `mut` | Scalar dynamic turbulent viscosity | Pa·s |

Forward Euler sampling requires the first three arrays. Forward RANS sampling
also requires `Eddy_Viscosity`. Inverse boundary sampling reads the fields
selected for each boundary.

Full-mesh post-processing reads reference arrays from VTK point data. Provide
point-associated fields for that stage, converting cell data beforehand if
necessary. Use the same coordinate system and units for CFD, Gmsh geometry,
and CSV observations. PIRFlow applies the physical reference scales internally;
do not pre-normalize the input fields. `Eddy_Viscosity` must contain dynamic
turbulent viscosity.

### Inputs required by each stage

- **Interior sampling:** reads the CFD mesh to locate the domain, including for
  inverse PINN collocation points. Gradient sampling uses the density-gradient
  magnitude from the first array in `cfd.fields`.
- **Boundary sampling:** reads the Gmsh geometry when a boundary count is
  positive. Its physical group names must match `sampling.boundaries.names`
  exactly, and its geometry must align with the CFD mesh.
- **Inverse observations:** velocity and pressure CSVs are read from
  `paths.observations` only when their modalities are enabled. Synthetic
  Schlieren uses the CFD density field; its array name is set by
  the density entry in `cfd.fields`. See the inverse
  observations section for filenames and CSV columns.
- **Saved samples:** `run.routines.sampling: false` loads the relevant NPZ
  files from `paths.samples`. It does not disable observation loading,
  synthetic Schlieren generation, or full-mesh post-processing; those stages
  still require their original inputs when enabled.

## Configuration essentials

Start from the complete `configs/configuration.yaml` and edit the existing
keys in your local copy as described above. The examples below show selected
sections. They are not complete runnable configurations.

### Choose the workflow and equations

`run.problem` chooses the source of training data. `run.model` controls whether
equation residuals are included in the losses, defining the network as a
physics-informed neural network (PINN) or a supervised model. The allowed
configurations are:

| Workflow | `run.problem` | `run.model` | `run.equation` | Architecture |
| --- | --- | --- | --- | --- |
| Fit sampled CFD fields | `forward` | `supervised` | `euler` or `rans` | `mlp` or `gnn` |
| Fit CFD with physical constraints | `forward` | `pinn` | `euler` or `rans` | `mlp` or `gnn` |
| Reconstruct from observations with physical constraints | `inverse` | `pinn` | `euler` or `rans` | `mlp` |

Choose `run.equation: euler` for `rho, u, v, p`, or `rans` to include
the turbulent-viscosity field `mut` and viscous/thermal terms. Set the top-level
`dimension` to `2`, since the current framework only supports two-dimensional
flows.

### Match the network to the problem

Set `network.architecture` to `mlp` or `gnn` as allowed above.
GNNs are not yet implemented for the inverse problem. The network's output
width is determined by the equation choice. An MLP takes two coordinates
and must have four outputs for Euler or five for RANS. For example,
`network.mlp.layers: [2, 32, 32, 32, 5]` defines three hidden layers of
width 32 for a RANS case. Set its activation and dropout using
`network.mlp.activation` and `network.mlp.dropout`, respectively.

For GNNs, configure `network.gnn` (latent width, neighbors, edge attributes,
and message-passing layers); the output width is selected from the equation.
Keep `network.gnn.attributes.node.boundary_marker: false`, since boundary
marker features are not implemented yet.

Set `run.device` to `cpu` for CPU execution, `cuda` for an available NVIDIA
GPU, or `mps` for GPU acceleration on compatible Macs. PIRFlow automatically
falls back to `cpu` when CUDA or MPS is requested but unavailable.

### Sample points and split data

Set `run.routines.sampling: true` on the first run. `sampling.data_points`
controls supervised CFD data points; `sampling.collocation_points` controls
locations where PINNs evaluate equation residuals. Supervised runs skip
collocation sampling. Inverse CFD data contain only the prescribed boundary
observations.

For example, edit the sampling entries as follows, retaining the existing
`gradient_sampling` settings if gradient points are requested:

```yaml
sampling:
  method: lhs
  boundaries:
    names: [inlet, bottom, outlet, top]
  data_points:
    interior: 1000
    gradient: 200
    boundary: [40, 100, 40, 100]
  collocation_points:
    interior: 2000
    gradient: 400
    boundary: [40, 100, 40, 100]
```

Boundary counts follow the order of `names`; both count lists must have that
same length. A zero count skips that boundary. Interior counts must be positive
for fresh forward-data and collocation sampling. `gradient: 0` disables extra
gradient points. `method` accepts `random` or `lhs` (Latin hypercube sampling).
Under `gradient_sampling`, `pool_factor` sets the candidate-pool multiplier,
and `alpha` controls preference for larger density-gradient magnitudes.
Domain filtering can reduce the number of retained points.

Set `dataset.p_training_data`, `p_validation_data`, and `p_test_data` to
nonnegative percentages summing to 100, for example 80, 10, and 10. Each dataset
being split needs at least three points; validation and test sets retain at
least one point. The top-level `seed` controls the shuffled split and selected
sampling/noise routines.

### Set physical reference scales

Use positive, case-specific values in `physics.reference`:

| Key | Meaning | Unit |
| --- | --- | --- |
| `Lref` | Reference length | m |
| `U_0` | Reference velocity | m/s |
| `rho` | Reference density | kg/m³ |
| `mu` | Reference molecular dynamic viscosity for RANS | Pa·s |

To nondimensionalize, PIRFlow divides coordinates by `Lref`, velocities by
`U_0`, density by `rho`, pressure by `rho * U_0**2`, and turbulent viscosity
by `mu`. Keep these scales consistent when reloading a checkpoint.

Set `physics.gas.gamma` for both Euler and RANS. RANS also uses the specific
gas constant `physics.gas.R`, the Prandtl number `physics.turbulence.Pr`,
the turbulent Prandtl number `physics.turbulence.Prt`, and the Sutherland
parameters `T0`, `mu0`, and `S` under `physics.sutherland` for
temperature-dependent molecular viscosity.

### Control training and post-processing

Enable the desired optimizers under `optimizer.adam` and `optimizer.lbfgs`.
Each enabled optimizer requires positive `iterations` and `learning_rate`;
Adam runs before L-BFGS when both are enabled. Adam's scheduler multiplies its
learning rate by `gamma` every `step_size` steps. For L-BFGS, `iterations`
counts outer steps and `max_iter_per_step` limits internal iterations per step.
To skip training, set both optimizers' `enabled` flags to `false`.

`loss.weights.data` weights CFD and boundary fields.
`loss.weights.observations` weights Schlieren, velocity, and pressure
measurements from inverse problems. `loss.weights.residual` weights mass
(`f1`), x-momentum (`f2`), y-momentum (`f3`), and energy (`f4`).
`loss.print_frequency` is a positive integer controlling how often training
losses are printed.

Set `run.checkpoint.save_model: true` to save after training, or
`run.checkpoint.load_model: true` to load the configured checkpoint before
training/evaluation. Set `run.routines.postprocessing: true` for mesh metrics
and VTK export. Keep the `post_processing` lists `latex`, `colormaps`, and
`scales` aligned with `cfd.fields` and `cfd.components` in the order
`rho, p, u, v` (and `mut` for RANS). For Euler, remove the final
turbulent-viscosity entry from every aligned list.

For older configurations, move `post_processing.fields` and
`post_processing.components` into `cfd`, and remove
`identification.observations.schlieren.density_name` and
`sampling.gradient_sampling.variable`. Both Schlieren generation and gradient
sampling use the configured density array.

## Inverse observations

Enable or disable each modality under `identification.observations`:

- **Schlieren:** generated from the CFD density field. Select `grad_x`,
  `grad_y`, or `magnitude` and configure image resolution, blur, noise, and
  sampling. This loader generates synthetic observations from CFD data.
- **Velocity profiles:** with `filename: velocity_profile` and `n_files: 2`,
  provide these four files:

  ```text
  velocity_profile_u_0.csv
  velocity_profile_u_1.csv
  velocity_profile_v_0.csv
  velocity_profile_v_1.csv
  ```

  Each file needs headers `x,y,u` or `x,y,v` for its component.
  Indices start at zero.
- **Pressure taps:** with `filename: pressure_taps`, provide
  `pressure_taps.csv` with headers `x,y,p`.

Optional `identification.boundary_conditions` selects named boundaries and
observed fields (`rho`, `u`, `v`, `p`, `mut`) to sample from CFD. Those boundary
names must also appear in `sampling.boundaries.names`. An inverse run needs
nonempty training observations or boundary data. Collocation sampling
still uses the CFD mesh to define the domain.

## Outputs and saved models

| Location | Contents |
| --- | --- |
| `paths.samples` | `data_points.npz` for forward data, `boundary_points.npz` for inverse boundary data, and `collocation_points.npz` for PINN runs, as applicable. |
| `paths.model` | `<files.model_name>.pth` when training with `run.checkpoint.save_model: true`. |
| `paths.results` | Plots and `flowfield_comparison.vtk` when full-mesh post-processing is enabled. The VTK file includes reference, predicted, and error fields. |
| Terminal | Training progress and evaluation metrics: MSE, RMSE, relative L2 error, and R². Mesh metrics use nondimensional fields. |

Relative L2 is reported as `NaN` when reference RMS is at or below the metric
threshold. R² is `NaN` for fewer than two values or reference standard
deviation at or below that threshold.

To evaluate a saved model without retraining, keep its configuration, input
data, and saved samples, then set:

- `run.routines.sampling: false`
- `run.checkpoint.load_model: true`
- `optimizer.adam.enabled: false`
- `optimizer.lbfgs.enabled: false`
- `run.routines.postprocessing: true`

Use the same architecture, physical scales, sample data, and model filename.
Inverse observation inputs are loaded again. To skip PyVista field plots while
still exporting VTK, add `plot: false` under `post_processing`.

## Development and tests

For an editable installation with test dependencies, run these commands from
the repository root with the virtual environment activated:

```bash
python -m pip install -e ".[test]"
python -m pytest -v
```

To run an individual test file:

```bash
python -m pytest tests/test_metrics.py -v
```

See [CONTRIBUTING.md](../CONTRIBUTING.md) in the repository for contribution
standards and scientific validation guidance. PIRFlow is released under the
MIT License; see [LICENSE](../LICENSE) for its terms.

## Compile the LaTeX guide

A standalone LaTeX version is available in [user-guide.tex](user-guide.tex).
With a TeX distribution providing `pdflatex` and `latexmk`, run from the
repository root:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=docs/build docs/user-guide.tex
```

The PDF is written to `docs/build/user-guide.pdf`. Build files are ignored by Git.
The LaTeX source can also be uploaded to Overleaf as the main document.
