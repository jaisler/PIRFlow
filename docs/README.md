# PIRFlow User Guide

PIRFlow reconstructs steady, two-dimensional compressible flow fields using
PyTorch. Forward runs learn from sampled CFD fields; inverse runs learn from
observations and optional boundary data. Physics-informed runs also minimize
Euler or RANS equation residuals at collocation points.

## Quick start

1. Follow the [installation instructions](../README.md#installation) using
   Python 3.12.
2. Copy the complete [configuration](../configs/configuration.yaml):

   ```bash
   cp configs/configuration.yaml configs/local.yaml
   ```

3. Edit `configs/local.yaml` for your data and run settings as described below.
   The supplied configuration describes an inverse RANS case and needs your
   own input files; datasets are not bundled with the project.
4. Run from the repository root:

   ```bash
   pirflow --config configs/local.yaml
   ```

The workflow prepares samples, splits training/validation/test data, builds
the network, trains, evaluates held-out data, and optionally post-processes
the full CFD mesh. Use `pirflow --help` to inspect command-line options.

## Input files and paths

### Directories and filenames

The `paths` section contains directories; `files` selects the CFD file, Gmsh
geometry, and checkpoint basename. Replace these sections in your local copy
with one entry per key. For example, with commands run from the repository
root:

```yaml
paths:
  flow: /path/to/case/cfd
  mesh: /path/to/case/geometry
  observations: /path/to/case/observations
  results: results/my_case
  samples: samples/my_case
  model: model/my_case

files:
  flowfield: flow.vtu
  mesh: domain.geo
  model_name: reconstruction

name: my_case
```

This reads `/path/to/case/cfd/flow.vtu` and, when boundary sampling is
requested, `/path/to/case/geometry/domain.geo`. A saved checkpoint becomes
`model/my_case/reconstruction.pth`; `name` supplies the comparison-file prefix,
giving `results/my_case/my_case_comparison.vtk` during post-processing.

Relative paths are resolved from the directory where you launch PIRFlow,
even when the YAML file lives in `configs/`. Absolute paths are also accepted.
PIRFlow creates the `results`, `samples`, and `model` directories; input files
must already exist. Separate output directories by case to keep saved samples
and checkpoints associated with the correct geometry and settings. Keep local
paths and datasets out of commits.

### CFD fields, coordinates, and units

Use a VTK dataset such as `flow.vtu` containing the computational mesh and
flow solution. The current workflow reconstructs the two velocity components
in the x-y plane. Supply dimensional coordinates in metres and dimensional
fields with these case-sensitive array names:

| CFD array | Model variable | Content | Unit |
| --- | --- | --- | --- |
| `Density` | `rho` | Scalar density | kg/m³ |
| `Velocity` | `u`, `v` | Vector; components 0 and 1 | m/s |
| `Pressure` | `p` | Scalar static pressure | Pa |
| `Eddy_Viscosity` | `mut` | Scalar dynamic turbulent viscosity | Pa·s |

Forward Euler sampling requires the first three arrays; forward RANS sampling
also requires `Eddy_Viscosity`. Inverse boundary sampling reads the fields
selected for each boundary. The CFD sampler currently uses these fixed array
names, so rename differently named arrays before sampling. Changing
`post_processing.fields` only changes the reference mapping for post-processing.

Full-mesh post-processing reads reference arrays from VTK point data. Provide
point-associated fields for that stage, converting cell data beforehand if
necessary. Use the same coordinate system and units for CFD, Gmsh geometry,
and CSV observations. PIRFlow applies the physical reference scales internally;
do not pre-normalize the input fields. `Eddy_Viscosity` must contain dynamic
turbulent viscosity, not kinematic viscosity or a viscosity ratio.

### Inputs required by each stage

- **Fresh sampling:** reads the CFD mesh to locate the domain, including for
  inverse PINN collocation points. Gradient sampling also needs the array named
  by `sampling.gradient_sampling.variable`.
- **Boundary sampling:** reads the Gmsh geometry when a boundary count is
  positive. Its physical group names must match `sampling.boundaries.names`
  exactly, and its geometry must align with the CFD mesh.
- **Inverse observations:** velocity and pressure CSVs are read from
  `paths.observations` only when their modalities are enabled. Synthetic
  Schlieren uses the CFD density field; its array name is set by
  `identification.observations.schlieren.density_name`. See the inverse
  observations section for filenames and CSV columns.
- **Saved samples:** `run.routines.sampling: false` loads the relevant NPZ
  files from `paths.samples`. It does not disable observation loading,
  synthetic Schlieren generation, or full-mesh post-processing; those stages
  still require their original inputs when enabled.

## Configuration essentials

Start from the complete `configs/configuration.yaml` and edit the existing
keys in your local copy. The examples below show selected sections; they are
not complete runnable configurations. Retain the other sections, since model
construction reads some settings even when the corresponding routine is off.

### Choose the workflow and equations

`run.problem` chooses the source of training data; `run.model` controls whether
equation residuals are included. The allowed configurations are:

| Workflow | `run.problem` | `run.model` | Architecture |
| --- | --- | --- | --- |
| Fit sampled CFD fields | `forward` | `supervised` | `mlp` or `gnn` |
| Fit CFD with physical constraints | `forward` | `pinn` | `mlp` or `gnn` |
| Reconstruct from observations with physical constraints | `inverse` | `pinn` | `mlp` |

Forward runs do not load inverse observation CSVs. Inverse runs require
nonempty training observations or enabled boundary data. Set `geometry.dimension`
to `2`. Choose `run.equation: euler` for `rho, u, v, p`, or `rans` to include
the turbulent-viscosity field `mut` and viscous/thermal terms.

### Match the network to the problem

Set `network.architecture` to `mlp` or `gnn` as allowed above. An MLP takes two
coordinates and must have four outputs for Euler or five for RANS. For example,
`network.mlp.layers: [2, 32, 32, 32, 5]` defines three hidden layers of width 32
for a RANS case. Set its activation and dropout in `network.mlp`.

For GNNs, configure `network.gnn` (latent width, neighbors, edge attributes,
and message-passing layers); the output width is selected from the equation.
Keep `network.gnn.attributes.node.boundary_marker: false`, since boundary
marker features are not implemented.

Use `run.device: cpu` for a CPU run or `cuda` for an available CUDA device.
Unavailable CUDA falls back to CPU. Other device strings are passed to PyTorch
without that availability fallback and require support for the operations used.

### Sample points and split data

Set `run.routines.sampling: true` on the first run. `sampling.data_points`
controls CFD observations; `sampling.collocation_points` controls locations
where PINNs evaluate equation residuals. Supervised runs skip collocation
sampling. Inverse CFD data contain only the selected boundary observations.

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
for fresh forward-data and collocation sampling; `gradient: 0` disables extra
gradient points. `method` accepts `random` or `lhs` (Latin hypercube sampling).
Under `gradient_sampling`, `variable` selects the CFD sensor, `pool_factor`
sets the candidate-pool multiplier, and `alpha` controls preference for larger
gradients. Domain filtering can reduce the number of retained points.

Set `dataset.p_training_data`, `p_validation_data`, and `p_test_data` to
nonnegative percentages summing to 100, for example 80, 10, and 10. Each dataset
being split needs at least three points; validation and test sets retain at
least one point. The top-level `seed` controls the shuffled split and selected
sampling/noise routines; it does not configure every random generator.

### Set physical reference scales

Use positive, case-specific values in `physics.reference`:

| Key | Meaning | Unit |
| --- | --- | --- |
| `Lref` | Reference length | m |
| `U_0` | Reference velocity | m/s |
| `rho` | Reference density | kg/m³ |
| `mu` | Reference molecular dynamic viscosity for RANS | Pa·s |

Coordinates are divided by `Lref`, velocities by `U_0`, density by `rho`,
pressure by `rho * U_0**2`, and turbulent viscosity by `mu`. The pressure scale
is derived from density and velocity. Keep these scales consistent when
reloading a checkpoint.

Set `physics.gas.gamma` for both equation choices. RANS also uses the specific
gas constant `physics.gas.R` in J/(kg·K), `physics.turbulence.Pr` and `Prt`,
and the Sutherland parameters `T0`, `mu0`, and `S` under `physics.sutherland`
for temperature-dependent molecular viscosity.

### Control training and post-processing

Enable the desired optimizers under `optimizer.adam` and `optimizer.lbfgs`.
Each enabled optimizer requires positive `iterations` and `learning_rate`;
Adam runs before L-BFGS when both are enabled. Adam's scheduler multiplies its
learning rate by `gamma` every `step_size` steps. For L-BFGS, `iterations`
counts outer steps and `max_iter_per_step` limits internal iterations per step.
To skip training, set both optimizers' `enabled` flags to `false`.

`loss.weights.data` weights CFD and boundary fields;
`loss.weights.observations` weights Schlieren, velocity, and pressure
measurements. `loss.weights.residual` weights mass (`f1`), x-momentum (`f2`),
y-momentum (`f3`), and energy (`f4`). `loss.print_frequency` is a positive
integer controlling how often training losses are printed.

Set `run.checkpoint.save_model: true` to save after training, or
`run.checkpoint.load_model: true` to load the configured checkpoint before
training/evaluation. Set `run.routines.postprocessing: true` for mesh metrics
and VTK export. In `post_processing`, keep `fields`, `components`, `latex`,
`colormaps`, and `scales` aligned in the order `rho, p, u, v` (then `mut` for
RANS). For the standard CFD arrays, use:

```yaml
post_processing:
  fields: [Density, Pressure, Velocity, Velocity, Eddy_Viscosity]
  components: [0, 0, 0, 1, 0]
```

Retain the other post-processing settings from the template. For Euler,
remove the final turbulent-viscosity entry from every aligned list.

## Inverse observations

Enable or disable each modality under `identification.observations`:

- **Schlieren:** generated from the CFD density field. Select `grad_x`,
  `grad_y`, or `magnitude` and configure image resolution, blur, noise, and
  sampling. This loader generates synthetic observations from CFD data.
- **Velocity profiles:** with `filename: velocity_profile` and `n_files: 2`,
  provide `velocity_profile_u_0.csv`, `velocity_profile_u_1.csv`,
  `velocity_profile_v_0.csv`, and `velocity_profile_v_1.csv`. Each file needs
  headers `x,y,u` or `x,y,v` for its component. Indices start at zero.
- **Pressure taps:** with `filename: pressure_taps`, provide
  `pressure_taps.csv` with headers `x,y,p`.

Optional `identification.boundary_conditions` selects named boundaries and
observed fields (`rho`, `u`, `v`, `p`, `mut`) to sample from CFD. Those boundary
names must also appear in `sampling.boundaries.names`. An inverse run needs
nonempty training observations or boundary data. Fresh collocation sampling
still uses the CFD mesh to define the domain.

## Outputs and saved models

| Location | Contents |
| --- | --- |
| `paths.samples` | `data_points.npz` for forward data, `boundary_points.npz` for inverse boundary data, and `collocation_points.npz` for PINN runs, as applicable. |
| `paths.model` | `<files.model_name>.pth` when training with `run.checkpoint.save_model: true`. |
| `paths.results` | Plots and `<name>_comparison.vtk` when full-mesh post-processing is enabled. The VTK file includes reference, predicted, and error fields. |
| Terminal | Training progress and evaluation metrics: MSE, RMSE, relative L2 error, and R². Mesh metrics use nondimensional fields; relative L2 is displayed as a percentage. |

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
still exporting VTK, add `plot: false` under `post_processing`; these field
plots otherwise run on CPU. Sampling and training plots have separate calls.

## Compile the LaTeX guide

A standalone LaTeX version is available in [user-guide.tex](user-guide.tex).
With a TeX distribution providing `pdflatex` and `latexmk`, run from the
repository root:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=docs/build docs/user-guide.tex
```

The PDF is written to `docs/build/user-guide.pdf`. Build files are ignored by Git.
The LaTeX source can also be uploaded to Overleaf as the main document.
