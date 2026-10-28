# SPDX-License-Identifier: MIT
import numpy as np
import pyvista as pv
import gmsh
import os
import pandas as pd
from pathlib import Path
from pirflow.utils.randomness import RANDOM_SEED


class SamplingData:
    """Sample, persist, and expose CFD data or collocation points."""

    # Initialize the class
    def __init__(self, params, collpts=False):
        """Initialize an empty sampling-data container.

        Parameters
        ----------
        params : dict
            PIRFlow configuration.
        collpts : bool, optional
            Whether this container represents collocation points.
        """

        self.collpts = collpts
        self.params = params
        self.dims = params['geometry']['dimension']
        self.rng = np.random.default_rng(RANDOM_SEED)
        # For a collocation object, it is always False
        self.boundary_only = (
            not self.collpts
            and params["run"].get("problem", "forward").lower() == "inverse"
        )

        self.pts_in = np.empty((0, 3), dtype=float)
        self.pts_bc = np.empty((0, 3), dtype=float)
        self.pts_grad = np.empty((0, 3), dtype=float)
        self.pts = np.empty((0, 3), dtype=float)

        self.X = None
        self.U = None
        self.rho = None
        self.p = None
        self.mut = None
        self.observed = None
        self.boundary_names = None

        print("---------------------------------------")
        print("Sample data initialized")
        print(f"  Sampling                       : {params['run']['routines']['sampling']}")
        if params['run']['routines']['sampling']:
            if self.collpts:
                print("Sampling collocation points ...")
            else:
                if self.boundary_only:
                    print("Sampling boundary condition points...") 
                else:
                    print("Sampling data points...")
        else:
            if self.collpts:
                print("Loading collocation points ...")
            else:
                if self.boundary_only:
                    print("Loading boudanry condition points ...")
                else:
                    print("Loading data points ...")

    def sample(self):
        """Sample points and interpolate the configured CFD flowfield.

        Returns
        -------
        None
            Sampled coordinates and fields are stored on this object.
        """

        # Reset arrays before sampling
        self.pts_in = np.empty((0, 3), dtype=float)
        self.pts_bc = np.empty((0, 3), dtype=float)
        self.pts_grad = np.empty((0, 3), dtype=float)
        self.pts = np.empty((0, 3), dtype=float)

        npinner, npgrad, boundaries = self._get_sampling_plan()
        if self.boundary_only:
            self._prepare_boundary_observations(boundaries)

        # No boundary data are requested for this inverse problem.
        if self.boundary_only and not boundaries:
            self.X = self.pts.copy()
            self.U = np.empty((0, 3), dtype=float)
            self.rho = np.empty((0,), dtype=float)
            self.p = np.empty((0,), dtype=float)
            self.mut = np.empty((0, 1), dtype=float)
            return

        # Load your solution
        # .vtk, .pvtu, .vtm, ...
        flowfield_path = (
            Path(self.params["paths"]["flow"])
            / self.params["files"]["flowfield"]
        )
        mesh = pv.read(flowfield_path)   

        # Sample points
        xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
        # Get base sampler function 
        base_sampler = self.get_base_sampler(self.params['sampling']['method'])

        # Inner points
        if npinner > 0:
            pts_in = base_sampler(npinner, xmin, xmax, ymin, ymax)

            if self.dims == 2:
                pts_in = np.column_stack(
                    [pts_in,np.full(pts_in.shape[0], zmin)]
                )

            self.pts_in = np.vstack([self.pts_in, pts_in])

            # Retain valid interior points for the sampling group.
            # Interpolate solution at all points
            sampled = pv.PolyData(self.pts_in).sample(mesh)
            # Interpolates point/cell data onto pts
            mask = sampled["vtkValidPointMask"].astype(bool)
            # Apply the mask in all points
            self.pts_in = self.pts_in[mask]

            # The final interpolation below applies its validity mask again.
            self.pts = np.vstack([self.pts, pts_in])

        elif not self.boundary_only:
            raise ValueError("Number of sample points must be provided.")        

        # Gradient points
        # Add extra points in regions detected by a sensor
        # Extra points based on gradient |grad(rho)|
        if npgrad > 0:
            grad_cfg = self.params["sampling"]["gradient_sampling"]

            pts_grad = self.sample_based_on_grad(
                mesh, 
                npgrad,
                xmin, xmax, ymin, ymax, zmin, zmax,
                base_sampler,
                var_name=grad_cfg.get("variable", "Density"),
                pool_factor=grad_cfg.get("pool_factor", 8),
                alpha=grad_cfg.get("alpha", 1.5)
            )

            self.pts_grad = np.vstack([self.pts_grad, pts_grad])
            # The final interpolation below applies its validity mask again.
            self.pts = np.vstack([self.pts, pts_grad])

        # Boundary condition points
        for phys_name, n_bc in boundaries:
            if n_bc > 0:
                pts_bc = self.sample_boundary_condition(
                    phys_name,
                    n_bc,
                )

                pts_bc = self.nudge_bc_points(
                    pts_bc,
                    phys_name,
                    xmin,
                    xmax,
                    ymin,
                    ymax,
                )

                self.pts_bc = np.vstack([self.pts_bc, pts_bc])
                self.pts = np.vstack([self.pts, pts_bc])

        # Interpolate solution at all points
        point_cloud = pv.PolyData(self.pts)
        # Interpolates point/cell data onto pts
        sampled = point_cloud.sample(mesh)  
        # Apply the mask in all points
        mask = sampled["vtkValidPointMask"].astype(bool)

        if self.boundary_only and not np.all(mask):
            # The cell locator can miss points near narrow cell edges.
            # Retry with the mesh's own cell search at the same coordinates.
            sampled = point_cloud.sample(mesh, locator=None)
            mask = sampled["vtkValidPointMask"].astype(bool)

        if self.boundary_only and not np.all(mask):
            raise ValueError(
                f"{np.count_nonzero(~mask)} selected boundary points "
                "could not be interpolated on the CFD mesh. "
                "Check the geometry, flow mesh, and boundary nudging."
            )

        # sampled.point_data now contains interpolated arrays at your points
        #print(sampled.point_data.keys())

        # Extract arrays
        # Note that the arrays are normalised
        self.pts = self.pts[mask]
        self.X = sampled.points[mask]        # (N,3) or (N,2)

        if self.collpts:
            # Note that, the data is not stored for the collocation points
            # since the predicted flow variables are provided to the 
            # governing equations.
            self.U = sampled.points[mask] * 0.0  
            self.rho = sampled.points[mask] * 0.0  
            self.p = sampled.points[mask] * 0.0
            self.mut = sampled.points[mask] * 0.0

        # The names Density, Velocity, Pressure and Eddy_Viscosity
        # come from the SU2 it should be given by the user.
        # It cannot be define directly from here.

        # 1. Understand the _sample_observed_field function
        # 2. Modify if necessary, adding information on the config file

        elif self.boundary_only:
            self.U = np.column_stack([
                self._sample_observed_field(sampled, "u", "Velocity", 0),
                self._sample_observed_field(sampled, "v", "Velocity", 1),
            ])
            self.rho = self._sample_observed_field(
                sampled, "rho", "Density"
            )
            self.p = self._sample_observed_field(sampled, "p", "Pressure")
            self.mut = self._sample_observed_field(
                sampled, "mut", "Eddy_Viscosity"
            )

        else:
            self.U = sampled["Velocity"][mask]   # (N,3) or (N,2) if vector
            self.rho = sampled["Density"][mask]  # (N,) 
            self.p = sampled["Pressure"][mask]   # (N,) if scalar

            # Depending on the equations the eddy viscosity returns
            # zero or the value from the CFD
            if (self.params['run']['equation'] == 'rans'):
                mut = sampled["Eddy_Viscosity"]
                #self.mut = mut[mask] / params["mu"]
                self.mut = mut[mask]
            elif (self.params['run']['equation'] == 'euler'):
                # Otherwise return zero
                self.mut = np.zeros((self.X.shape[0], 1), dtype=float)

    def _get_sampling_plan(self):
        """Return interior counts, gradient counts, and boundary/count pairs."""

        sampling_config = self.params["sampling"]

        if self.collpts:
            point_config = sampling_config["collocation_points"]
        else:
            point_config = sampling_config["data_points"]

        boundary_names = sampling_config["boundaries"]["names"]
        boundary_counts = point_config["boundary"]

        if len(boundary_names) != len(boundary_counts):
            raise ValueError(
                "sampling.boundaries.names and sampling.data_points.boundary "
                "must have the same length."
            )

        if len(set(boundary_names)) != len(boundary_names):
            raise ValueError("Configured boundary names must be unique.")

        # Preserve the existing plan for forward data and collocation points.
        if not self.boundary_only:
            return (
                point_config["interior"],
                point_config["gradient"],
                list(zip(boundary_names, boundary_counts)),
            )

        # Inverse data contain only enabled, selected boundaries.
        boundary_config = self.params.get(
            "identification", {}).get(
            "boundary_conditions", {}
        )

        if not boundary_config.get("enabled", False):
            return 0, 0, []

        selected_names = boundary_config.get("names", [])

        if not isinstance(selected_names, list) or not selected_names:
            raise ValueError(
                "identification.boundary_conditions.names must be "
                "a non-empty list when boundary conditions are enabled."
            )

        if not all(isinstance(name, str) for name in selected_names):
            raise ValueError("Selected boundary names for identification " \
                             "must be strings.")

        if len(set(selected_names)) != len(selected_names):
            raise ValueError("Selected boundary names for identification " \
                             "must not be repeated.")

        counts_by_name = dict(zip(boundary_names, boundary_counts))

        selected_boundaries = []

        for name in selected_names:
            if name not in counts_by_name:
                raise ValueError(
                    f"Boundary '{name}' is not listed in "
                    "sampling.boundary.names."
                )

            count = counts_by_name[name]

            if (isinstance(count, bool)
                or not isinstance(count, int)
                or count <= 0
            ):
                raise ValueError(
                    f"Selected boundary '{name}' requires a positive "
                    "integer point count."
                )

            selected_boundaries.append((name, int(count)))

        return 0, 0, selected_boundaries

    def _prepare_boundary_observations(self, boundaries):
        """Mark the observed fields at each selected boundary point.

        Missing ``fields`` configuration preserves complete CFD targets.
        Explicit field lists must cover every selected boundary. Masks have
        shape ``(N,)`` and follow the sampling order of the boundary points.
        """

        # Inverse boundary condition config
        config = self.params["identification"]["boundary_conditions"]
        # Prescribed fields
        fields = config.get("fields")

        allowed = ["rho", "u", "v", "p"]
        if self.params["run"]["equation"] == "rans":
            allowed.append("mut")

        # Check if fields is None, if yes fields receives all the 
        # the fields
        names = [name for name, _ in boundaries]
        if fields is None:
            fields = {name: allowed for name in names}

        # Check if fields is a dict or has the same names 
        elif not isinstance(fields, dict) or set(fields) != set(names):
            raise ValueError(
                "boundary_conditions.fields must define exactly the "
                "selected boundary names."
            )

        # Check the selected fields
        for name in names:
            selected = fields[name]
            # 
            if (
                not isinstance(selected, list)
                or not selected
                or not all(isinstance(field, str) for field in selected)
                or len(set(selected)) != len(selected)
                or not set(selected).issubset(allowed)
            ):
                raise ValueError(
                    f"Boundary '{name}' requires a non-empty list of "
                    f"unique fields from {allowed}."
                )


        # Add boundary name for each point        
        self.boundary_names = np.asarray([
            name for name, count in boundaries for _ in range(count)
        ], dtype=str)

        # Add true or false for each variable for the corresponding
        # boundary condition
        self.observed = {
            field: np.asarray([
                field in fields[name]
                for name, count in boundaries
                for _ in range(count)
            ], dtype=bool)
            for field in ("rho", "u", "v", "p", "mut")
        }

    def _sample_observed_field(self, sampled, field, array, component=None):
        """Extract only observed values; unobserved entries remain NaN."""

        # Retrive the if the variable was provided for each boundary and field
        # {"rho", [True, False, ...]} 
        observed = self.observed[field]
        # Create an array with the same shape, filled with Nan 
        values = np.full(observed.shape, np.nan, dtype=float)
        
        if not np.any(observed):
            return values
        
        # Extract the unique values of the boundary conditions 
        names = np.unique(self.boundary_names[observed]).tolist()

        if array not in sampled.point_data:
            raise ValueError(
                f"Boundaries {names} request '{field}', but CFD array "
                f"'{array}' is missing."
            )

        sampled_values = np.asarray(sampled[array])
        if component is not None:
            if (
                sampled_values.ndim != 2
                or sampled_values.shape[1] <= component
            ):
                raise ValueError(
                    f"CFD array '{array}' has no component '{field}'."
                )
            sampled_values = sampled_values[:, component]

        values[observed] = sampled_values.reshape(-1)[observed]
        if not np.all(np.isfinite(values[observed])):
            raise ValueError(
                f"Boundaries {names} have non-finite '{field}' observations."
            )

        return values


    def get_base_sampler(self, sampling_type: str):
        """Return the point-sampling function selected by name.

        Parameters
        ----------
        sampling_type : str
            Sampling method, either ``"random"`` or ``"lhs"``.

        Returns
        -------
        callable
            Bound method that generates points.
        """
        if sampling_type == "random":
            return self.sample_random_points
        elif sampling_type == "lhs":
            return self.sample_latin_hypercube
        else:
            raise ValueError("sampling type must be 'random' or 'lhs'")

    def sample_random_points(self, npoin, xmin, xmax, ymin, ymax):
        """Draw uniformly distributed points in a rectangular domain.

        Parameters
        ----------
        npoin : int
            Number of points.
        xmin, xmax, ymin, ymax : float
            Coordinate bounds.

        Returns
        -------
        numpy.ndarray
            Sample coordinates with shape ``(npoin, 2)``.
        """
        pts = np.column_stack([
            self.rng.uniform(xmin, xmax, npoin),
            self.rng.uniform(ymin, ymax, npoin),
            ])
        return pts 
    
    def sample_latin_hypercube(self, npoin, xmin, xmax, ymin, ymax):
        """Draw Latin-hypercube points in a rectangular domain.

        Parameters
        ----------
        npoin : int
            Number of points.
        xmin, xmax, ymin, ymax : float
            Coordinate bounds.

        Returns
        -------
        numpy.ndarray
            Sample coordinates with shape ``(npoin, 2)``.
        """
        try:
            from scipy.stats import qmc
            sampler = qmc.LatinHypercube(d=self.dims, rng=self.rng)
            u01 = sampler.random(n=npoin)              # (N,2) in [0,1)
        except ImportError:
            u01 = np.empty((npoin, self.dims))
            for j in range(self.dims):
                perm = self.rng.permutation(npoin)
                # one stratum per point
                u01[:,j] = (perm + self.rng.random(npoin)) / npoin

        # Map to physical domain bounds
        pts = np.empty_like(u01)
        pts[:,0] = xmin + (xmax - xmin) * u01[:,0]
        pts[:,1] = ymin + (ymax - ymin) * u01[:,1]
        
        return pts
    
    def sample_boundary_condition(self, phys_name, npoin_bc):
        """Sample nodes from a physical group in the Gmsh geometry.

        Parameters
        ----------
        phys_name : str
            Name of the physical boundary group.
        npoin_bc : int
            Number of boundary points to draw.

        Returns
        -------
        numpy.ndarray
            Boundary coordinates with shape ``(npoin_bc, 3)``.
        """

        rng = self.rng
        if npoin_bc <= 0:
            raise ValueError("Number of boundary sampling points must be > 0")

        gmsh.initialize()
        # Turn off terminal output from Gmsh
        gmsh.option.setNumber("General.Terminal", 0)
        try:
            geometry_path = (
                Path(self.params["paths"]["geometry"])
                / self.params["files"]["geometry"]
            )
            gmsh.open(str(geometry_path))
            gmsh.model.geo.synchronize()
            #gmsh.model.mesh.generate(self.dims)
            # For 1D mesh generation before 2D
            gmsh.model.mesh.generate(self.dims-1)   # mesh curves (boundary)
            gmsh.model.mesh.generate(self.dims)   # mesh surface

            # Find physical group tag by name
            # Gmsh stores physical groups by (dim, tag)
            phys_groups = gmsh.model.getPhysicalGroups()
            matches = []

            for d, tag in phys_groups:
                name = gmsh.model.getPhysicalName(d, tag)
                if name == phys_name:
                    matches.append((d, tag))

            if not matches:
                raise ValueError(f"Physical group '{phys_name}' not found.")

            # if self.dims == 2 (2D domain), boundary is dim=1 (curves)
            # if self.dims == 3 (3D domain), boundary is dim=2 (surfaces)
            bc_dim = self.dims - 1
            d_use, tag_use = matches[0]  # There is only one per call
            for d, tag in matches:
                if d == bc_dim:
                    d_use, tag_use = d, tag

           # Robust node extraction: physical group -> entities -> nodes
            ent_tags = gmsh.model.getEntitiesForPhysicalGroup(d_use, tag_use)
            if len(ent_tags) == 0:
                raise RuntimeError(f"Physical group '{phys_name}' has no entities \
                                   (dim={d_use}, tag={tag_use}).")

            all_pts_bc = []
            for e in ent_tags:
                nodes = gmsh.model.mesh.getNodes(d_use, e)
                node_coords = nodes[1]  # works for both 2-return and 3-return variants
                if len(node_coords) > 0:
                    # Make the array (N,3)
                    all_pts_bc.append(np.asarray(node_coords, dtype=float).reshape(-1, 3))

            if not all_pts_bc:
                raise RuntimeError(
                    f"No nodes found for physical group '{phys_name}' on entities {ent_tags} "
                    f"(dim={d_use}, tag={tag_use})."
                )

            # Put the list of arrays in a single array, since all_pyts is the 
            # composition of many lists related to the possible entities that 
            # compose the boundary condition, e.g., many lines or surfaces.
            pts_bc = np.vstack(all_pts_bc)

            # Remove duplicates (curves share endpoints)
            # axis=0 means “consider each row [x,y,z] as one item
            pts_bc = np.unique(pts_bc, axis=0)

            # Sample with replacement if you ask for more than available.
            # It also reduce the number of points if it is too high accordanly
            # with the value provided (npoin_bc)
            idx = rng.integers(0, pts_bc.shape[0], size=npoin_bc)
            return pts_bc[idx]

        finally:
            gmsh.finalize()                

    def sample_based_on_grad(self, mesh, npoin_grad,
        xmin, xmax, ymin, ymax, zmin, zmax, base_sampler,
        var_name="Density",
        pool_factor=8,
        alpha=1.5,
        eps=1e-12):
        """Sample mesh points with preference for large field gradients.

        Parameters
        ----------
        mesh : pyvista.DataSet
            CFD mesh containing the selected field.
        npoin_grad : int
            Requested number of points.
        xmin, xmax, ymin, ymax, zmin, zmax : float
            Mesh bounds.
        base_sampler : callable
            Function used to create candidate points.
        var_name : str, optional
            Field whose gradient controls sampling.
        pool_factor : int, optional
            Candidate count multiplier.
        alpha : float, optional
            Gradient-weight exponent.
        eps : float, optional
            Small value used in gradient weights.

        Returns
        -------
        numpy.ndarray
            Selected coordinates with three components per point.
        """

        rng = self.rng

        # Compute grad(var) on the mesh
        if var_name not in mesh.array_names:
            raise ValueError(f"'{var_name}' not found in mesh arrays: \
                {mesh.array_names}")

        mesh_g = mesh.compute_derivative(scalars=var_name, gradient=True)
        if "gradient" not in mesh_g.point_data:
            raise RuntimeError("Gradient not found after compute_derivative().")

        # Get the gradient vector 
        grad_vec = mesh_g.point_data["gradient"]
        # Calculate the norm
        mesh_g.point_data["grad_mag"] = np.linalg.norm(grad_vec, axis=1)

        # Candidate pool
        npoin_pool = int(pool_factor * npoin_grad)
        cand = base_sampler(npoin_pool, xmin, xmax, ymin, ymax)   

        if self.dims == 2:
            cand = np.column_stack([cand, np.full((cand.shape[0],), zmin)])

        # Keep only candidates that lie in some cell
        cell_ids = mesh_g.find_containing_cell(cand)
        cand = cand[cell_ids >= 0]
        if cand.shape[0] == 0:
            return np.empty((0, 3))

        # Evaluate grad magnitude at candidates 
        sampled = pv.PolyData(cand).sample(mesh_g)

        if "vtkValidPointMask" in sampled.point_data:
            mask = sampled["vtkValidPointMask"].astype(bool)
            pts_in = sampled.points[mask]
            grad_mag = sampled["grad_mag"][mask]
        else:
            pts_in = sampled.points
            grad_mag = sampled["grad_mag"]
        
        # Check if pts_in is zero, so no point was interpolated, all points
        # were outside of the geometry.
        if pts_in.shape[0] == 0:
            return np.empty((0, 3))

        # accept–reject with p = (g+eps)^alpha
        w = (grad_mag + eps) ** alpha
        wmax = np.max(w) 
        # Check for Nan/inf or negativa values
        if (not np.isfinite(wmax)) or wmax <= 0:
            return np.empty((0, 3))

        # Convert weights to acceptance probabilities in [0,1]
        p = w / wmax # probability
        # Note that if q < p, we keep the points because p
        # has a high probability value.
        keep = rng.random(p.shape[0]) < p
        pts_grad = pts_in[keep]

        # Ensure exactly npoin_grad points (top-up by highest weights)
        if pts_grad.shape[0] < npoin_grad:
            order = np.argsort(w)[::-1]
            need = npoin_grad - pts_grad.shape[0]
            pts_grad = np.vstack([pts_grad, pts_in[order[:need]]])
        else:
            pts_grad = pts_grad[:npoin_grad]

        return pts_grad
        
    def nudge_bc_points(self, pts_bc, name, xmin, xmax, ymin, ymax):
        """Move axis-aligned boundary points slightly into the domain.

        Parameters
        ----------
        pts_bc : numpy.ndarray
            Boundary coordinates.
        name : str
            Boundary name.
        xmin, xmax, ymin, ymax : float
            Domain bounds used to scale the displacement.

        Returns
        -------
        numpy.ndarray
            Nudged copy of the boundary coordinates.
        """
        pts = pts_bc.copy()

        # scale-aware eps (tiny fraction of domain size)
        epsx = 1e-7 * (xmax - xmin)
        epsy = 1e-7 * (ymax - ymin)

        n = name.lower()
        if n == "inlet":      # x = xmin
            pts[:, 0] += epsx
        elif n == "outlet":   # x = xmax
            pts[:, 0] -= epsx
        elif n == "bottom":   # y = ymin
            pts[:, 1] += epsy
        elif n == "top":      # y = ymax
            pts[:, 1] -= epsy

        return pts

    def write_data_to_npz(self):
        """Save the sampled arrays to the configured sample directory.

        Returns
        -------
        None
            Data are written to a compressed NumPy file.
        """

        path_data = Path(self.params['paths']['samples'])
        path_data.mkdir(parents=True, exist_ok=True)

        filename = self._get_sample_path()

        if self.collpts:
            np.savez_compressed(
                filename,
                X=self.X,
                pts_in=self.pts_in,
                pts_bc=self.pts_bc,
                pts_grad=self.pts_grad,
                collpts=np.array(True),
            )

        else:
            boundary_info = {}

            if self.boundary_only:
                _, _, boundaries = self._get_sampling_plan()

                boundary_info = {
                    "boundary_names": np.asarray(
                        [name for name, _ in boundaries],
                        dtype=str,
                    ),
                    "boundary_counts": np.asarray(
                        [count for _, count in boundaries],
                        dtype=int,
                    ),
                    **{
                        f"observed_{field}": observed
                        for field, observed in self.observed.items()
                    },
                }

            np.savez_compressed(
                filename,
                X=self.X,
                U=self.U,
                rho=self.rho,
                p=self.p,
                mut=self.mut,
                pts_in=self.pts_in,
                pts_bc=self.pts_bc,
                pts_grad=self.pts_grad,
                collpts=np.array(False),
                **boundary_info, # dictionary unpacking: ** 
            )

        print("---------------------------------------")
        print(f"Saved data points to: {filename}")

    def read_data_from_npz(self):
        """Load sampled arrays from the configured sample directory.

        Returns
        -------
        tuple
            Coordinates, point groups, and optional flow variables.
        """

        filename = self._get_sample_path()

        if not filename.is_file():
            raise FileNotFoundError(
                f"Sample file was not found:\n  {filename}\n"
                "Enable run.routines.sampling to generate it."
            )

        data = np.load(filename)

        if self.boundary_only:
            _, _, boundaries = self._get_sampling_plan()
            self._prepare_boundary_observations(boundaries)

            expected_names = [name for name, _ in boundaries]
            expected_counts = [count for _, count in boundaries]

            if (
                "boundary_names" not in data
                or "boundary_counts" not in data
                or data["boundary_names"].tolist() != expected_names
                or data["boundary_counts"].tolist() != expected_counts
            ):
                raise ValueError(
                    "Saved boundary names/counts differ from the "
                    "configuration. "
                    "Enable run.routines.sampling to regenerate the samples."
                )

            for field, observed in self.observed.items():
                key = f"observed_{field}"
                if key in data and not np.array_equal(data[key], observed):
                    raise ValueError(
                        "Saved boundary fields differ from the "
                        "configuration. Enable run.routines.sampling "
                        "to regenerate the samples."
                    )

        X = data["X"]
        pts_in = data["pts_in"]
        pts_bc = data["pts_bc"]
        pts_grad = data["pts_grad"]

        if not self.collpts:
            U = data["U"]
            rho = data["rho"]
            p = data["p"]
            mut = data["mut"]

            if self.boundary_only:
                if len(X) != len(self.boundary_names):
                    raise ValueError(
                        "Saved boundary coordinates do not match the "
                        "configured point counts. Regenerate the samples."
                    )
                # Legacy caches contain all fields. Apply current selection.
                values = {"rho": rho, "u": U[:, 0], "v": U[:, 1],
                          "p": p, "mut": mut}
                for field, target in values.items():
                    selected = self.observed[field]
                    target = target.reshape(-1)
                    if not np.all(np.isfinite(target[selected])):
                        raise ValueError(
                            f"Saved '{field}' observations are non-finite. "
                            "Regenerate the boundary samples."
                        )
                    target[~selected] = np.nan
        else:
            U = None
            rho = None
            p = None
            mut = None

        return X, pts_in, pts_bc, pts_grad, U, rho, p, mut

    def _get_sample_path(self):
        """Return the sample-file path for this sampling object."""

        if self.collpts:
            filename = "collocation_points.npz"
        elif self.boundary_only:
            filename = "boundary_points.npz"
        else:
            filename = "data_points.npz"

        return Path(self.params["paths"]["samples"]) / filename

    def get_boundary_marker(self):
        """Return the boundary-marker array, when available.

        Returns
        -------
        numpy.ndarray or None
            Boundary markers associated with sampled points.
        """
        return self.boundary_marker

    def get_pts_in(self):       
        """Return sampled interior points.

        Returns
        -------
        numpy.ndarray
            Interior coordinates.
        """
        return self.pts_in
    
    def get_pts_bc(self):       
        """Return sampled boundary points.

        Returns
        -------
        numpy.ndarray
            Boundary coordinates.
        """
        return self.pts_bc

    def get_pts_grad(self):       
        """Return sampled gradient-focused points.

        Returns
        -------
        numpy.ndarray
            Gradient-focused coordinates.
        """
        return self.pts_grad

    def get_x(self):       
        """Return all valid sampled coordinates.

        Returns
        -------
        numpy.ndarray
            Coordinates retained after mesh masking.
        """
        return self.X

    def get_rho(self):
        """Return sampled density values.

        Returns
        -------
        numpy.ndarray or None
            Density observations, or ``None`` for loaded collocation data.
        """
        return self.rho

    def get_u(self):       
        """Return sampled velocity vectors.

        Returns
        -------
        numpy.ndarray or None
            Velocity observations, or ``None`` for loaded collocation data.
        """
        return self.U

    def get_p(self):
        """Return sampled pressure values.

        Returns
        -------
        numpy.ndarray or None
            Pressure observations, or ``None`` for loaded collocation data.
        """
        return self.p
    
    def get_mut(self):
        """Return sampled turbulent-viscosity values.

        Returns
        -------
        numpy.ndarray or None
            Turbulent viscosity values, or ``None`` when unavailable.
        """
        return self.mut
