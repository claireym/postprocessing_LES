#!/usr/bin/env python3
"""
3D axisymmetric reconstruction and isosurface visualization of time–theta averaged velocity

Author
------
Claire Yeo MacDougall

Context
-------
Post-processing script for time-averaged CFD simulations (charLES) of a rotating device.
The script:
  1. Reads scattered point data (u_x, u_y, u_z, x, y, z) from a .dat file.
  2. Converts velocities into cylindrical coordinates (u_r, u_theta, u_z) and
     normalizes them by the blade tip speed.
  3. Computes a double mean in (r, z) (averaging over azimuth and time).
  4. Revolves the (r, z) field into 3D to build a PyVista StructuredGrid.
  5. Plots isosurfaces of u_r, u_theta, and +/- u_z overlaid with STL geometry
     for multiple geometries in a single PyVista figure.

Inputs
------
- A list of "cases", each specifying:
    - data_file   : path to a time-averaged, scattered CFD point file.
    - stl_file    : path to an STL file of the device geometry.
    - tip_speed   : tip speed of the device (used to non-dimensionalize velocity).
    - device_radius, device_length : (currently unused in plotting, but kept for clarity).

Data file format (one point per line, 7 columns):
    u_x  u_y  u_z ---  x  y  z

Notes:
- Lines beginning with '#' or empty lines are ignored.
- The 4th column is not used in this script.

Outputs
-------
- A PyVista window with a 3x4 grid of subplots (for 3 cases × 4 variables):
    - row per case
    - columns:
        (a) u_r isosurface
        (b) u_theta isosurface
        (c) positive u_z isosurface
        (d) negative u_z isosurface
- Optional compressed .npz files storing the (r, z) double-mean fields.

Dependencies
------------
- numpy
- pyvista

How to use
----------
1. Edit the `CASES` list to point to your data files, STL files, and tip speeds.
2. Run:
       python double_mean_continuous.py
3. A PyVista window should open with the isosurface figure.

"""

import os
from typing import Tuple, List, Dict, Any

import numpy as np
import pyvista as pv

# -----------------------------------------------------------------------------
# Global PyVista settings
# -----------------------------------------------------------------------------

pv.global_theme.font.label_size = 12
pv.global_theme.font.title_size = 14
pv.global_theme.font.family = "times"

# If True, .npz files of double-mean fields will be saved/replaced.
# If False and a matching .npz exists, it will be loaded instead of recomputing.
SAVE_MEAN: bool = False

# -----------------------------------------------------------------------------
# Case configuration
# -----------------------------------------------------------------------------

CASES: List[Dict[str, Any]] = [
    {
        "name": "Convex Blade",
        "data_file": "/path/to/downsampled_outputs/CON/t_av_10kconvexblade.dat",
        "stl_file": "/path/to/claire_convex.stl",
        "tip_speed": 3.92,      #m/s
        "device_radius": 0.0035,
        "device_length": 0.009,
    },
    {
        "name": "New Geometry",
        "data_file": "/path/to/downsampled_outputs/NEW/t_av_10knew.dat",
        "stl_file": "/path/to/claire_new_geom.stl",
        "tip_speed": 3.92,      # m/s
        "device_radius": 0.0035,
        "device_length": 0.009,
    },
    {
        "name": "Cylindrical",
        "data_file": "/path/to/downsampled_outputs/CYL/t_av_10kcyl.dat",
        "stl_file": "/path/to/spincyl.stl",
        "tip_speed": 3.92,      # m/s
        "device_radius": 0.0012,
        "device_length": 0.009,
    },
]

# -----------------------------------------------------------------------------
# Isosurface settings
# -----------------------------------------------------------------------------

# Factors multiplied by tip_speed to get isosurface levels (dimensionless)
ISO_FACTOR_UR: float = 1.0 / 15.0
ISO_FACTOR_UTHETA: float = 1.0 / 10.0
ISO_FACTOR_UZ_POS: float = 1.0 / 20.0
ISO_FACTOR_UZ_NEG: float = -1.0 / 20.0

# Colors for each field
COLOR_UR: str = "orange"
COLOR_UTHETA: str = "deepskyblue"
COLOR_UZ_POS: str = "mediumpurple"
COLOR_UZ_NEG: str = "violet"

# -----------------------------------------------------------------------------
# Data reading and double-mean computation
# -----------------------------------------------------------------------------

def read_scattered_dat(filename: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Read scattered point data from a .dat file.

    Expected format: 7 floating-point columns:
        u_x  u_y  u_z  ???  x  y  z

    Parameters
    ----------
    filename : str
        Path to data file.

    Returns
    -------
    u_x, u_y, u_z, x, y, z : np.ndarray
    """
    raw: List[List[float]] = []
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) == 7:
                raw.append([float(v) for v in parts])

    data_np = np.array(raw, dtype=float)
    if data_np.size == 0:
        raise ValueError(f"No valid data lines in {filename}")

    u_x = data_np[:, 0]
    u_y = data_np[:, 1]
    u_z = data_np[:, 2]
    x_ = data_np[:, 4]
    y_ = data_np[:, 5]
    z_ = data_np[:, 6]
    return u_x, u_y, u_z, x_, y_, z_


def theta_time_mean(
    u_x: np.ndarray,
    u_y: np.ndarray,
    u_z: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    tip_speed: float,
    save_path: str | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute a "double-mean" field in (r, z) for u_r, u_theta, u_z.

    This bins scattered points into an (Nr, Nz) grid in (r, z) space,
    and averages all points that fall in each bin.

    Velocities are converted to cylindrical coordinates and normalized
    by the tip speed before averaging.

    Parameters
    ----------
    u_x, u_y, u_z : np.ndarray
        Cartesian velocity components at each scattered point.
    x, y, z : np.ndarray
        Point coordinates.
    tip_speed : float
        Tip speed of the device (for non-dimensionalization).
    save_path : str or None
        If not None and SAVE_MEAN is True, save results to this .npz path.

    Returns
    -------
    rAxis, zAxis : np.ndarray
        1D arrays of radial and axial bin centers.
    mean_ur, mean_utheta, mean_uz : np.ndarray
        2D arrays (Nr x Nz) of double-mean fields.
    """
    r = np.sqrt(x**2 + y**2)

    # theta = np.arctan2(y, x)

    Nr, Nz = 100, 100
    rAxis = np.linspace(np.min(r), np.max(r), Nr)
    zAxis = np.linspace(np.min(z), np.max(z), Nz)

    def mean2d(var: np.ndarray) -> np.ndarray:
        sum_f = np.zeros((Nr, Nz), dtype=float)
        count_f = np.zeros((Nr, Nz), dtype=float)

        for i in range(len(var)):
            rr = r[i]
            zz = z[i]
            # Find nearest bin in r and z
            rbin = np.searchsorted(rAxis, rr)
            zbin = np.searchsorted(zAxis, zz)

            # Clamp indices to valid range
            rbin = max(0, min(Nr - 1, rbin))
            zbin = max(0, min(Nz - 1, zbin))

            sum_f[rbin, zbin] += var[i]
            count_f[rbin, zbin] += 1.0

        F_rz = np.zeros((Nr, Nz), dtype=float)
        mask = count_f > 0.0
        F_rz[mask] = sum_f[mask] / count_f[mask]
        return F_rz

    # Cylindrical components (avoid division by zero at r=0)
    r_safe = r + 1e-30
    u_r = (x * u_x + y * u_y) / r_safe
    u_theta = (-y * u_x + x * u_y) / r_safe

    # Non-dimensionalize
    u_r /= tip_speed
    u_theta /= tip_speed
    u_z = u_z / tip_speed

    # Compute double-mean in (r, z)
    mean_ur = mean2d(u_r)
    mean_utheta = mean2d(u_theta)
    mean_uz = mean2d(u_z)

    if save_path is not None and SAVE_MEAN:
        np.savez_compressed(
            save_path,
            rAxis=rAxis,
            zAxis=zAxis,
            mean_ur=mean_ur,
            mean_utheta=mean_utheta,
            mean_uz=mean_uz,
        )

    return rAxis, zAxis, mean_ur, mean_utheta, mean_uz


def load_double_mean(npz_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load a previously saved double-mean field from a .npz file.

    Parameters
    ----------
    npz_path : str
        Path to .npz file created by theta_time_mean.

    Returns
    -------
    rAxis, zAxis, mean_ur, mean_utheta, mean_uz : np.ndarray
    """
    arrs = np.load(npz_path)
    return (
        arrs["rAxis"],
        arrs["zAxis"],
        arrs["mean_ur"],
        arrs["mean_utheta"],
        arrs["mean_uz"],
    )


def revolve_axisymmetric(
    rAxis: np.ndarray,
    zAxis: np.ndarray,
    F_rz: np.ndarray,
    field_name: str = "f_averaged",
) -> pv.StructuredGrid:
    """
    Revolve an axisymmetric (r, z) field into 3D to create a StructuredGrid.

    The field is assumed to be axisymmetric, so F_rz(ir, iz) is replicated around
    the full circle in θ. We only keep physical points for z >= 0. (Negative z
    are set to NaN so they won't contribute to isosurfaces.)

    Parameters
    ----------
    rAxis, zAxis : np.ndarray
        1D arrays of radial and axial coordinates.
    F_rz : np.ndarray, shape (Nr, Nz)
        2D field defined on (rAxis, zAxis).
    field_name : str
        Name used for the scalar field in the PyVista grid.

    Returns
    -------
    grid : pv.StructuredGrid
        PyVista StructuredGrid with coordinates (x, y, z) and scalar field.
    """
    Nr = len(rAxis)
    Nz = len(zAxis)
    Ntheta = 84

    thetaArr = np.linspace(0.0, 2.0 * np.pi, Ntheta + 1, endpoint=True)

    X = np.zeros((Nr, Ntheta + 1, Nz), dtype=float)
    Y = np.zeros((Nr, Ntheta + 1, Nz), dtype=float)
    Z = np.zeros((Nr, Ntheta + 1, Nz), dtype=float)
    Fvals = np.zeros((Nr, Ntheta + 1, Nz), dtype=float)

    for ir in range(Nr):
        rr = rAxis[ir]
        for iz in range(Nz):
            val_ = F_rz[ir, iz]
            z_ = zAxis[iz]

            if z_ >= 0.0:
                for it, th_ in enumerate(thetaArr):
                    X[ir, it, iz] = rr * np.cos(th_)
                    Y[ir, it, iz] = rr * np.sin(th_)
                    Z[ir, it, iz] = z_
                    Fvals[ir, it, iz] = val_
            else:
                # Mark negative z as NaN so they do not appear in plots
                X[ir, :, iz] = np.nan
                Y[ir, :, iz] = np.nan
                Z[ir, :, iz] = np.nan
                Fvals[ir, :, iz] = np.nan

    pts = np.column_stack(
        [
            X.ravel(order="F"),
            Y.ravel(order="F"),
            Z.ravel(order="F"),
        ]
    )
    f_flat = Fvals.ravel(order="F")

    grid = pv.StructuredGrid()
    grid.points = pts
    grid.dimensions = (Nr, Ntheta + 1, Nz)
    grid[field_name] = f_flat
    grid.set_active_scalars(field_name)
    return grid


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def main() -> None:
    """
    Main pipeline:
    - For each case, compute or load double-mean fields.
    - Revolve into 3D structured grids.
    - Create a 3 x 4 PyVista figure (3 cases × 4 variables) with isosurfaces.
    """
    n_cases = len(CASES)
    n_cols = 4

    # Precompute and/or load double-mean data for each case
    double_mean_paths: List[str] = []
    for i, case in enumerate(CASES):
        case_file_name = f"CASE{i + 1}"
        npz_path = f"doublemean_{case_file_name}.npz"
        double_mean_paths.append(npz_path)

        if not (os.path.exists(npz_path) and not SAVE_MEAN):
            # Compute and optionally save double-mean
            u_x, u_y, u_z, x, y, z = read_scattered_dat(case["data_file"])
            theta_time_mean(u_x, u_y, u_z, x, y, z, case["tip_speed"], save_path=npz_path)

    # Create a PyVista plotter with n_cases rows and 4 columns
    p = pv.Plotter(shape=(n_cases, n_cols), window_size=[1600, 1200])
    p.set_background("white")

    def setup_subplot(row_idx: int, col_idx: int, grid: pv.StructuredGrid, iso_factor: float, color: str) -> None:
        """
        Add an isosurface subplot for a given grid and iso_factor in the Plotter.

        Parameters
        ----------
        row_idx, col_idx : int
            Row and column indices in the PyVista subplot grid.
        grid : pv.StructuredGrid
            StructuredGrid with active scalar field.
        iso_factor : float
            Factor multiplied by case['tip_speed'] to get isosurface level.
        color : str
            Color of the isosurface mesh.
        """
        p.subplot(row_idx, col_idx)

        # Light gray background for individual subplots
        p.set_background("lightgrey")

        case = CASES[row_idx]
        iso_val = iso_factor * (grid[field_name := grid.active_scalars_name].max() if case["tip_speed"] == 0.0 else case["tip_speed"])
        # Usually we want iso_val ~ iso_factor * tip_speed.
        iso_val = iso_factor * case["tip_speed"]

        iso = grid.contour(isosurfaces=[iso_val], scalars=grid.active_scalars_name)
        p.add_mesh(iso, color=color, opacity=1.0, show_scalar_bar=False)

        # STL geometry
        stl_mesh = pv.read(case["stl_file"])
        p.add_mesh(
            stl_mesh,
            color="darkgray",
            opacity=0.4,
            lighting=True,
            ambient=0.3,
            diffuse=0.7,
            show_scalar_bar=False,
        )

        # Subfigure label (a, b, c, ...)
        subfig_index = row_idx * n_cols + col_idx
        subfig_letter = chr(ord("a") + subfig_index)
        title = f"{subfig_letter})"
        p.add_text(title, position="upper_left", font_size=11, color="black")

        # Top-down camera view
        R_cam = 0.1
        cx, cy, cz = 0.0, 0.0, 0.005  # focus point
        cam_x, cam_y, cam_z = 0.0, 0.0, R_cam
        p.camera_position = [(cam_x, cam_y, cam_z), (cx, cy, cz), (0, 1, 0)]

        
        p.hide_axes()

    # Fill in all subplots
    for row_idx, case in enumerate(CASES):
        case_file_name = f"CASE{row_idx + 1}"
        npz_path = f"doublemean_{case_file_name}.npz"

        rAxis, zAxis, mean_ur, mean_utheta, mean_uz = load_double_mean(npz_path)
        grid_ur = revolve_axisymmetric(rAxis, zAxis, mean_ur, "u_r")
        grid_utheta = revolve_axisymmetric(rAxis, zAxis, mean_utheta, "u_theta")
        grid_uz = revolve_axisymmetric(rAxis, zAxis, mean_uz, "u_z")

        # Column 0: u_r
        setup_subplot(row_idx, 0, grid_ur, ISO_FACTOR_UR, COLOR_UR)
        # Column 1: u_theta
        setup_subplot(row_idx, 1, grid_utheta, ISO_FACTOR_UTHETA, COLOR_UTHETA)
        # Column 2: positive u_z
        setup_subplot(row_idx, 2, grid_uz, ISO_FACTOR_UZ_POS, COLOR_UZ_POS)
        # Column 3: negative u_z
        setup_subplot(row_idx, 3, grid_uz, ISO_FACTOR_UZ_NEG, COLOR_UZ_NEG)

    p.show()


if __name__ == "__main__":
    main()
