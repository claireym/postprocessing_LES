#!/usr/bin/env python3
"""
Discrete 2D wedge-averaged field reconstruction and STL overlay

Author
------
Claire Yeo MacDougall

Context
-------
Post-processing script for 3D CFD simulations (charLES) of a rotating device.
This script reconstructs 2D wedge-averaged fields at several axial planes from
unstructured probe data and overlays the results on STL slices of the geometry.

Overview
--------
The pipeline is:

1. Read probe locations from `*.README` files and corresponding time series
   data from `*.P` / `*.COMP(U,*)` files for each axial "plane" (z-cut).

2. For each plane:
   - Optionally time-average a chosen field (pressure, axial velocity,
     radial or azimuthal velocity magnitude).
   - Convert probe coordinates (x, y) to polar coordinates (r, θ).
   - Split the annulus into six azimuthal wedges (60° each).
   - In each wedge, map scattered points into a local Cartesian wedge
     coordinate system and interpolate onto a regular (r, θ) grid.

3. Average the interpolated fields over all valid wedges to obtain a single
   representative wedge field for that plane.

4. Slice the STL geometry at the corresponding z-cut, divide the segments
   into six wedges, and overlay the first wedge on top of the wedge-averaged
   field for visual comparison.

5. Optionally, for a selected plane, generate a multi-panel figure showing
   different variables (azimuthal, radial, axial velocity, pressure) side by side.

Inputs
------
- BASE_DIR (str): Root directory containing subdirectories for each data type
                  (e.g. "p", "ux", "uy", "uz"). Each subdirectory should contain
                  files named like:
                      {plane_id}.README
                      {plane_id}.P
                      {plane_id}.COMP(U,0)
                      {plane_id}.COMP(U,1)
                      {plane_id}.COMP(U,2)

- STL file (str): Triangulated surface mesh of the device, used for slicing.

- PLANES (dict): Mapping of plane IDs (e.g. "1", "2", "3", "4", "tip") to
                 z-cut locations (float).

Outputs
-------
- Figures showing wedge-averaged fields for all planes with STL wedge overlays.
- A multi-panel figure showing different variables on a selected plane.

Dependencies
------------
- numpy
- scipy (scipy.interpolate.griddata)
- matplotlib
- numpy-stl (stl.Mesh)
- mpl_toolkits.axes_grid1 (make_axes_locatable)

Notes
-----
- Tip speed is used to non-dimensionalize velocities.
- Pressure is normalized by its mean magnitude in the averaging window.
- This script is written as research post-processing code; for production. 
Future work: add CLI argument parsing and error handling.

"""

from __future__ import annotations

import os
import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from scipy.interpolate import griddata
from stl import mesh
from mpl_toolkits.axes_grid1 import make_axes_locatable

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

#: Base directory for probe data (adjust this path before use)
BASE_DIR: str = "/path/to/downsampled_outputs/PROBESi"

#: Mapping from logical data type to file suffix used by the charLES outputs
DATA_TYPES: Dict[str, str] = {
    "p": "P",
    "ux": "COMP(U,0)",
    "uy": "COMP(U,1)",
    "uz": "COMP(U,2)",
}

#: Axial planes and their corresponding z-cuts (in meters)
PLANES: Dict[str, float] = {
    "1": 0.0015,
    "2": 0.0030,
    "3": 0.0050,
    "4": 0.0070,
    "tip": 0.0100,
}

#: Tip speed used for non-dimensionalization of velocity components
TIP_SPEED: float = 3.92  # adjust if needed


# -----------------------------------------------------------------------------
# Geometry utilities (STL slicing and wedge partitioning)
# -----------------------------------------------------------------------------

def line_plane_intersection(p1: np.ndarray, p2: np.ndarray, z_cut: float) -> Optional[np.ndarray]:
    """
    Compute the intersection of a segment (p1 -> p2) with the plane z = z_cut.

    Returns
    -------
    intersection : np.ndarray or None
        Array [x, y, z_cut] if the segment intersects the plane, otherwise None.
    """
    z1, z2 = p1[2], p2[2]

    # If both points on same side of plane or segment is nearly parallel in z,
    # we consider there is no intersection.
    if (z1 > z_cut and z2 > z_cut) or (z1 < z_cut and z2 < z_cut):
        return None
    if abs(z2 - z1) < 1e-12:
        return None

    t = (z_cut - z1) / (z2 - z1)
    if 0.0 <= t <= 1.0:
        x = p1[0] + t * (p2[0] - p1[0])
        y = p1[1] + t * (p2[1] - p1[1])
        return np.array([x, y, z_cut], dtype=float)

    return None


def slice_mesh_at_z(stl_mesh: mesh.Mesh, z_cut: float) -> List[np.ndarray]:
    """
    Slice an STL mesh with a horizontal plane z = z_cut and return 2D line segments.

    Parameters
    ----------
    stl_mesh : stl.mesh.Mesh
        Triangulated surface mesh.
    z_cut : float
        Axial location of the slicing plane.

    Returns
    -------
    segments : list of (2, 2) arrays
        Each segment is defined by two points [x, y].
    """
    segments: List[np.ndarray] = []

    for triangle in stl_mesh.vectors:
        points_2d: List[np.ndarray] = []
        for i in range(3):
            p1, p2 = triangle[i], triangle[(i + 1) % 3]
            intersection = line_plane_intersection(p1, p2, z_cut)
            if intersection is not None:
                points_2d.append(intersection[:2])
        if len(points_2d) == 2:
            segments.append(np.vstack(points_2d))

    return segments


def divide_into_wedges(
    segments: Sequence[np.ndarray],
    num_wedges: int = 6
) -> List[List[np.ndarray]]:
    """
    Divide STL line segments into azimuthal wedges based on segment midpoint angle.

    Parameters
    ----------
    segments : sequence of (2, 2) arrays
        Line segments on the z-cut plane.
    num_wedges : int
        Number of azimuthal wedges (default: 6).

    Returns
    -------
    wedges : list of list of segments
        wedges[k] contains the segments belonging to wedge k.
    """
    wedges: List[List[np.ndarray]] = [[] for _ in range(num_wedges)]
    wedge_angle = 2.0 * np.pi / num_wedges

    for seg in segments:
        midpoint = np.mean(seg, axis=0)
        angle = math.atan2(midpoint[1], midpoint[0])
        if angle < 0.0:
            angle += 2.0 * np.pi
        wedge_index = int(angle / wedge_angle) % num_wedges
        wedges[wedge_index].append(seg)

    return wedges


# -----------------------------------------------------------------------------
# Probe data parsing
# -----------------------------------------------------------------------------

def parse_readme(file_path: str) -> List[Tuple[int, float, float, float]]:
    """
    Parse a {plane_id}.README file containing probe indices and coordinates.

    Expected format (whitespace separated, comments starting with '#'):
        probe_index  x  y  z

    Returns
    -------
    points : list of (idx, x, y, z)
    """
    points: List[Tuple[int, float, float, float]] = []

    if not os.path.exists(file_path):
        print(f"Warning: README file {file_path} does not exist.")
        return points

    with open(file_path, "r") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            if len(parts) == 4:
                probe_idx, x, y, z = parts
                points.append((int(probe_idx), float(x), float(y), float(z)))

    return points


def parse_data_file(file_path: str) -> np.ndarray:
    """
    Parse a time-series data file for a given plane and data type.

    Expected format (whitespace separated, comments starting with '#'):
        time_index  probe_index  ???  value_1  value_2 ...  value_N

    Only the numeric probe values after the first three columns are retained.

    Returns
    -------
    data : np.ndarray, shape (num_snapshots, num_probes)
    """
    if not os.path.exists(file_path):
        print(f"Warning: data file {file_path} does not exist.")
        return np.array([])

    data: List[List[float]] = []

    with open(file_path, "r") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 4:
                _, _, _, *probe_data = parts
                data.append(list(map(float, probe_data)))

    return np.array(data, dtype=float)


# -----------------------------------------------------------------------------
# Velocity decomposition and time averaging
# -----------------------------------------------------------------------------

def compute_radial_and_azimuthal_velocity(
    points: Sequence[Tuple[int, float, float, float]],
    ux_values: np.ndarray,
    uy_values: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute radial and azimuthal velocity components in the x-y plane.

    Parameters
    ----------
    points : sequence of (idx, x, y, z)
        Probe coordinates.
    ux_values, uy_values : np.ndarray, shape (num_probes,)
        Instantaneous velocity components in Cartesian coordinates.

    Returns
    -------
    u_r : np.ndarray
        Radial velocity component at each probe.
    u_theta : np.ndarray
        Azimuthal velocity component at each probe.
    """
    u_r: List[float] = []
    u_theta: List[float] = []

    for (_, x, y, _), ux, uy in zip(points, ux_values, uy_values):
        r = math.sqrt(x * x + y * y)
        if r == 0.0:
            # At or near origin: set components to zero
            u_r.append(0.0)
            u_theta.append(0.0)
            continue

        # Unit vectors in radial and azimuthal directions
        e_r = np.array([x / r, y / r])
        e_theta = np.array([-y / r, x / r])

        u_vec = np.array([ux, uy])
        u_r.append(float(np.dot(u_vec, e_r)))
        u_theta.append(float(np.dot(u_vec, e_theta)))

    return np.array(u_r), np.array(u_theta)


def compute_average_field(data: np.ndarray, start_index: int, end_index: int) -> np.ndarray:
    """
    Compute the mean field over snapshots in [start_index, end_index).

    Parameters
    ----------
    data : np.ndarray
        Time-series data, shape (num_snapshots, num_probes).
    start_index, end_index : int
        Start (inclusive) and end (exclusive) snapshot indices.

    Returns
    -------
    mean_field : np.ndarray
        Averaged field over the specified window.
    """
    return np.mean(data[start_index:end_index], axis=0)


def process_plane_and_type(
    plane_id: str,
    data_type: str,
    max_points: Optional[int] = 10_000,
) -> Tuple[List[Tuple[int, float, float, float]], np.ndarray]:
    """
    Load probe coordinates and time-series data for a given plane and data type.

    Parameters
    ----------
    plane_id : str
        One of the keys in PLANES, e.g. "1", "2", "3", "4", "tip".
    data_type : str
        One of "p", "ux", "uy", "uz".
    max_points : int or None
        Optional limit on the number of probes used (for sampling).

    Returns
    -------
    plane_points : list of (idx, x, y, z)
    data : np.ndarray, shape (num_snapshots, num_probes)
    """
    file_suffix = DATA_TYPES.get(data_type)
    if not file_suffix:
        raise ValueError(f"Invalid data type '{data_type}'.")

    readme_file = os.path.join(BASE_DIR, data_type, f"{plane_id}.README")
    data_file = os.path.join(BASE_DIR, data_type, f"{plane_id}.{file_suffix}")

    plane_points = parse_readme(readme_file)
    data = parse_data_file(data_file)

    if max_points is not None:
        plane_points = plane_points[:max_points]
        data = data[:, :len(plane_points)]

    return plane_points, data


def process_snapshot(
    plane_id: str,
    data_type: str,
    snapshot_index: Optional[int] = None,
    average_window: Optional[Tuple[int, int]] = None,
) -> Tuple[
    Optional[List[Tuple[int, float, float, float]]],
    Optional[np.ndarray],
    Optional[Tuple[int, int]],
]:
    """
    Extract a snapshot or time-averaged field for a given plane and variable.

    Parameters
    ----------
    plane_id : str
        Key in PLANES.
    data_type : str
        One of "p", "ux", "uy", "uz", "radial", "azimuthal".
    snapshot_index : int, optional
        If given, use a single snapshot index from the time series.
    average_window : (start, end), optional
        If given, average over snapshots in [start, end).

    Returns
    -------
    plane_points : list or None
        Probe coordinates.
    field_values : np.ndarray or None
        Field values for the selected snapshot or average.
    window : (start, end) or None
        The averaging window used, if any.
    """
    if snapshot_index is None and average_window is None:
        raise ValueError("Either snapshot_index or average_window must be provided.")

    # Radial/azimuthal derived from ux, uy
    if data_type in ("radial", "azimuthal"):
        plane_points, ux_data = process_plane_and_type(plane_id, "ux")
        _, uy_data = process_plane_and_type(plane_id, "uy")

        if ux_data.size == 0 or uy_data.size == 0 or not plane_points:
            print(f"Missing ux/uy data for radial/azimuthal calculation on plane {plane_id}.")
            return None, None, average_window

        if snapshot_index is not None:
            u_r, u_theta = compute_radial_and_azimuthal_velocity(
                plane_points, ux_data[snapshot_index], uy_data[snapshot_index]
            )
            values = u_r if data_type == "radial" else u_theta
            values = values / TIP_SPEED
            return plane_points, values, None

        # Time-averaged
        start, end = average_window
        radial_list: List[np.ndarray] = []
        azimuthal_list: List[np.ndarray] = []
        for i in range(start, end):
            u_r, u_theta = compute_radial_and_azimuthal_velocity(
                plane_points, ux_data[i], uy_data[i]
            )
            radial_list.append(u_r)
            azimuthal_list.append(u_theta)

        avg_r = np.mean(radial_list, axis=0) / TIP_SPEED
        avg_theta = np.mean(azimuthal_list, axis=0) / TIP_SPEED
        values = avg_r if data_type == "radial" else avg_theta
        return plane_points, values, average_window

    # Direct fields: p, ux, uy, uz
    plane_points, data = process_plane_and_type(plane_id, data_type)
    if data.size == 0 or not plane_points:
        print(f"No data available for plane {plane_id} and data type {data_type}.")
        return None, None, average_window

    if snapshot_index is not None:
        snapshot_values = data[snapshot_index]
        if data_type == "uz":
            snapshot_values = snapshot_values / TIP_SPEED
        elif data_type == "p":
            mean_p = np.mean(snapshot_values)
            if mean_p != 0.0:
                snapshot_values = snapshot_values / mean_p
        return plane_points, snapshot_values, None

    # Time-averaged
    start, end = average_window
    avg_field = compute_average_field(data, start, end)
    if data_type == "uz":
        avg_field = avg_field / TIP_SPEED
    elif data_type == "p":
        mean_p = abs(np.mean(avg_field))
        if mean_p != 0.0:
            avg_field = avg_field / mean_p

    return plane_points, avg_field, average_window


# -----------------------------------------------------------------------------
# Wedge averaging and interpolation
# -----------------------------------------------------------------------------

def compute_averaged_wedge_field(
    plane_points: Optional[Sequence[Tuple[int, float, float, float]]],
    values: Optional[np.ndarray],
    grid_radius: float = 0.005,
    grid_resolution: int = 100,
) -> Tuple[
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[float],
    Optional[float],
]:
    """
    Interpolate a field onto a local wedge grid and average over symmetric wedges.

    Parameters
    ----------
    plane_points : list of (idx, x, y, z) or None
        Probe coordinates.
    values : np.ndarray or None
        Field values at each probe (same length as plane_points).
    grid_radius : float
        Maximum radial extent of the local wedge grid.
    grid_resolution : int
        Number of grid points in radial and angular directions.

    Returns
    -------
    averaged_wedge : np.ndarray or None
        Wedge-averaged field values on the (r, θ) grid.
    wedge_x_grid, wedge_y_grid : np.ndarray or None
        Cartesian coordinates of wedge grid points.
    global_min, global_max : float or None
        Min/max of interpolated wedges (for color scaling).
    """
    if plane_points is None or values is None:
        return None, None, None, None, None

    # Extract x, y from probe coordinates
    x = np.array([p[1] for p in plane_points], dtype=float)
    y = np.array([p[2] for p in plane_points], dtype=float)
    values = np.asarray(values, dtype=float)

    # Polar coordinates of probes
    angles = np.arctan2(y, x)
    radius = np.sqrt(x ** 2 + y ** 2)

    # Define six wedges in [-π, π)
    wedge_boundaries = np.linspace(-np.pi, np.pi, 7)
    wedge_width = np.pi / 3.0

    # Local wedge grid in (r, θ)
    wedge_theta = np.linspace(0.0, wedge_width, grid_resolution)
    wedge_r = np.linspace(0.0, grid_radius, grid_resolution)
    wedge_theta_grid, wedge_r_grid = np.meshgrid(wedge_theta, wedge_r)
    wedge_x_grid = wedge_r_grid * np.cos(wedge_theta_grid)
    wedge_y_grid = wedge_r_grid * np.sin(wedge_theta_grid)

    global_min = np.inf
    global_max = -np.inf
    wedge_data_list: List[Optional[np.ndarray]] = []

    for i in range(6):
        wedge_mask = (angles >= wedge_boundaries[i]) & (angles < wedge_boundaries[i + 1])
        if not np.any(wedge_mask):
            wedge_data_list.append(None)
            continue

        wedge_r_vals = radius[wedge_mask]
        wedge_angle_vals = angles[wedge_mask]
        wedge_val = values[wedge_mask]

        # Shift angles so each wedge starts at 0
        angle_start = wedge_boundaries[i]
        wedge_angle_shifted = wedge_angle_vals - angle_start

        # Map data to local wedge coordinates
        wedge_x_mapped = wedge_r_vals * np.cos(wedge_angle_shifted)
        wedge_y_mapped = wedge_r_vals * np.sin(wedge_angle_shifted)

        points_xy = np.vstack((wedge_x_mapped, wedge_y_mapped)).T
        wedge_z = griddata(points_xy, wedge_val, (wedge_x_grid, wedge_y_grid), method="linear")

        if wedge_z is not None:
            current_min = np.nanmin(wedge_z)
            current_max = np.nanmax(wedge_z)
            if current_min < global_min:
                global_min = current_min
            if current_max > global_max:
                global_max = current_max

        wedge_data_list.append(wedge_z)

    if np.isinf(global_min) or np.isinf(global_max):
        global_min, global_max = 0.0, 1.0

    valid_wedges = [w for w in wedge_data_list if w is not None]
    if len(valid_wedges) == 0:
        return None, None, None, None, None

    stacked_wedges = np.dstack(valid_wedges)
    averaged_wedge = np.nanmean(stacked_wedges, axis=2)

    return averaged_wedge, wedge_x_grid, wedge_y_grid, float(global_min), float(global_max)


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def plot_all_averaged_wedges(
    planes: Sequence[str],
    data_type: str,
    average_window: Tuple[int, int],
    stl_file: str,
    grid_radius: float = 0.02,
    grid_resolution: int = 100,
) -> None:
    """
    Plot wedge-averaged fields for all planes with STL wedge overlays.

    Parameters
    ----------
    planes : list of str
        Plane IDs to plot (keys in PLANES).
    data_type : str
        One of "p", "ux", "uy", "uz", "radial", "azimuthal".
    average_window : (start, end)
        Time-averaging window for the field.
    stl_file : str
        Path to STL geometry file.
    grid_radius : float
        Radial extent of wedge grid for interpolation.
    grid_resolution : int
        Resolution of wedge grid.
    """
    num_planes = len(planes)
    fig, axes = plt.subplots(
        1,
        num_planes,
        figsize=(4.0 * num_planes, 4.0),
        constrained_layout=False,
    )
    if num_planes == 1:
        axes = [axes]

    subplot_labels = [chr(97 + i) + "." for i in range(num_planes)]

    # Load and rotate STL mesh by 30° around z-axis
    stl_mesh = mesh.Mesh.from_file(stl_file)
    angle_rad = np.deg2rad(30.0)
    rotation_matrix = np.array(
        [
            [np.cos(angle_rad), -np.sin(angle_rad), 0.0],
            [np.sin(angle_rad), np.cos(angle_rad), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    stl_mesh.vectors = np.dot(stl_mesh.vectors, rotation_matrix.T)

    wedges_data = []
    all_wedges: List[np.ndarray] = []

    # Collect all wedge fields first to determine global color limits
    for plane_id in planes:
        plane_points, averaged_field, _ = process_snapshot(
            plane_id, data_type, average_window=average_window
        )
        averaged_wedge, wx, wy, _, _ = compute_averaged_wedge_field(
            plane_points, averaged_field, grid_radius, grid_resolution
        )

        z_cut = PLANES.get(plane_id)
        stl_wedge_1 = None
        if z_cut is not None:
            segments = slice_mesh_at_z(stl_mesh, z_cut)
            stl_wedges = divide_into_wedges(segments)
            if stl_wedges:
                stl_wedge_1 = stl_wedges[0]

        wedges_data.append((plane_id, averaged_wedge, wx, wy, stl_wedge_1))

        if averaged_wedge is not None:
            all_wedges.append(averaged_wedge)

    if all_wedges:
        concatenated = np.concatenate([w.flatten() for w in all_wedges])
        global_min = float(np.nanmin(concatenated))
        global_max = float(np.nanmax(concatenated))
    else:
        global_min, global_max = 0.0, 1.0

    for i, (plane_id, averaged_wedge, wx, wy, stl_wedge_1) in enumerate(wedges_data):
        ax = axes[i]
        ax.text(
            0.05,
            0.95,
            subplot_labels[i],
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            va="top",
        )

        if averaged_wedge is None:
            ax.set_aspect("equal")
            ax.set_xlim([-grid_radius, grid_radius])
            ax.set_ylim([-grid_radius, grid_radius])
            ax.axis("off")
            continue

        cs = ax.contourf(
            wx,
            wy,
            averaged_wedge,
            levels=50,
            cmap="twilight",
            vmin=global_min,
            vmax=global_max,
        )
        ax.set_aspect("equal")
        ax.set_xlim([-grid_radius, grid_radius])
        ax.set_ylim([-grid_radius, grid_radius])
        ax.axis("off")

        # Overlay STL wedge outline
        if stl_wedge_1:
            for seg in stl_wedge_1:
                ax.plot(
                    [seg[0][0], seg[1][0]],
                    [seg[0][1], seg[1][1]],
                    "w",
                    linewidth=2,
                )

        # Colorbar under each subplot
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("bottom", size="3%", pad=0.01)
        cbar = fig.colorbar(cs, cax=cax, orientation="horizontal")
        cbar.locator = ticker.MaxNLocator(nbins=5)
        cbar.update_ticks()
        cax.xaxis.set_label_position("bottom")
        cax.xaxis.set_ticks_position("bottom")

        ax.set_title(f"Plane {plane_id}", fontsize=10)

    plt.tight_layout(pad=0.1)
    plt.show()


def plot_plane_multivariables(
    plane_id: str,
    average_window: Tuple[int, int],
    stl_file: str,
    grid_radius: float = 0.02,
    grid_resolution: int = 100,
) -> None:
    """
    Plot multiple variables (azimuthal, radial, uz, p) for a single plane.

    Parameters
    ----------
    plane_id : str
        Plane ID to plot.
    average_window : (start, end)
        Averaging window for time-averaged fields.
    stl_file : str
        Path to STL file.
    grid_radius : float
        Wedge grid radius.
    grid_resolution : int
        Wedge grid resolution.
    """
    variables = ["azimuthal", "radial", "uz", "p"]
    num_vars = len(variables)

    fig, axes = plt.subplots(1, num_vars, figsize=(5.0 * num_vars, 5.0))
    if num_vars == 1:
        axes = [axes]

    subplot_labels = [chr(97 + i) + "." for i in range(num_vars)]

    # Load and rotate STL
    stl_mesh = mesh.Mesh.from_file(stl_file)
    angle_rad = np.deg2rad(30.0)
    rotation_matrix = np.array(
        [
            [np.cos(angle_rad), -np.sin(angle_rad), 0.0],
            [np.sin(angle_rad), np.cos(angle_rad), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    stl_mesh.vectors = np.dot(stl_mesh.vectors, rotation_matrix.T)

    z_cut = PLANES[plane_id]
    segments = slice_mesh_at_z(stl_mesh, z_cut)
    stl_wedges = divide_into_wedges(segments)
    stl_wedge_1 = stl_wedges[0] if stl_wedges else None

    wedges_data = []

    for var in variables:
        if var in ("radial", "azimuthal"):
            plane_points, ux_data = process_plane_and_type(plane_id, "ux")
            _, uy_data = process_plane_and_type(plane_id, "uy")

            if ux_data.size == 0 or uy_data.size == 0 or not plane_points:
                wedges_data.append((None, None, None, None, None, var))
                continue

            start, end = average_window
            radial_list: List[np.ndarray] = []
            azimuthal_list: List[np.ndarray] = []

            for i in range(start, end):
                u_r, u_theta = compute_radial_and_azimuthal_velocity(
                    plane_points, ux_data[i], uy_data[i]
                )
                radial_list.append(u_r)
                azimuthal_list.append(u_theta)

            averaged_radial = np.mean(radial_list, axis=0) / TIP_SPEED
            averaged_azimuthal = np.mean(azimuthal_list, axis=0) / TIP_SPEED
            values = averaged_radial if var == "radial" else averaged_azimuthal

        else:
            plane_points, data = process_plane_and_type(plane_id, var)
            if data.size == 0 or not plane_points:
                wedges_data.append((None, None, None, None, None, var))
                continue

            start, end = average_window
            averaged_field = compute_average_field(data, start, end)

            if var == "uz":
                values = averaged_field / TIP_SPEED
            elif var == "p":
                mean_p = abs(np.mean(averaged_field))
                values = averaged_field / mean_p if mean_p != 0.0 else averaged_field
            else:
                values = averaged_field

        averaged_wedge, wx, wy, vmin, vmax = compute_averaged_wedge_field(
            plane_points, values, grid_radius, grid_resolution
        )
        wedges_data.append((averaged_wedge, wx, wy, vmin, vmax, var))

    for i, (averaged_wedge, wx, wy, vmin, vmax, var) in enumerate(wedges_data):
        ax = axes[i]
        ax.text(
            0.05,
            0.95,
            subplot_labels[i],
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            va="top",
            ha="left",
            color="black",
        )

        cs = None
        if averaged_wedge is not None:
            # Optional trimming in angle to avoid interpolation artifacts near wedge edges
            trim_angle = np.deg2rad(2.0)
            wedge_width = np.pi / 3.0
            angles = np.arctan2(wy, wx)
            mask = (angles >= trim_angle) & (angles <= (wedge_width - trim_angle))
            averaged_wedge_masked = np.where(mask, averaged_wedge, np.nan)

            cs = ax.contourf(
                wx,
                wy,
                averaged_wedge_masked,
                levels=100,
                cmap="twilight",
                vmin=vmin,
                vmax=vmax,
            )

        ax.set_title(f"{var} (plane {plane_id})")
        ax.set_aspect("equal", "box")
        ax.set_xlim([-grid_radius, grid_radius])
        ax.set_ylim([-grid_radius, grid_radius])
        ax.grid(False)

        if stl_wedge_1:
            for segment in stl_wedge_1:
                ax.plot(
                    [segment[0][0], segment[1][0]],
                    [segment[0][1], segment[1][1]],
                    "w",
                    linewidth=2,
                )

        ax.axis("off")

        if cs is not None:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("bottom", size="2%", pad=0.35)
            cbar = fig.colorbar(cs, cax=cax, orientation="horizontal")
            cbar.locator = ticker.MaxNLocator(nbins=5)
            cbar.update_ticks()
            cax.xaxis.set_ticks_position("bottom")
            cax.xaxis.set_label_position("bottom")

    fig.patch.set_edgecolor("black")
    fig.patch.set_linewidth(2)
    plt.subplots_adjust(wspace=0.25)
    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # Adjust these paths/parameters before running
    STL_FILE = "/path/to/claires.stl"

    
    averaging_window = (0, 29)
    selected_data_type = "uz"  # "radial", "azimuthal", "ux", "uy", "p", "uz"

    plot_all_averaged_wedges(
        planes=list(PLANES.keys()),
        data_type=selected_data_type,
        average_window=averaging_window,
        stl_file=STL_FILE,
        grid_radius=0.02,
    )

    
    plot_plane_multivariables(
        plane_id="4",
        average_window=averaging_window,
        stl_file=STL_FILE,
        grid_radius=0.02,
    )
