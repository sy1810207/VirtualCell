"""
VirtualCell v0.85 — Grooved Substrate Generation
Periodic grooves along y-axis with triangulated surface.
"""
import numpy as np
from config import SimConfig


def generate_grooved_substrate(config: SimConfig, n_periods=8, ny=10, nx_per_period=16):
    """
    Generate a rigid grooved substrate with sinusoidal cross-section.

    Geometry (cross-section in xz-plane, grooves run along y):
      z(x) = -(h/2) * (1 - cos(2π (x - x_start) / period)),  period = 2w
      - Peaks (ridge tops) at z = 0, spaced every 2w in x
      - Troughs (groove floors) at z = -h, located mid-period
      - Smooth sinusoidal walls (no flat ridge/floor)
      - Invariant along y

    Parameters
    ----------
    config : SimConfig
    n_periods : int — number of groove periods
    ny : int — number of subdivisions along y-axis
    nx_per_period : int — number of x-subdivisions per period (mesh resolution)

    Returns
    -------
    vertices : (N, 3)
    faces : (M, 3)
    """
    w = config.w
    h = config.h
    R = config.R_cell

    period = 2.0 * w
    x_total = n_periods * period
    x_start = -x_total / 2.0

    y_min = -R * 1.5
    y_max = R * 1.5
    ys = np.linspace(y_min, y_max, ny + 1)

    total_nx = n_periods * nx_per_period
    xs = np.linspace(x_start, x_start + x_total, total_nx + 1)
    if h > 1e-6:
        zs = -(h / 2.0) * (1.0 - np.cos(2.0 * np.pi * (xs - x_start) / period))
    else:
        zs = np.zeros_like(xs)

    X, Y = np.meshgrid(xs, ys)
    Z = np.broadcast_to(zs, X.shape)
    vertices = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1).astype(float)

    nx1 = total_nx + 1
    faces = []
    for j in range(ny):
        for i in range(total_nx):
            a = j * nx1 + i
            b = j * nx1 + (i + 1)
            c = (j + 1) * nx1 + (i + 1)
            d = (j + 1) * nx1 + i
            faces.append([a, b, c])
            faces.append([a, c, d])

    return vertices, np.array(faces, dtype=int)


def generate_flat_substrate(config: SimConfig, nx=20, ny=20):
    """Flat substrate at z=0 for testing."""
    R = config.R_cell * 1.5
    xs = np.linspace(-R, R, nx + 1)
    ys = np.linspace(-R, R, ny + 1)

    vertices = []
    faces = []
    idx_grid = np.zeros((nx + 1, ny + 1), dtype=int)
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            idx_grid[i, j] = len(vertices)
            vertices.append([x, y, 0.0])

    for i in range(nx):
        for j in range(ny):
            a = idx_grid[i, j]
            b = idx_grid[i + 1, j]
            c = idx_grid[i + 1, j + 1]
            d = idx_grid[i, j + 1]
            faces.append([a, b, c])
            faces.append([a, c, d])

    return np.array(vertices, dtype=float), np.array(faces, dtype=int)
