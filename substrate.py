"""
VirtualCell v0.85 — Grooved Substrate Generation
Periodic grooves along y-axis with triangulated surface.
"""
import numpy as np
from config import SimConfig


def generate_grooved_substrate(config: SimConfig, n_periods=8, ny=10):
    """
    Generate a rigid grooved substrate with trapezoidal cross-section.

    Geometry (cross-section in xz-plane, grooves run along y):
      - Ridge top at z = 0, width = w_top
      - Sloped walls (trapezoid angle)
      - Groove floor at z = -h, width = w_floor
      - Period = 2w (one ridge + one groove, measured at top)

    Parameters
    ----------
    config : SimConfig
    n_periods : int — number of groove periods
    ny : int — number of subdivisions along y-axis

    Returns
    -------
    vertices : (N, 3)
    faces : (M, 3)
    """
    w = config.w
    h = config.h
    R = config.R_cell

    # Trapezoid slope: walls at ~60° from horizontal
    # slope_dx = horizontal offset from top edge to bottom edge
    slope_dx = h * 0.4  # controls wall angle (smaller = steeper)

    # y extent: cover cell diameter with margin
    y_min = -R * 1.5
    y_max = R * 1.5
    ys = np.linspace(y_min, y_max, ny + 1)

    # x extent: n_periods full periods, centered at 0
    period = 2.0 * w
    x_total = n_periods * period
    x_start = -x_total / 2.0

    vertices = []
    faces = []

    def add_vertex(x, y, z):
        idx = len(vertices)
        vertices.append([x, y, z])
        return idx

    def add_quad(a, b, c, d):
        """Add two triangles for a quad (a,b,c,d) in CCW order."""
        faces.append([a, b, c])
        faces.append([a, c, d])

    # For each y-strip between ys[j] and ys[j+1]
    for j in range(ny):
        y0 = ys[j]
        y1 = ys[j + 1]

        for p in range(n_periods):
            x_base = x_start + p * period

            # Ridge top: x_base to x_base + w, at z = 0
            x_r0 = x_base
            x_r1 = x_base + w
            a = add_vertex(x_r0, y0, 0.0)
            b = add_vertex(x_r1, y0, 0.0)
            c = add_vertex(x_r1, y1, 0.0)
            d = add_vertex(x_r0, y1, 0.0)
            add_quad(a, b, c, d)

            if h > 1e-6:
                # Left slope: from (x_r1, z=0) down to (x_r1 + slope_dx, z=-h)
                x_left_bot = x_r1 + slope_dx
                e = add_vertex(x_left_bot, y0, -h)
                f = add_vertex(x_left_bot, y1, -h)
                add_quad(b, e, f, c)

                # Groove floor: from x_left_bot to x_right_bot at z=-h
                x_g1 = x_base + 2 * w  # top edge of right slope
                x_right_bot = x_g1 - slope_dx
                # Ensure floor width is positive
                x_right_bot = max(x_right_bot, x_left_bot + 0.1 * w)
                g = add_vertex(x_right_bot, y0, -h)
                hv = add_vertex(x_right_bot, y1, -h)
                add_quad(e, g, hv, f)

                # Right slope: from (x_right_bot, z=-h) up to (x_g1, z=0)
                i = add_vertex(x_g1, y0, 0.0)
                jv = add_vertex(x_g1, y1, 0.0)
                add_quad(g, i, jv, hv)

    vertices = np.array(vertices, dtype=float)
    faces = np.array(faces, dtype=int)
    return vertices, faces


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
