"""Quick debug: print substrate geometry and groove positions."""
import sys
sys.path.insert(0, '.')
import numpy as np
from config import SimConfig
from mesh import IcosphereMesh
from utils import bond_length_stats
from substrate import generate_grooved_substrate

cfg = SimConfig()
mesh = IcosphereMesh(cfg.R_cell, cfg.ico_subdiv_cell)
_, _, a = bond_length_stats(mesh.vertices, mesh.edges)
cfg.scale_by_a(a)

sub_v, sub_f = generate_grooved_substrate(cfg)

print(f"a = {a:.4f}")
print(f"w (scaled) = {cfg.w:.4f}")
print(f"h (scaled) = {cfg.h:.4f}")
print(f"Substrate x range: [{sub_v[:,0].min():.2f}, {sub_v[:,0].max():.2f}]")
print(f"Substrate y range: [{sub_v[:,1].min():.2f}, {sub_v[:,1].max():.2f}]")
print(f"Substrate z range: [{sub_v[:,2].min():.2f}, {sub_v[:,2].max():.2f}]")

# Find groove floors
z_min = sub_v[:, 2].min()
floor_mask = sub_v[:, 2] < (z_min + 0.1 * cfg.h)
floor_verts = sub_v[floor_mask]
print(f"\nGroove floor vertices: {len(floor_verts)}")
print(f"Floor x values (unique): {np.unique(np.round(floor_verts[:,0], 2))}")

# Find ridge tops
ridge_mask = sub_v[:, 2] > -0.01
ridge_verts = sub_v[ridge_mask]
print(f"\nRidge x values (unique): {np.unique(np.round(ridge_verts[:,0], 2))}")

# Groove nearest x=0
floor_x = floor_verts[:, 0]
near_zero = floor_x[np.abs(floor_x) < cfg.w * 1.5]
if len(near_zero) > 0:
    print(f"\nGroove nearest x=0: x in [{near_zero.min():.2f}, {near_zero.max():.2f}], center={near_zero.mean():.2f}")
